"""Docker backend logic that needs no Docker: isolation flags, classification, request validation."""
import json
import unittest

from app.sandbox._proc import BoundedResult
from app.sandbox.docker_backend import (EXPECTED_RUNNER_VERSION, DockerBackend, build_docker_command, classify, host_stdout_cap,
                                         parse_envelope)
from app.sandbox.interface import (SandboxInfrastructureError, SandboxLimits, SandboxRequest,
                                   SandboxRequestError)

L = SandboxLimits(memory_mb=256, pids=32, tmp_mb=16, max_result_bytes=10_000, max_stdout_bytes=1000)


def bounded(stdout=b"", stderr=b"", rc=0, timed_out=False, over=False):
    return BoundedResult(rc, stdout, stderr, timed_out, over, False, 1.5)


def env(kind="ok", **kw):
    base = {"v": 1, "runner_version": EXPECTED_RUNNER_VERSION, "kind": kind, "detail": "", "error": None, "result": {"result": 1},
            "stdout": "", "stderr": "", "exec_seconds": 0.25}
    base.update(kw)
    return json.dumps(base).encode()


def run(b, oom=False):
    return classify(b, oom=oom, limits=L, code_sha="c" * 64, inputs_sha={})


class Flags(unittest.TestCase):
    def cmd(self):
        return build_docker_command(docker_bin="docker", image="img:1", name="n1", limits=L)

    def pair(self, c, flag):
        return c[c.index(flag) + 1]

    def test_isolation_flags_present(self):
        c = self.cmd()
        self.assertEqual(self.pair(c, "--network"), "none")
        self.assertIn("--read-only", c)
        self.assertEqual(self.pair(c, "--cap-drop"), "ALL")
        self.assertEqual(self.pair(c, "--security-opt"), "no-new-privileges")
        self.assertEqual(self.pair(c, "--user"), "65534:65534")
        self.assertEqual(self.pair(c, "--pull"), "never")
        self.assertEqual(self.pair(c, "--pids-limit"), "32")
        self.assertEqual(self.pair(c, "--memory"), "256m")
        self.assertEqual(self.pair(c, "--memory-swap"), "256m")      # no swap
        mount, _, options = self.pair(c, "--tmpfs").partition(":")
        self.assertEqual(mount, "/work")
        self.assertTrue({"rw", "noexec", "nosuid", "nodev", "size=16m"} <= set(options.split(",")))
        self.assertIn("--init", c)
        self.assertEqual(c[-1], "img:1")

    def test_work_tmpfs_is_owned_by_the_sandbox_user_and_private(self):
        c = self.cmd()
        opts = dict(o.split("=", 1) if "=" in o else (o, True) for o in self.pair(c, "--tmpfs").split(":", 1)[1].split(","))
        user = self.pair(c, "--user")
        self.assertEqual(f"{opts['uid']}:{opts['gid']}", user)             # the process owns what it must write
        self.assertEqual(opts["mode"], "0700")                              # and nobody else can use it
        self.assertEqual(user, "65534:65534")                               # still unprivileged

    def test_nothing_that_widens_the_sandbox(self):
        tokens = self.cmd()
        for forbidden in ("--privileged", "-v", "--volume", "--mount", "--cap-add", "--pid", "--ipc",
                          "--uts", "--device", "--env-file", "--rm", "--userns", "--cgroupns"):
            self.assertNotIn(forbidden, tokens, forbidden)       # exact flag tokens, not substrings
        self.assertFalse(any("docker.sock" in tok for tok in tokens))
        self.assertNotEqual(tokens[tokens.index("--network") + 1], "host")

    def test_only_fixed_env_vars_are_passed(self):
        c = self.cmd()
        envs = [c[i + 1] for i, a in enumerate(c) if a == "--env"]
        self.assertEqual(sorted(envs), ["HOME=/work", "PYTHONDONTWRITEBYTECODE=1", "TMPDIR=/work"])
        self.assertTrue(all("=" in e and "SECRET" not in e.upper() and "KEY" not in e.upper() for e in envs))


class Classification(unittest.TestCase):
    def test_ok(self):
        r = run(bounded(env()))
        self.assertEqual((r.status, r.result_json, r.duration_s), ("ok", {"result": 1}, 0.25))

    def test_error_envelope_merges_traceback_into_stderr(self):
        r = run(bounded(env("error", detail="ValueError", error="Traceback...boom", result=None, stderr="s")))
        self.assertEqual((r.status, r.detail), ("error", "ValueError"))
        self.assertIn("boom", r.stderr); self.assertIn("s", r.stderr)

    def test_timeout_and_overflow_beat_any_envelope(self):
        self.assertEqual(run(bounded(env(), timed_out=True)).status, "timeout")
        self.assertEqual(run(bounded(env(), over=True)).status, "output_too_large")

    def test_oom_only_without_a_valid_envelope(self):
        self.assertEqual(run(bounded(b"", rc=137), oom=True).status, "oom")
        self.assertEqual(run(bounded(env()), oom=True).status, "ok")

    def test_signal_exits(self):
        self.assertEqual(run(bounded(b"", rc=152)).detail, "cpu_limit")
        self.assertEqual(run(bounded(b"", rc=-24)).detail, "cpu_limit")
        self.assertEqual(run(bounded(b"", rc=137)).status, "killed")
        self.assertEqual(run(bounded(b"", rc=-9)).status, "killed")

    def test_no_envelope_is_never_ok(self):
        for rc in (0, 1, 2):
            self.assertEqual(run(bounded(b"garbage", rc=rc)).status, "error")

    def test_forged_or_malformed_envelopes_are_not_trusted(self):
        for raw in (b"{}", b'{"v":2,"kind":"ok"}', b'{"v":1,"kind":"weird"}', b"[1]", b"\xff\xfe"):
            self.assertIsNone(parse_envelope(raw))
        # ok envelope whose result lacks the required key
        r = run(bounded(env(result={"answer": 1})))
        self.assertEqual((r.status, r.detail), ("error", "bad_return_shape"))
        r = run(bounded(env(result="not a dict")))
        self.assertEqual(r.status, "error")

    def test_host_rechecks_result_size_independently_of_the_runner(self):
        r = run(bounded(env(result={"result": "x" * 50_000})))
        self.assertEqual((r.status, r.detail), ("output_too_large", "result_too_large"))

    def test_setup_error_and_stale_image_are_infrastructure_errors(self):
        with self.assertRaises(SandboxInfrastructureError) as cm:
            run(bounded(env("setup_error", detail="sandbox_setup_failed", error="PermissionError: uid=65534", result=None)))
        self.assertIn("PermissionError", str(cm.exception))
        stale = json.dumps({"v": 1, "kind": "ok", "result": {"result": 1}, "stdout": "", "stderr": ""}).encode()
        with self.assertRaises(SandboxInfrastructureError) as cm:
            run(bounded(stale))                                             # envelope from an old image: no runner_version
        self.assertIn("docker build", str(cm.exception))

    def test_infrastructure_failure_is_raised_not_reported_as_code_failure(self):
        with self.assertRaises(SandboxInfrastructureError):
            run(bounded(b"", stderr=b"docker: Cannot connect to the Docker daemon", rc=125))
        # exit 125 from user code (docker stderr empty) is an ordinary error
        self.assertEqual(run(bounded(b"", rc=125)).status, "error")

    def test_stdout_cap_formula_covers_worst_case_escaping(self):
        self.assertGreaterEqual(host_stdout_cap(L), L.max_result_bytes + 2 * 6 * L.max_stdout_bytes)


class Validation(unittest.TestCase):
    def test_bad_requests_never_reach_docker(self):
        bad = [
            SandboxRequest(code=""),
            SandboxRequest(code="x", inputs={"../etc/passwd": b""}),
            SandboxRequest(code="x", inputs={"a/b": b""}),
            SandboxRequest(code="x", inputs={"main.py": b""}),
            SandboxRequest(code="x", inputs={"a..b": b""}),
            SandboxRequest(code="x", inputs={".hidden": b""}),
            SandboxRequest(code="x", inputs={"ok": "not bytes"}),
            SandboxRequest(code="x", limits=SandboxLimits(memory_mb=0)),
            SandboxRequest(code="x" * 300_000),
        ]
        backend = DockerBackend(docker_bin="definitely-not-docker")
        for request in bad:
            with self.assertRaises(SandboxRequestError):
                backend.run(request)          # validation fires before any docker call

    def test_missing_docker_is_an_infrastructure_error(self):
        with self.assertRaises(SandboxInfrastructureError):
            DockerBackend(docker_bin="definitely-not-docker").run(SandboxRequest(code="def solve(r,e): return {'result':1}"))

    def test_missing_image_error_tells_you_how_to_build_it(self):
        class Fake(DockerBackend):
            def _docker(self, args, timeout=20):
                import subprocess
                return subprocess.CompletedProcess(args, 1, b"", b"no such image")
        with self.assertRaises(SandboxInfrastructureError) as cm:
            Fake().ensure_image()
        self.assertIn("docker build", str(cm.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
