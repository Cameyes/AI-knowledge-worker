"""Supervisor/worker split: one authoritative envelope, no inherited envelope fd, no forged setup_error,
descendants cannot continue runner control flow, worker deaths are classified.

Tests using run_contract run on the contract backend (local subprocess on POSIX, the real container on
Windows). Tests marked RAW run runner.py directly and therefore need a POSIX host.
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import unittest

from app.sandbox import runner
from app.sandbox.docker_backend import EXPECTED_RUNNER_VERSION, parse_envelope
from app.sandbox.interface import SandboxInfrastructureError, SandboxLimits
from app.sandbox.tests._harness import RUNNER, run_contract, run_runner_raw, unavailable_reason

POSIX = os.name == "posix"
ROOT = POSIX and os.geteuid() == 0
LIM = SandboxLimits(wall_seconds=6, cpu_seconds=5)


def setUpModule():
    reason = unavailable_reason()
    if reason:
        raise unittest.SkipTest(reason)


def count_envelopes(raw: bytes):
    text, dec, i, found = raw.decode("utf-8", "replace"), json.JSONDecoder(), 0, []
    while i < len(text):
        try:
            obj, i = dec.raw_decode(text, i)
        except ValueError:
            break
        found.append(obj)
    return found


def is_alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except (OSError, IndexError):
        return False


# ---------------------------------------------------------------------------- user-code snippets
FALL_THROUGH = '''
import os, time
def solve(records, extra):
    pid = os.fork()
    if pid != 0:
        time.sleep(0.4)             # let the child reach the end of solve() first
    return {"result": "parent" if pid != 0 else "child"}
'''
FD_AUDIT = '''
import os
def solve(records, extra):
    open_fds = []
    for fd in range(3, 1024):
        try:
            os.fstat(fd); open_fds.append(fd)
        except OSError:
            pass
    return {"result": {"open_fds": open_fds}}
'''
FD_FORGE = '''
import os, json
def solve(records, extra):
    fake = json.dumps({"v": 1, "runner_version": %r, "kind": "setup_error", "detail": "sandbox_setup_failed",
                       "error": "forged", "result": None, "stdout": "", "stderr": "", "exec_seconds": 0.0}).encode()
    for fd in range(3, 1024):
        try:
            os.write(fd, fake)
        except OSError:
            pass
    return {"result": "legit"}
''' % EXPECTED_RUNNER_VERSION
FRAME_FORGE = '''
import os, json, struct
def solve(records, extra):
    body = %r.encode()
    for fd in range(3, 1024):
        try:
            os.fstat(fd)
            os.write(fd, struct.pack(">Q", len(body)) + body)
        except OSError:
            pass
    os._exit(0)
'''
LEFTOVER = '''
import os, time
def solve(records, extra):
    pid = os.fork()
    if pid == 0:
        time.sleep(60); os._exit(0)
    return {"result": pid}
'''
FIFO_SWAP = '''
import os
def solve(records, extra):
    p = os.path.join(os.environ.get("SANDBOX_WORKDIR", "/work"), ".stdout")
    os.unlink(p); os.mkfifo(p)
    return {"result": "swapped"}
'''
SYMLINK_SWAP = '''
import os
def solve(records, extra):
    work = os.environ.get("SANDBOX_WORKDIR", "/work")
    p = os.path.join(work, ".stdout")
    os.unlink(p); os.symlink("/etc/hostname", p)
    return {"result": "swapped"}
'''
FORK_BOMB_CAPPED = '''
import os, resource
resource.setrlimit(resource.RLIMIT_NPROC, (30, 30))
def solve(records, extra):
    while True:
        os.fork()
'''


class SingleAuthoritativeEnvelope(unittest.TestCase):
    def test_fall_through_child_cannot_replace_the_parents_result(self):
        r = run_contract(FALL_THROUGH, limits=LIM)
        self.assertEqual((r.status, r.result_json["result"]), ("ok", "parent"), r)   # child's frame arrives first

    @unittest.skipUnless(POSIX, "RAW: needs a POSIX host")
    def test_raw_output_contains_exactly_one_envelope(self):
        with tempfile.TemporaryDirectory() as wd:
            b = run_runner_raw(FALL_THROUGH, workdir=wd, limits=LIM)
        envelopes = count_envelopes(b.stdout)
        self.assertEqual(len(envelopes), 1, b.stdout[:300])
        self.assertIsNotNone(parse_envelope(b.stdout))

    @unittest.skipUnless(ROOT, "RAW: needs root to drop to an unprivileged uid so RLIMIT_NPROC applies")
    def test_fork_bomb_under_a_process_cap_yields_one_attributable_envelope(self):
        payload = json.dumps({"code": FORK_BOMB_CAPPED, "inputs": {}, "limits": {
            "cpu_seconds": 10, "max_result_bytes": 10**6, "max_stdout_bytes": 64000}}).encode()
        root = pathlib.Path(tempfile.mkdtemp()); os.chmod(root, 0o755)
        work = root / "work"; work.mkdir(); os.chown(work, 65534, 65534); os.chmod(work, 0o700)
        try:
            p = subprocess.run([sys.executable, "-I", "-B", str(RUNNER)], input=payload, capture_output=True,
                               user=65534, group=65534, timeout=30,
                               env={"SANDBOX_WORKDIR": str(work), "PATH": os.environ["PATH"]})
        finally:
            subprocess.run(["pkill", "-9", "-u", "65534"], capture_output=True)
        envelopes = count_envelopes(p.stdout)
        self.assertEqual(len(envelopes), 1, f"{len(envelopes)} envelopes")            # before the split: 30
        env = parse_envelope(p.stdout)
        self.assertIsNotNone(env)
        self.assertEqual((env["kind"], env["detail"]), ("error", "BlockingIOError"))   # the cap actually stopped it

    @unittest.skipUnless(POSIX, "RAW: needs /proc and a POSIX host")
    def test_descendants_are_killed_when_the_run_ends(self):
        r = run_contract(LEFTOVER, limits=LIM)
        self.assertEqual(r.status, "ok", r)
        pid = r.result_json["result"]
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and is_alive(pid):
            time.sleep(0.05)
        self.assertFalse(is_alive(pid), "a forked descendant outlived the run")


class EnvelopeIsUnreachableFromUserCode(unittest.TestCase):
    def test_worker_inherits_no_fds_except_the_result_pipe(self):
        r = run_contract(FD_AUDIT, limits=LIM)
        self.assertEqual(r.status, "ok", r)
        self.assertLessEqual(len(r.result_json["result"]["open_fds"]), 1, r.result_json)

    def test_writing_a_forged_setup_error_to_every_fd_cannot_raise_an_infrastructure_error(self):
        try:
            r = run_contract(FD_FORGE, limits=LIM)
        except SandboxInfrastructureError as exc:                                        # the pre-split behavior
            self.fail(f"user code raised an infrastructure error: {exc}")
        self.assertIn(r.status, {"ok", "error"}, r)
        self.assertIn(r.detail, {"", "bad_worker_frame"}, r)       # it only corrupted its own result channel

    def test_worker_cannot_produce_setup_error(self):
        body = json.dumps({"kind": "setup_error", "detail": "sandbox_setup_failed", "error": "forged", "result": None})
        try:
            r = run_contract(FRAME_FORGE % body, limits=LIM)
        except SandboxInfrastructureError as exc:
            self.fail(f"worker produced setup_error: {exc}")
        self.assertEqual((r.status, r.detail), ("error", "bad_worker_frame"), r)

    def test_documented_limit_a_forged_ok_frame_has_only_the_authority_of_solves_return_value(self):
        forged = json.dumps({"kind": "ok", "detail": "", "error": None, "result": {"result": 123}})
        r = run_contract(FRAME_FORGE % forged, limits=LIM)
        self.assertEqual((r.status, r.result_json), ("ok", {"result": 123}), r)          # same as `return {"result": 123}`

    def test_user_code_cannot_hang_the_supervisor_by_swapping_capture_files(self):
        r = run_contract(FIFO_SWAP, limits=SandboxLimits(wall_seconds=6, cpu_seconds=5))
        self.assertEqual((r.status, r.result_json["result"]), ("ok", "swapped"), r)       # a FIFO would block a naive read
        r = run_contract(SYMLINK_SWAP, limits=LIM)
        self.assertEqual(r.status, "ok", r)
        self.assertEqual(r.stdout, "")                                                    # the symlink is not followed


class WorkerTerminationIsClassified(unittest.TestCase):
    def test_sigkill(self):
        r = run_contract("import os, signal\ndef solve(r, e):\n    os.kill(os.getpid(), signal.SIGKILL)\n", limits=LIM)
        self.assertEqual((r.status, r.detail), ("killed", "sigkill"), r)

    def test_sigxcpu_is_the_cpu_limit(self):
        r = run_contract("import os, signal\ndef solve(r, e):\n    os.kill(os.getpid(), signal.SIGXCPU)\n", limits=LIM)
        self.assertEqual((r.status, r.detail), ("timeout", "cpu_limit"), r)

    def test_real_cpu_exhaustion_is_the_cpu_limit(self):
        r = run_contract("def solve(r, e):\n    while True:\n        pass\n",
                         limits=SandboxLimits(wall_seconds=30, cpu_seconds=1))
        self.assertEqual((r.status, r.detail), ("timeout", "cpu_limit"), r)

    def test_other_signals_name_the_signal(self):
        r = run_contract("import os, signal\ndef solve(r, e):\n    os.kill(os.getpid(), signal.SIGSEGV)\n", limits=LIM)
        self.assertEqual(r.status, "error", r)
        self.assertIn("SIGSEGV", r.stderr)

    def test_nonzero_exit_without_a_result_is_attributable(self):
        r = run_contract("import os\ndef solve(r, e):\n    os._exit(3)\n", limits=LIM)
        self.assertEqual((r.status, r.detail), ("error", "worker_exit_nonzero"), r)
        self.assertIn("status 3", r.stderr)

    def test_after_a_kill_the_next_run_is_healthy(self):
        run_contract("import os, signal\ndef solve(r, e):\n    os.kill(os.getpid(), signal.SIGKILL)\n", limits=LIM)
        r = run_contract("def solve(r, e):\n    return {'result': 42}\n", limits=LIM)
        self.assertEqual((r.status, r.result_json["result"]), ("ok", 42))


class FrameProtocol(unittest.TestCase):
    """Pure functions: the frame the supervisor will and will not accept."""

    def frame(self, obj):
        return runner._encode_frame(obj)

    def test_parse_states(self):
        good = self.frame({"kind": "ok", "result": {"result": 1}})
        self.assertEqual(runner._parse_frame(b"", 1000)[0], "need_more")
        self.assertEqual(runner._parse_frame(good[:5], 1000)[0], "need_more")
        self.assertEqual(runner._parse_frame(good[:-1], 1000)[0], "need_more")
        self.assertEqual(runner._parse_frame(good, 1000)[0], "ok")
        self.assertEqual(runner._parse_frame(good + b"trailing", 1000)[0], "ok")      # a second frame is never read
        self.assertEqual(runner._parse_frame(b'{"v":1,"kind":"ok"}', 1000), ("bad", "frame_too_large"))
        bad_json = runner.FRAME_HEADER.pack(3) + b"xyz"
        self.assertEqual(runner._parse_frame(bad_json, 1000), ("bad", "frame_not_json"))

    def test_validate_accepts_only_the_three_worker_kinds(self):
        ok = {"kind": "ok", "detail": "", "error": None, "result": {"result": 1}}
        self.assertEqual(runner._validate_outcome(ok, 1000)["kind"], "ok")
        for kind in ("setup_error", "weird", None, 5):
            self.assertIsNone(runner._validate_outcome({**ok, "kind": kind}, 1000), kind)
        self.assertEqual(runner._validate_outcome({"kind": "error", "detail": "X", "error": "e"}, 1000)["kind"], "error")

    def test_validate_rejects_malformed_outcomes(self):
        bad = [
            "not a dict", {"kind": "ok", "result": None}, {"kind": "ok", "result": {"answer": 1}},
            {"kind": "ok", "result": {"result": "x" * 5000}},            # over max_result
            {"kind": "ok", "result": {"result": float("nan")}},
            {"kind": "error", "detail": 5}, {"kind": "error", "error": 5},
        ]
        for outcome in bad:
            self.assertIsNone(runner._validate_outcome(outcome, 1000), outcome)

    def test_validate_bounds_strings_and_drops_result_for_non_ok(self):
        out = runner._validate_outcome({"kind": "error", "detail": "d" * 500, "error": "e" * 9000, "result": {"result": 1}}, 1000)
        self.assertEqual((len(out["detail"]), len(out["error"]), out["result"]), (100, runner.MAX_TRACEBACK_CHARS, None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
