"""ADVERSARIAL GATE. Requires Docker and the sandbox image. Skipped (loudly) otherwise.

    docker build -t rag-sandbox:py312-v1 app/sandbox
    python -m unittest app.sandbox.tests.test_adversarial_docker -v

Every test submits hostile code to the real DockerBackend and asserts on what the code could and
could not do. Assertions on exact status names are relaxed to sets only where Docker/OS behavior
legitimately varies; the security property (the attack failed, the host is fine, the next run works)
is never relaxed.
"""
import os
import shutil
import subprocess
import time
import unittest
import uuid

from app.sandbox.docker_backend import DEFAULT_IMAGE, LABEL, DockerBackend
from app.sandbox.interface import SandboxLimits, SandboxRequest


def _docker_ready_unused() -> tuple[bool, str]:
    if not shutil.which("docker"):
        return False, "docker CLI not found"
    try:
        proc = subprocess.run(["docker", "image", "inspect", DEFAULT_IMAGE], capture_output=True, timeout=20)
    except Exception as exc:
        return False, f"docker not usable: {exc}"
    return (proc.returncode == 0, "" if proc.returncode == 0 else f"image {DEFAULT_IMAGE} not built")


from app.sandbox.tests._harness import docker_ready
READY, WHY = docker_ready()
SECRET = f"host-secret-{uuid.uuid4().hex}"
os.environ["SANDBOX_TEST_SECRET"] = SECRET        # lives in the test process env; must never reach the sandbox

QUICK = SandboxLimits(wall_seconds=4, cpu_seconds=10, memory_mb=256, pids=64, tmp_mb=8)

# ----------------------------------------------------------------------------- attack snippets
SANITY = '''
def solve(records, extra):
    return {"result": sum(r["value"] for r in records)}
'''

NETWORK = '''
import os, socket
def solve(records, extra):
    out = {}
    for label, target in (("tcp_ip", ("1.1.1.1", 53)), ("tcp_ip2", ("8.8.8.8", 443))):
        try:
            socket.create_connection(target, timeout=3).close(); out[label] = True
        except Exception as e:
            out[label] = False
    try:
        socket.getaddrinfo("example.com", 80); out["dns"] = True
    except Exception:
        out["dns"] = False
    out["interfaces"] = sorted(os.listdir("/sys/class/net"))
    out["docker_sock"] = os.path.exists("/var/run/docker.sock")
    return {"result": out}
'''

FILESYSTEM = '''
import os
def attempt(fn):
    try:
        fn(); return "allowed"
    except Exception as e:
        return type(e).__name__
def solve(records, extra):
    c = {
        "read_shadow": attempt(lambda: open("/etc/shadow").read()),
        "list_root_home": attempt(lambda: os.listdir("/root")),
        "write_etc": attempt(lambda: open("/etc/pwned", "w").write("x")),
        "write_slash": attempt(lambda: open("/pwned", "w").write("x")),
        "write_tmp": attempt(lambda: open("/tmp/pwned", "w").write("x")),
        "write_usr": attempt(lambda: open("/usr/pwned", "w").write("x")),
        "tamper_runner": attempt(lambda: open("/opt/runner/runner.py", "a").write("#")),
        "write_work": attempt(lambda: open("/work/ok.txt", "w").write("x")),
    }
    mounts = open("/proc/self/mountinfo").read()
    return {"result": {"checks": c, "docker_sock_in_mounts": "docker.sock" in mounts}}
'''

DISK_FILL = '''
def solve(records, extra):
    written, err = 0, None
    try:
        with open("/work/fill.bin", "wb") as f:
            for _ in range(200):
                f.write(b"x" * 1_000_000); f.flush(); written += 1_000_000
    except OSError as e:
        err = e.errno
    return {"result": {"written": written, "errno": err}}
'''

ENVIRONMENT = '''
import glob, os
def solve(records, extra):
    secret = %r
    leaked = [k for k, v in os.environ.items() if secret in v or secret in k or "SECRET" in k.upper()]
    in_procs = []
    for path in glob.glob("/proc/*/environ"):
        try:
            if secret.encode() in open(path, "rb").read():
                in_procs.append(path)
        except Exception:
            pass
    status = dict(l.split(":\\t", 1) for l in open("/proc/self/status").read().splitlines() if ":\\t" in l)
    return {"result": {"leaked": leaked, "in_procs": in_procs, "uid": os.getuid(), "gid": os.getgid(),
                        "no_new_privs": status.get("NoNewPrivs"), "cap_eff": status.get("CapEff", "").strip(),
                        "env_keys": sorted(os.environ)}}
''' % SECRET

SUBPROCESS = '''
import subprocess
def solve(records, extra):
    uid = subprocess.run(["id", "-u"], capture_output=True, text=True).stdout.strip()
    shadow = subprocess.run(["cat", "/etc/shadow"], capture_output=True, text=True)
    return {"result": {"uid": uid, "shadow_rc": shadow.returncode, "shadow_out": shadow.stdout}}
'''

CROSS_RUN_WRITE = '''
import os
def solve(records, extra):
    open("/work/marker.txt", "w").write("state from run 1")
    try:
        open("/tmp/marker.txt", "w").write("x")
    except Exception:
        pass
    return {"result": os.listdir("/work")}
'''
CROSS_RUN_READ = '''
import os
def solve(records, extra):
    return {"result": {"work": sorted(os.listdir("/work")), "marker": os.path.exists("/work/marker.txt"),
                        "tmp_marker": os.path.exists("/tmp/marker.txt")}}
'''

INFINITE_LOOP = 'def solve(records, extra):\n    while True:\n        pass\n'
SLEEP = 'import time\ndef solve(records, extra):\n    time.sleep(600)\n'
MEMORY_BOMB = '''
def solve(records, extra):
    hog = []
    while True:
        hog.append(bytearray(50_000_000))
'''
FORK_BOUNDED = '''
import os, time
def solve(records, extra):
    pids, err = [], None
    for _ in range(1000):
        try:
            pid = os.fork()
        except OSError as e:
            err = type(e).__name__; break
        if pid == 0:
            time.sleep(60); os._exit(0)
        pids.append(pid)
    for p in pids:
        try:
            os.kill(p, 9); os.waitpid(p, 0)
        except Exception:
            pass
    return {"result": {"children": len(pids), "stopped_by": err}}
'''
WORK_MOUNT = '''
import os, stat, subprocess
def solve(records, extra):
    st = os.stat("/work")
    opts = ""
    for line in open("/proc/self/mountinfo"):
        f = line.split()
        if f[4] == "/work":
            opts = f[5]
    script = "/work/run.sh"
    open(script, "w").write("#!/bin/sh\\necho executed\\n"); os.chmod(script, 0o755)
    try:
        subprocess.run([script], capture_output=True, check=True); ran = True
    except Exception as e:
        ran = False
    return {"result": {"uid": st.st_uid, "gid": st.st_gid, "mode": oct(stat.S_IMODE(st.st_mode)), "opts": opts, "exec_ran": ran}}
'''
FORK_BOMB = 'import os\ndef solve(records, extra):\n    while True:\n        os.fork()\n'
STDOUT_BOUNDED = 'def solve(records, extra):\n    print("x" * 8_000_000)\n    return {"result": 1}\n'
STDOUT_FLOOD = 'def solve(records, extra):\n    while True:\n        print("x" * 1000)\n'
HUGE_RESULT = 'def solve(records, extra):\n    return {"result": "x" * 500_000}\n'
BAD_SHAPE = 'def solve(records, extra):\n    return [1, 2, 3]\n'
RAISES = 'def solve(records, extra):\n    raise RuntimeError("boom")\n'

ALL_SNIPPETS = {k: v for k, v in dict(
    SANITY=SANITY, NETWORK=NETWORK, FILESYSTEM=FILESYSTEM, DISK_FILL=DISK_FILL, ENVIRONMENT=ENVIRONMENT,
    SUBPROCESS=SUBPROCESS, CROSS_RUN_WRITE=CROSS_RUN_WRITE, CROSS_RUN_READ=CROSS_RUN_READ,
    INFINITE_LOOP=INFINITE_LOOP, SLEEP=SLEEP, WORK_MOUNT=WORK_MOUNT, MEMORY_BOMB=MEMORY_BOMB, FORK_BOUNDED=FORK_BOUNDED,
    FORK_BOMB=FORK_BOMB, STDOUT_BOUNDED=STDOUT_BOUNDED, STDOUT_FLOOD=STDOUT_FLOOD,
    HUGE_RESULT=HUGE_RESULT, BAD_SHAPE=BAD_SHAPE, RAISES=RAISES).items()}


@unittest.skipUnless(READY, f"Docker adversarial gate NOT RUN: {WHY}")
class Adversarial(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = DockerBackend(startup_grace_seconds=4)
        # Fail fast and loudly: if the sandbox cannot run a trivial program, every "attack failed" assertion
        # below would pass vacuously. A broken sandbox must never look like a secure one.
        probe = cls.backend.run(SandboxRequest(code=SANITY, inputs={"records.jsonl": b'{"value": 1}'}, limits=QUICK))
        if probe.status != "ok":
            raise RuntimeError(f"sandbox cannot run a trivial program ({probe.status}/{probe.detail}): {probe.stderr[-500:]}")

    @classmethod
    def tearDownClass(cls):
        leftovers = subprocess.run(["docker", "ps", "-aq", "--filter", f"label={LABEL}"],
                                   capture_output=True, text=True).stdout.split()
        cls.backend.cleanup_stale()
        assert not leftovers, f"containers leaked: {leftovers}"

    def go(self, code, inputs=None, limits=QUICK):
        return self.backend.run(SandboxRequest(code=code, inputs=inputs or {}, limits=limits))

    def ok(self, code, **kw):
        r = self.go(code, **kw)
        self.assertEqual(r.status, "ok", f"{r.status}/{r.detail}\nSTDERR:\n{r.stderr}")
        return r.result_json["result"]

    # ---------------------------------------------------------------- baseline
    def test_00_sanity_sum(self):
        records = b'{"value": 1}\n{"value": 2}\n{"value": 3}'
        self.assertEqual(self.ok(SANITY, inputs={"records.jsonl": records}), 6)

    # ---------------------------------------------------------------- confinement
    def test_work_tmpfs_is_private_to_the_sandbox_user_and_not_executable(self):
        out = self.ok(WORK_MOUNT)
        self.assertEqual((out["uid"], out["gid"], out["mode"]), (65534, 65534, "0o700"), out)
        for flag in ("rw", "noexec", "nosuid", "nodev"):
            self.assertIn(flag, out["opts"].split(","), out)
        self.assertFalse(out["exec_ran"], "a file written to /work was executable")

    def test_network_is_unreachable(self):
        out = self.ok(NETWORK)
        self.assertFalse(out["tcp_ip"]); self.assertFalse(out["tcp_ip2"]); self.assertFalse(out["dns"])
        self.assertEqual(out["interfaces"], ["lo"])
        self.assertFalse(out["docker_sock"])

    def test_filesystem_is_read_only_and_unprivileged(self):
        out = self.ok(FILESYSTEM)
        checks = out["checks"]
        for name in ("read_shadow", "list_root_home", "write_etc", "write_slash", "write_tmp", "write_usr", "tamper_runner"):
            self.assertNotEqual(checks[name], "allowed", f"{name}: {checks}")
        self.assertEqual(checks["write_work"], "allowed")
        self.assertFalse(out["docker_sock_in_mounts"])

    def test_tmpfs_size_limit_stops_disk_filling(self):
        out = self.ok(DISK_FILL)                              # tmp_mb=8
        self.assertIn(out["errno"], (28, 27), out)             # ENOSPC (or EFBIG)
        self.assertLessEqual(out["written"], 9_000_000)

    def test_environment_is_clean_and_process_is_unprivileged(self):
        out = self.ok(ENVIRONMENT)
        self.assertEqual(out["leaked"], [], out)
        self.assertEqual(out["in_procs"], [], out)
        self.assertEqual((out["uid"], out["gid"]), (65534, 65534))
        self.assertEqual(out["no_new_privs"].strip(), "1")
        self.assertEqual(int(out["cap_eff"], 16), 0, out)

    def test_subprocess_cannot_escape_the_unprivileged_context(self):
        out = self.ok(SUBPROCESS)
        self.assertEqual(out["uid"], "65534")
        self.assertNotEqual(out["shadow_rc"], 0)
        self.assertEqual(out["shadow_out"], "")

    def test_no_state_survives_between_runs(self):
        self.assertIn("marker.txt", self.ok(CROSS_RUN_WRITE))
        after = self.ok(CROSS_RUN_READ)
        self.assertFalse(after["marker"]); self.assertFalse(after["tmp_marker"])
        self.assertNotIn("marker.txt", after["work"])

    # ---------------------------------------------------------------- resource exhaustion
    def test_infinite_loop_is_terminated(self):
        started = time.monotonic()
        r = self.go(INFINITE_LOOP, limits=SandboxLimits(wall_seconds=3, cpu_seconds=60, memory_mb=256, tmp_mb=8))
        self.assertEqual(r.status, "timeout", r)
        self.assertLess(time.monotonic() - started, 30)

    def test_cpu_limit_terminates_busy_loop_early(self):
        started = time.monotonic()
        r = self.go(INFINITE_LOOP, limits=SandboxLimits(wall_seconds=30, cpu_seconds=2, memory_mb=256, tmp_mb=8))
        self.assertIn((r.status, r.detail), {("timeout", "cpu_limit"), ("killed", "sigkill")}, r)   # soft or hard CPU limit
        self.assertLess(time.monotonic() - started, 20)

    def test_sleep_is_terminated(self):
        r = self.go(SLEEP, limits=SandboxLimits(wall_seconds=3, cpu_seconds=60, memory_mb=256, tmp_mb=8))
        self.assertEqual(r.status, "timeout", r)

    def test_memory_exhaustion_is_contained(self):
        started = time.monotonic()
        r = self.go(MEMORY_BOMB, limits=SandboxLimits(wall_seconds=15, cpu_seconds=30, memory_mb=128, tmp_mb=8))
        self.assertIn(r.status, {"oom", "killed"}, r)         # not "error": a startup crash must not satisfy this
        self.assertLess(time.monotonic() - started, 40)

    def test_fork_is_capped_by_pids_limit(self):
        out = self.ok(FORK_BOUNDED, limits=SandboxLimits(wall_seconds=10, cpu_seconds=10, memory_mb=256, pids=32, tmp_mb=8))
        self.assertIsNotNone(out["stopped_by"], out)           # the limit actually stopped the forking
        self.assertLess(out["children"], 32, out)

    def test_unbounded_fork_bomb_is_contained(self):
        started = time.monotonic()
        r = self.go(FORK_BOMB, limits=SandboxLimits(wall_seconds=3, cpu_seconds=60, memory_mb=256, pids=32, tmp_mb=8))
        self.assertNotEqual(r.status, "ok", r)
        self.assertNotEqual(r.detail, "no_envelope", r)       # the bomb must have actually run, not a startup crash
        self.assertLess(time.monotonic() - started, 40)

    # ---------------------------------------------------------------- output channel
    def test_large_stdout_is_truncated_and_does_not_fail(self):
        r = self.go(STDOUT_BOUNDED, limits=SandboxLimits(wall_seconds=10, max_stdout_bytes=2000, tmp_mb=8))
        self.assertEqual(r.status, "ok", r)
        self.assertLessEqual(len(r.stdout), 2000)

    def test_infinite_stdout_flood_is_bounded(self):
        r = self.go(STDOUT_FLOOD, limits=SandboxLimits(wall_seconds=3, cpu_seconds=60, max_stdout_bytes=2000, tmp_mb=8))
        self.assertEqual(r.status, "timeout", r)             # it ran until the wall clock killed it
        self.assertLessEqual(len(r.stdout), 2000)

    def test_huge_result_is_rejected(self):
        r = self.go(HUGE_RESULT, limits=SandboxLimits(wall_seconds=10, max_result_bytes=10_000, tmp_mb=8))
        self.assertEqual(r.status, "output_too_large", r)
        self.assertIsNone(r.result_json)

    def test_malformed_returns_and_exceptions_are_errors(self):
        r = self.go(BAD_SHAPE); self.assertEqual((r.status, r.result_json), ("error", None))
        r = self.go(RAISES); self.assertEqual((r.status, r.detail), ("error", "RuntimeError"))
        self.assertIn("boom", r.stderr)

    # ---------------------------------------------------------------- recovery (keep last)
    def test_zz_sandbox_is_healthy_after_all_attacks(self):
        self.assertEqual(self.ok(SANITY, inputs={"records.jsonl": b'{"value": 40}\n{"value": 2}'}), 42)


class SnippetHygiene(unittest.TestCase):
    """Runs even without Docker: the attack snippets themselves must be valid programs."""

    def test_every_snippet_compiles_and_defines_solve(self):
        for name, code in ALL_SNIPPETS.items():
            tree = compile(code, name, "exec", flags=__import__("ast").PyCF_ONLY_AST)
            self.assertTrue(any(getattr(n, "name", None) == "solve" for n in tree.body), name)

    def test_harmless_snippets_run_through_the_contract_backend(self):
        from app.sandbox.tests._harness import run_contract as run_local, unavailable_reason
        if unavailable_reason():
            self.skipTest(unavailable_reason())
        r = run_local(SANITY, {"records.jsonl": b'{"value": 5}\n{"value": 6}'}); self.assertEqual(r.result_json["result"], 11)
        r = run_local(BAD_SHAPE); self.assertEqual(r.status, "error")
        r = run_local(RAISES); self.assertEqual(r.detail, "RuntimeError")
        r = run_local(HUGE_RESULT, limits=SandboxLimits(max_result_bytes=10_000, wall_seconds=5))
        self.assertEqual(r.status, "output_too_large")
        r = run_local(STDOUT_BOUNDED, limits=SandboxLimits(max_stdout_bytes=2000, wall_seconds=10))
        self.assertEqual(r.status, "ok"); self.assertLessEqual(len(r.stdout), 2000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
