"""Runner + host contract tests. Local subprocess on POSIX; the real container on Windows (see _harness.py)."""
import json
import unittest

from app.sandbox.interface import SandboxLimits
from app.sandbox.tests._harness import run_contract as run_local, unavailable_reason

SUM = '''
def solve(records, extra):
    return {"result": sum(r["value"] for r in records), "inputs_used": [r["id"] for r in records]}
'''
RECORDS = b"\n".join(json.dumps({"id": f"r{i}", "value": v}).encode() for i, v in enumerate([1, 2, 3]))
FAST = SandboxLimits(wall_seconds=4, cpu_seconds=3)


def setUpModule():
    reason = unavailable_reason()
    if reason:
        raise unittest.SkipTest(reason)


class Contract(unittest.TestCase):
    def test_sum_over_records(self):
        r = run_local(SUM, {"records.jsonl": RECORDS})
        self.assertEqual(r.status, "ok", r)
        self.assertEqual(r.result_json["result"], 6)
        self.assertEqual(r.result_json["inputs_used"], ["r0", "r1", "r2"])
        self.assertEqual(len(r.code_sha256), 64)
        self.assertIn("records.jsonl", r.inputs_sha256)

    def test_extra_json_is_parsed(self):
        r = run_local('def solve(records, extra):\n    return {"result": extra["k"]}\n',
                      {"extra.json": b'{"k": 7}'})
        self.assertEqual((r.status, r.result_json["result"]), ("ok", 7))

    def test_stdout_is_captured_and_cannot_forge_the_envelope(self):
        code = '''
import os
def solve(records, extra):
    print("hello")
    os.write(1, b'{"v":1,"kind":"ok","result":{"result":999},"stdout":"","stderr":""}')
    os.write(2, b"to-stderr")
    return {"result": 1}
'''
        r = run_local(code)
        self.assertEqual(r.status, "ok", r)
        self.assertEqual(r.result_json["result"], 1)          # real return value, not the forged 999
        self.assertIn("hello", r.stdout)
        self.assertIn('"result":999', r.stdout)                # forged text is just captured stdout
        self.assertIn("to-stderr", r.stderr)

    def test_exceptions_return_error_with_traceback(self):
        r = run_local('def solve(records, extra):\n    return {"result": 1 / 0}\n')
        self.assertEqual((r.status, r.detail), ("error", "ZeroDivisionError"))
        self.assertIn("ZeroDivisionError", r.stderr)

    def test_syntax_error_missing_solve_and_exit(self):
        self.assertEqual(run_local("def solve(:\n").detail, "SyntaxError")
        self.assertEqual(run_local("x = 1\n").detail, "missing_solve")
        r = run_local("import sys\ndef solve(r, e):\n    sys.exit(0)\n")
        self.assertEqual((r.status, r.detail), ("error", "SystemExit"))

    def test_silent_hard_exit_is_an_attributable_error_not_ok(self):
        # Before the supervisor/worker split the host saw "no_envelope"; now the supervisor itself reports it.
        r = run_local("import os\ndef solve(r, e):\n    os._exit(0)\n")
        self.assertEqual((r.status, r.detail), ("error", "no_result"))

    def test_bad_return_shapes(self):
        cases = {
            "not a dict": 'def solve(r, e):\n    return [1]\n',
            "no result key": 'def solve(r, e):\n    return {"answer": 1}\n',
            "nan": 'def solve(r, e):\n    return {"result": float("nan")}\n',
            "not serializable": 'def solve(r, e):\n    return {"result": {1, 2}}\n',
        }
        for label, code in cases.items():
            r = run_local(code)
            self.assertEqual(r.status, "error", label)
            self.assertIsNone(r.result_json, label)

    def test_result_too_large(self):
        r = run_local('def solve(r, e):\n    return {"result": "x" * 50000}\n',
                      limits=SandboxLimits(max_result_bytes=10_000, wall_seconds=4))
        self.assertEqual((r.status, r.detail), ("output_too_large", "result_too_large"))

    def test_bounded_stdout_is_truncated_not_failed(self):
        code = 'def solve(r, e):\n    print("x" * 5_000_000)\n    return {"result": 1}\n'
        r = run_local(code, limits=SandboxLimits(max_stdout_bytes=1000, wall_seconds=6))
        self.assertEqual(r.status, "ok", r)
        self.assertLessEqual(len(r.stdout), 1000)

    def test_infinite_stdout_flood_ends_at_wall_clock_with_bounded_memory(self):
        code = 'def solve(r, e):\n    while True:\n        print("x" * 1000)\n'
        r = run_local(code, limits=SandboxLimits(wall_seconds=2, cpu_seconds=30, max_stdout_bytes=1000))
        self.assertEqual(r.status, "timeout", r)

    def test_infinite_loop_is_killed_by_wall_clock(self):
        r = run_local("def solve(r, e):\n    while True:\n        pass\n",
                      limits=SandboxLimits(wall_seconds=2, cpu_seconds=30))
        self.assertEqual((r.status, r.detail), ("timeout", "wall_clock"))
        self.assertLess(r.duration_s, 20)

    def test_sleep_is_killed_by_wall_clock(self):
        r = run_local("import time\ndef solve(r, e):\n    time.sleep(60)\n",
                      limits=SandboxLimits(wall_seconds=2, cpu_seconds=30))
        self.assertEqual(r.status, "timeout")

    def test_cpu_limit_terminates_a_busy_loop_before_the_wall_clock(self):
        r = run_local("def solve(r, e):\n    while True:\n        pass\n",
                      limits=SandboxLimits(wall_seconds=30, cpu_seconds=1))
        self.assertIn((r.status, r.detail), {("timeout", "cpu_limit"), ("killed", "sigkill")}, r)   # soft or hard CPU limit
        self.assertLess(r.duration_s, 20)

    def test_user_exception_named_like_an_internal_error_is_still_just_an_error(self):
        code = ("class sandbox_setup_failed(Exception):\n    pass\n"
                "def solve(r, e):\n    raise sandbox_setup_failed('x')\n")
        r = run_local(code)                                   # must NOT raise SandboxInfrastructureError
        self.assertEqual((r.status, r.detail), ("error", "sandbox_setup_failed"))

    def test_large_stdin_payload_does_not_deadlock(self):
        big = b"a" * 5_000_000
        r = run_local('def solve(r, e):\n    return {"result": 1}\n', {"blob.bin": big},
                      limits=SandboxLimits(wall_seconds=10))
        self.assertEqual(r.status, "ok", r)



class FailClosed(unittest.TestCase):
    """Setup failures must be reported as infrastructure errors and must never run user code."""
    MARKER_CODE = "import pathlib\npathlib.Path({marker!r}).write_text('ran')\ndef solve(r, e):\n    return {{'result': 1}}\n"

    def run_and_check(self, *, prelude="", workdir=None, expect_in_error=()):
        import json as _json, tempfile as _tf, pathlib
        from app.sandbox._proc import BoundedResult
        from app.sandbox.docker_backend import classify, parse_envelope
        from app.sandbox.interface import SandboxInfrastructureError
        from app.sandbox.tests._harness import run_runner_raw
        with _tf.TemporaryDirectory() as tmp:
            marker = pathlib.Path(tmp) / "marker"
            wd = workdir(tmp) if workdir else pathlib.Path(tmp) / "work"
            b = run_runner_raw(self.MARKER_CODE.format(marker=str(marker)), workdir=wd, prelude=prelude)
            env = parse_envelope(b.stdout)
            self.assertIsNotNone(env, b.stderr)
            self.assertEqual((env["kind"], env["detail"]), ("setup_error", "sandbox_setup_failed"))
            for needle in expect_in_error:
                self.assertIn(needle, env["error"])
            self.assertIn("uid=", env["error"])                          # diagnostics for exactly this failure class
            self.assertFalse(marker.exists(), "user code ran despite a setup failure")
            with self.assertRaises(SandboxInfrastructureError):
                classify(b, oom=False, limits=SandboxLimits(), code_sha="c" * 64, inputs_sha={})

    def test_unusable_workdir_is_an_infrastructure_error(self):
        def under_a_regular_file(tmp):
            import pathlib
            f = pathlib.Path(tmp) / "file"; f.write_text("x")
            return f / "work"                                             # mkdir beneath a file cannot succeed
        self.run_and_check(workdir=under_a_regular_file)

    def test_missing_resource_module_refuses_to_run_user_code(self):
        self.run_and_check(prelude="sys.modules['resource'] = None", expect_in_error=("resource",))

    @unittest.skipUnless(__import__("os").name == "posix", "setrlimit exists on POSIX only")
    def test_failure_to_apply_a_limit_refuses_to_run_user_code(self):
        self.run_and_check(prelude="import resource\n"
                           "def _deny(*a, **k):\n    raise OSError('denied')\nresource.setrlimit = _deny",
                           expect_in_error=("denied",))



@unittest.skipUnless(hasattr(__import__("os"), "geteuid") and __import__("os").geteuid() == 0,
                     "needs root to drop privileges to the sandbox UID")
class WorkdirOwnership(unittest.TestCase):
    """The /work failure mode, reproduced at the permission level: UID 65534 vs. who owns the workdir."""

    def trial(self, mode, owner):
        import json as _json, os, pathlib, subprocess, sys, tempfile
        from app.sandbox.tests._harness import RUNNER
        payload = _json.dumps({"code": "def solve(r, e):\n    return {'result': 'hello'}\n", "inputs": {}, "limits": {}}).encode()
        root = pathlib.Path(tempfile.mkdtemp()); os.chmod(root, 0o755)
        work = root / "work"; work.mkdir(); os.chown(work, *owner); os.chmod(work, mode)
        p = subprocess.run([sys.executable, "-I", "-B", str(RUNNER)], input=payload, capture_output=True,
                           user=65534, group=65534, env={"SANDBOX_WORKDIR": str(work), "PATH": os.environ["PATH"]})
        return _json.loads(p.stdout.decode())

    def test_root_owned_workdir_is_reported_as_setup_error_with_diagnostics(self):
        env = self.trial(0o755, (0, 0))
        self.assertEqual(env["kind"], "setup_error")
        self.assertIn("PermissionError", env["error"]); self.assertIn("uid=65534", env["error"])

    def test_workdir_owned_by_the_sandbox_user_with_mode_0700_works(self):
        env = self.trial(0o700, (65534, 65534))
        self.assertEqual(env["kind"], "ok", env)


if __name__ == "__main__":
    unittest.main(verbosity=2)
