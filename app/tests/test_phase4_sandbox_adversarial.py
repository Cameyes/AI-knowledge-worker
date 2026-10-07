"""Sandbox-layer adversarial corpus.

These tests intentionally bypass static_precheck and send adversarial code
straight to Sandbox v3. Run them on the production-target Linux/Docker setup.
"""
import os
import textwrap
import unittest

RUN = os.environ.get("RUN_PHASE4_SANDBOX") == "1"

if RUN:
    from app.sandbox.docker_backend import DockerBackend
    from app.sandbox.interface import SandboxLimits, SandboxRequest


if RUN:
    class SandboxAdversarial(unittest.TestCase):
        LIMITS = SandboxLimits(
            wall_seconds=3,
            cpu_seconds=2,
            memory_mb=256,
            pids=32,
            tmp_mb=8,
        )
        backend = DockerBackend()

        def run_direct(self, code):
            request = SandboxRequest(
                code=code,
                inputs={
                    "records.jsonl": b'{"id":"r1","value":1}\n',
                    "extra.json": b'{}',
                },
                limits=self.LIMITS,
            )
            return self.backend.run(request)

        def test_fork_bomb_is_contained(self):
            r = self.run_direct(
                'import os\n'
                'def solve(records, extra):\n'
                '    while True: os.fork()\n'
            )
            self.assertIn(r.status, {"timeout", "killed", "oom", "error"})
            self.assertNotEqual(r.detail, "sandbox_setup_failed")

        def test_memory_bomb_is_contained(self):
            code = (
                'def solve(records, extra):\n'
                '    x=[]\n'
                '    while True: x.append(bytearray(10_000_000))\n'
            )
            r = self.run_direct(code)
            self.assertIn(r.status, {"oom", "timeout", "killed", "error"})

        def test_huge_exponent_is_contained(self):
            r = self.run_direct(
                'def solve(records, extra):\n'
                '    return {"result": 10 ** 10 ** 10}\n'
            )
            self.assertIn(
                r.status,
                {"timeout", "error", "killed", "output_too_large"},
            )

        def test_infinite_loop_is_contained(self):
            r = self.run_direct(
                'def solve(records, extra):\n'
                '    while True: pass\n'
            )
            self.assertEqual(r.status, "timeout")

        def test_redos_is_contained(self):
            code = textwrap.dedent(
                """
                import re

                def solve(records, extra):
                    re.match(r"(a+)+$", "a" * 100000 + "!")
                    return {
                        "result": 1,
                        "inputs_used": ["r1"],
                        "excluded": [],
                        "assumptions": [],
                        "fields_used": {"r1": []},
                    }
                """
            )
            r = self.run_direct(code)
            self.assertIn(r.status, {"timeout", "error", "killed"})

        def test_filesystem_write_is_contained(self):
            code = textwrap.dedent(
                """
                def solve(records, extra):
                    try:
                        open("/work/escape.txt", "w").write("x")
                    except Exception:
                        pass

                    return {
                        "result": 1,
                        "inputs_used": ["r1"],
                        "excluded": [],
                        "assumptions": [],
                        "fields_used": {"r1": []},
                    }
                """
            )
            r = self.run_direct(code)
            self.assertEqual(r.status, "ok", r)

        def test_network_connect_is_contained(self):
            code = textwrap.dedent(
                """
                import socket

                def solve(records, extra):
                    try:
                        socket.create_connection(("1.1.1.1", 53), timeout=1)
                        connected = True
                    except Exception:
                        connected = False

                    return {
                        "result": connected,
                        "inputs_used": ["r1"],
                        "excluded": [],
                        "assumptions": [],
                        "fields_used": {"r1": []},
                    }
                """
            )
            r = self.run_direct(code)
            self.assertEqual(r.status, "ok", r)
            self.assertFalse(r.result_json["result"])

        def test_proc_and_environment_are_not_trusted(self):
            code = textwrap.dedent(
                """
                import os

                def solve(records, extra):
                    proc_exists = os.path.exists("/proc/self/status")
                    env = dict(os.environ)

                    return {
                        "result": {
                            "proc": proc_exists,
                            "env": sorted(env.keys()),
                        },
                        "inputs_used": ["r1"],
                        "excluded": [],
                        "assumptions": [],
                        "fields_used": {"r1": []},
                    }
                """
            )

            r = self.run_direct(code)

            self.assertEqual(r.status, "ok", r)

            result = r.result_json["result"]
            env_keys = set(result["env"])

            # /proc exists inside the Linux container; its presence is expected.
            self.assertTrue(result["proc"])

            # The container may have fixed image/runtime environment variables.
            # What must NOT happen is propagation of host-specific environment state.
            forbidden_host_vars = {
                "APPDATA",
                "LOCALAPPDATA",
                "USERPROFILE",
                "USERNAME",
                "COMSPEC",
                "PATHEXT",
                "PROGRAMDATA",
                "PROGRAMFILES",
                "PROGRAMFILES(X86)",
                "PROMPT",
                "PSMODULEPATH",
                "HOMEDRIVE",
                "HOMEPATH",
            }

            self.assertTrue(
                env_keys.isdisjoint(forbidden_host_vars),
                f"host-specific environment leaked: {env_keys & forbidden_host_vars}",
            )

            # These are intentionally supplied by the Sandbox-v3 Docker configuration.
            self.assertIn("HOME", env_keys)
            self.assertIn("TMPDIR", env_keys)
            self.assertIn("PYTHONDONTWRITEBYTECODE", env_keys)

        def test_huge_stdout_is_bounded(self):
            code = (
                'def solve(records, extra):\n'
                '    print("x" * 8_000_000)\n'
                '    return {'
                '"result": 1, '
                '"inputs_used": ["r1"], '
                '"excluded": [], '
                '"assumptions": [], '
                '"fields_used": {"r1": []}'
                '}\n'
            )
            r = self.run_direct(code)
            self.assertEqual(r.status, "ok", r)
            self.assertLessEqual(
                len(r.stdout),
                self.LIMITS.max_stdout_bytes,
            )

else:
    class SandboxAdversarial(unittest.TestCase):
        pass