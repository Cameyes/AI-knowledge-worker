"""UNSAFE local harness. Runs runner.py in a plain subprocess to test the runner/host contract.

This provides NO isolation. It exists only so the contract (envelope, caps, classification,
timeouts) can be verified on machines without Docker. Never use it to run untrusted code.
"""
import base64, hashlib, json, os, sys, tempfile
from pathlib import Path

from app.sandbox._proc import run_bounded
from app.sandbox.docker_backend import classify, host_stdout_cap
from app.sandbox.interface import SandboxLimits, SandboxRequest, SandboxResult

RUNNER = Path(__file__).resolve().parents[1] / "runner.py"


def run_local(code, inputs=None, limits=None, grace=0):
    limits = limits or SandboxLimits()
    request = SandboxRequest(code=code, inputs=inputs or {}, limits=limits)
    request.validate()
    payload = json.dumps({
        "code": code,
        "inputs": {n: base64.b64encode(bytes(c)).decode() for n, c in request.inputs.items()},
        "limits": {"cpu_seconds": limits.cpu_seconds, "max_result_bytes": limits.max_result_bytes,
                   "max_stdout_bytes": limits.max_stdout_bytes},
    }).encode()
    with tempfile.TemporaryDirectory() as work:
        env = {"PATH": os.environ.get("PATH", ""), "SANDBOX_WORKDIR": work}
        bounded = run_bounded(
            [sys.executable, "-I", "-B", str(RUNNER)], payload,
            wall_seconds=limits.wall_seconds + grace,
            stdout_cap=host_stdout_cap(limits), stderr_cap=64_000, env=env,
        )
    return classify(
        bounded, oom=False, limits=limits,
        code_sha=hashlib.sha256(code.encode()).hexdigest(),
        inputs_sha={n: hashlib.sha256(bytes(c)).hexdigest() for n, c in request.inputs.items()},
    )
