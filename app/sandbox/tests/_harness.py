"""Backend selection for the contract tests.

POSIX host  -> run runner.py in a plain local subprocess (fast, but NO isolation: contract checks only).
Windows host -> run the same tests through the real Docker container (the runner is Linux-only by design).
Override with SANDBOX_CONTRACT_BACKEND=docker|local.
"""
import base64, hashlib, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

from app.sandbox._proc import run_bounded
from app.sandbox.docker_backend import DEFAULT_IMAGE, DockerBackend, classify, host_stdout_cap
from app.sandbox.interface import SandboxLimits, SandboxRequest

RUNNER = Path(__file__).resolve().parents[1] / "runner.py"


def _has_resource() -> bool:
    try:
        import resource  # noqa: F401
        return os.name == "posix"
    except ImportError:
        return False


def docker_ready() -> tuple[bool, str]:
    if not shutil.which("docker"):
        return False, "docker CLI not found"
    try:
        proc = subprocess.run(["docker", "image", "inspect", DEFAULT_IMAGE], capture_output=True, timeout=20)
    except Exception as exc:
        return False, f"docker not usable: {exc}"
    return (proc.returncode == 0, "" if proc.returncode == 0 else f"image {DEFAULT_IMAGE} not built")


def backend_name() -> str:
    forced = os.environ.get("SANDBOX_CONTRACT_BACKEND", "").strip().lower()
    if forced in {"docker", "local"}:
        return forced
    return "local" if _has_resource() else "docker"


def unavailable_reason() -> str | None:
    if backend_name() == "local":
        return None if _has_resource() else "local backend needs a POSIX host with the 'resource' module"
    ready, why = docker_ready()
    return None if ready else f"contract tests need Docker on this host: {why}"


def run_contract(code, inputs=None, limits=None):
    limits = limits or SandboxLimits()
    request = SandboxRequest(code=code, inputs=inputs or {}, limits=limits)
    request.validate()
    if backend_name() == "docker":
        return DockerBackend(startup_grace_seconds=4).run(request)

    payload = json.dumps({
        "code": code,
        "inputs": {n: base64.b64encode(bytes(c)).decode() for n, c in request.inputs.items()},
        "limits": {"cpu_seconds": limits.cpu_seconds, "max_result_bytes": limits.max_result_bytes,
                   "max_stdout_bytes": limits.max_stdout_bytes},
    }).encode()
    with tempfile.TemporaryDirectory() as work:
        env = dict(os.environ, SANDBOX_WORKDIR=work)
        bounded = run_bounded([sys.executable, "-I", "-B", str(RUNNER)], payload, wall_seconds=limits.wall_seconds,
                              stdout_cap=host_stdout_cap(limits), stderr_cap=64_000, env=env)
    return classify(bounded, oom=False, limits=limits, code_sha=hashlib.sha256(code.encode()).hexdigest(),
                    inputs_sha={n: hashlib.sha256(bytes(c)).hexdigest() for n, c in request.inputs.items()})


def run_runner_raw(code, *, workdir, prelude="", limits=None, wall_seconds=20):
    """Run runner.py directly with a Python prelude (used to simulate setup failures). Returns BoundedResult."""
    limits = limits or SandboxLimits()
    payload = json.dumps({"code": code, "inputs": {}, "limits": {
        "cpu_seconds": limits.cpu_seconds, "max_result_bytes": limits.max_result_bytes,
        "max_stdout_bytes": limits.max_stdout_bytes}}).encode()
    boot = f"import runpy, sys\n{prelude}\nrunpy.run_path({str(RUNNER)!r}, run_name='__main__')"
    env = dict(os.environ, SANDBOX_WORKDIR=str(workdir))
    return run_bounded([sys.executable, "-c", boot], payload, wall_seconds=wall_seconds,
                       stdout_cap=host_stdout_cap(limits), stderr_cap=64_000, env=env)
