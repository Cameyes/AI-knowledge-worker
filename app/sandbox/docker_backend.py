"""Docker backend: one fresh, network-less, read-only container per execution.

No bind mounts and no `docker cp`: the payload goes in over stdin and a single JSON envelope comes
back over stdout. That keeps the host filesystem out of the picture and behaves identically on
Linux and Windows (Docker Desktop).
"""
from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import uuid
from pathlib import Path

from ._proc import BoundedResult, run_bounded
from .interface import (
    SandboxInfrastructureError,
    SandboxLimits,
    SandboxRequest,
    SandboxResult,
)

DEFAULT_IMAGE = "rag-sandbox:py312-v1"
LABEL = "rag-sandbox=1"
SANDBOX_DIR = Path(__file__).resolve().parent
DOCKER_STDERR_CAP = 64_000
SANDBOX_UID = 65534
EXPECTED_RUNNER_VERSION = "3"   # must equal runner.RUNNER_VERSION baked into the image


def build_docker_command(*, docker_bin: str, image: str, name: str, limits: SandboxLimits) -> list[str]:
    """Pure function so the isolation flags can be unit-tested without Docker."""
    mem = f"{limits.memory_mb}m"
    return [
        docker_bin, "run", "-i",
        "--init",                      # runner is not PID 1: kernel-sent signals (e.g. SIGXCPU) are not ignored
        "--name", name,
        "--label", LABEL,
        "--pull", "never",
        "--network", "none",
        "--read-only",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--pids-limit", str(limits.pids),
        "--memory", mem,
        "--memory-swap", mem,          # equal to memory: no swap
        "--cpus", "1",
        "--user", f"{SANDBOX_UID}:{SANDBOX_UID}",
        # Docker does not guarantee who owns a --tmpfs mount or its mode, so declare both: owned by the
        # sandbox user, private to it (0700), and still noexec/nosuid/nodev with a size cap.
        "--tmpfs", f"/work:rw,noexec,nosuid,nodev,uid={SANDBOX_UID},gid={SANDBOX_UID},mode=0700,size={limits.tmp_mb}m",
        "--ulimit", "nofile=256:256",
        "--ulimit", "core=0",
        "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--env", "HOME=/work",
        "--env", "TMPDIR=/work",
        image,
    ]


def host_stdout_cap(limits: SandboxLimits) -> int:
    """Upper bound for a legitimate envelope: result + two streams escaped up to 6x + overhead."""
    return limits.max_result_bytes + 2 * 6 * limits.max_stdout_bytes + 64_000


def parse_envelope(raw: bytes) -> dict | None:
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if isinstance(data, dict) and data.get("v") == 1 and data.get("kind") in {
        "ok", "error", "output_too_large", "setup_error"
    }:
        return data
    return None


def classify(
    bounded: BoundedResult,
    *,
    oom: bool,
    limits: SandboxLimits,
    code_sha: str,
    inputs_sha: dict[str, str],
) -> SandboxResult:
    """Map process facts + envelope to a SandboxResult. Raises for infrastructure failures."""
    envelope = parse_envelope(bounded.stdout)
    docker_stderr = bounded.stderr.decode("utf-8", "replace")

    def build(status, detail, result=None, stdout="", stderr="", duration=None):
        return SandboxResult(
            status=status, result_json=result, stdout=stdout, stderr=stderr,
            duration_s=round(bounded.elapsed if duration is None else duration, 3),
            peak_mem_mb=None, code_sha256=code_sha, inputs_sha256=inputs_sha, detail=detail,
        )

    if bounded.timed_out:
        return build("timeout", "wall_clock", stderr=docker_stderr[-2000:])
    if bounded.stdout_overflow:
        return build("output_too_large", "envelope_overflow")

    if envelope is not None:
        if envelope.get("runner_version") != EXPECTED_RUNNER_VERSION:
            raise SandboxInfrastructureError(
                f"sandbox image runner version {envelope.get('runner_version')!r} != expected "
                f"{EXPECTED_RUNNER_VERSION!r}: the image is stale. Rebuild it: docker build -t {DEFAULT_IMAGE} {SANDBOX_DIR}"
            )
        if envelope["kind"] == "setup_error":
            raise SandboxInfrastructureError(f"sandbox setup failed before user code ran: {envelope.get('error')}")
        stdout = str(envelope.get("stdout", ""))
        stderr = str(envelope.get("stderr", ""))
        error = envelope.get("error")
        if error:
            stderr = (stderr + "\n" + str(error)).strip()
        duration = float(envelope.get("exec_seconds") or 0.0)
        detail = str(envelope.get("detail", ""))
        kind = envelope["kind"]
        if kind == "ok":
            result = envelope.get("result")
            if not isinstance(result, dict) or "result" not in result:
                return build("error", "bad_return_shape", stdout=stdout, stderr=stderr, duration=duration)
            if len(json.dumps(result).encode("utf-8")) > limits.max_result_bytes:
                return build("output_too_large", "result_too_large", stdout=stdout, stderr=stderr, duration=duration)
            return build("ok", "", result=result, stdout=stdout, stderr=stderr, duration=duration)
        return build(kind, detail, stdout=stdout, stderr=stderr, duration=duration)

    rc = bounded.returncode
    if oom:
        return build("oom", "oom_killed")
    # Docker reports a signal death as 128+N; a local subprocess reports -N.
    if rc in (152, -24):   # SIGXCPU: the RLIMIT_CPU the runner sets on itself
        return build("timeout", "cpu_limit")
    if rc in (137, -9):    # SIGKILL not attributed to OOM or our own timeout
        return build("killed", "sigkill", stderr=docker_stderr[-2000:])
    if rc in (125, 126, 127) and docker_stderr.strip():
        raise SandboxInfrastructureError(f"docker failed (exit {rc}): {docker_stderr.strip()[:500]}")
    return build("error", "no_envelope", stderr=docker_stderr[-2000:])


class DockerBackend:
    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        docker_bin: str = "docker",
        startup_grace_seconds: int = 10,
    ):
        self.image = image
        self.docker_bin = docker_bin
        self.startup_grace_seconds = startup_grace_seconds
        self._image_checked = False

    # ---- docker CLI helpers -------------------------------------------------
    def _docker(self, args: list[str], timeout: int = 20) -> subprocess.CompletedProcess:
        try:
            return subprocess.run([self.docker_bin, *args], capture_output=True, timeout=timeout)
        except FileNotFoundError as exc:
            raise SandboxInfrastructureError(f"docker CLI not found ({self.docker_bin!r})") from exc
        except subprocess.TimeoutExpired as exc:
            raise SandboxInfrastructureError(f"docker {args[0]} timed out") from exc

    def ensure_image(self) -> None:
        if self._image_checked:
            return
        proc = self._docker(["image", "inspect", self.image])
        if proc.returncode != 0:
            raise SandboxInfrastructureError(
                f"sandbox image {self.image!r} is missing or the Docker daemon is unreachable. "
                f"Build it with: docker build -t {self.image} {SANDBOX_DIR}"
            )
        self._image_checked = True

    def build_image(self) -> None:
        proc = self._docker(["build", "-t", self.image, str(SANDBOX_DIR)], timeout=900)
        if proc.returncode != 0:
            raise SandboxInfrastructureError(proc.stderr.decode("utf-8", "replace")[-1000:])
        self._image_checked = True

    def cleanup_stale(self) -> int:
        """Remove leftover sandbox containers (e.g. after a host crash). Returns the number removed."""
        listing = self._docker(["ps", "-aq", "--filter", f"label={LABEL}"])
        ids = listing.stdout.decode().split()
        for cid in ids:
            self._docker(["rm", "-f", cid])
        return len(ids)

    def _oom_killed(self, name: str) -> bool:
        try:
            proc = self._docker(["inspect", "-f", "{{.State.OOMKilled}}", name], timeout=10)
        except SandboxInfrastructureError:
            return False
        return proc.returncode == 0 and proc.stdout.decode().strip().lower() == "true"

    # ---- main entry ---------------------------------------------------------
    def run(self, request: SandboxRequest) -> SandboxResult:
        request.validate()
        self.ensure_image()
        limits = request.limits

        code_sha = hashlib.sha256(request.code.encode("utf-8")).hexdigest()
        inputs_sha = {n: hashlib.sha256(bytes(c)).hexdigest() for n, c in request.inputs.items()}
        payload = json.dumps({
            "code": request.code,
            "inputs": {n: base64.b64encode(bytes(c)).decode("ascii") for n, c in request.inputs.items()},
            "limits": {
                "cpu_seconds": limits.cpu_seconds,
                "max_result_bytes": limits.max_result_bytes,
                "max_stdout_bytes": limits.max_stdout_bytes,
            },
        }).encode("utf-8")

        name = f"rag-sbx-{uuid.uuid4().hex[:12]}"
        cmd = build_docker_command(docker_bin=self.docker_bin, image=self.image, name=name, limits=limits)

        def kill_container(_proc) -> None:
            try:
                self._docker(["kill", name], timeout=10)
            except SandboxInfrastructureError:
                pass

        try:
            try:
                bounded = run_bounded(
                    cmd, payload,
                    wall_seconds=limits.wall_seconds + self.startup_grace_seconds,
                    stdout_cap=host_stdout_cap(limits),
                    stderr_cap=DOCKER_STDERR_CAP,
                    kill=kill_container,
                )
            except FileNotFoundError as exc:
                raise SandboxInfrastructureError(f"docker CLI not found ({self.docker_bin!r})") from exc
            oom = self._oom_killed(name)
            return classify(bounded, oom=oom, limits=limits, code_sha=code_sha, inputs_sha=inputs_sha)
        finally:
            try:
                self._docker(["rm", "-f", name], timeout=15)
            except SandboxInfrastructureError:
                pass
