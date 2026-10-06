"""Bounded subprocess execution: capped output, drained pipes, hard wall-clock kill."""
from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class BoundedResult:
    returncode: int | None
    stdout: bytes
    stderr: bytes
    timed_out: bool
    stdout_overflow: bool
    stderr_overflow: bool
    elapsed: float


def run_bounded(
    cmd: list[str],
    stdin_bytes: bytes,
    *,
    wall_seconds: float,
    stdout_cap: int,
    stderr_cap: int,
    kill: Callable[[subprocess.Popen], None] | None = None,
    env: dict[str, str] | None = None,
) -> BoundedResult:
    """Run `cmd`, never buffering more than the caps and never outliving `wall_seconds`.

    Output beyond a cap is drained and discarded (so the child cannot block on a full pipe)
    and triggers a kill. `kill` lets a caller terminate more than the local process
    (e.g. `docker kill <name>`); the local process is always killed as well.
    """
    start = time.monotonic()
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
    )
    state = {"out": bytearray(), "err": bytearray(), "out_over": False, "err_over": False}
    kill_lock = threading.Lock()
    killed = {"done": False}

    def do_kill() -> None:
        with kill_lock:
            if killed["done"]:
                return
            killed["done"] = True
        try:
            if kill is not None:
                kill(proc)
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass

    def reader(stream, buf_key: str, over_key: str, cap: int) -> None:
        total = 0
        try:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                total += len(chunk)
                if total <= cap:
                    state[buf_key].extend(chunk)
                else:
                    keep = cap - (total - len(chunk))
                    if keep > 0:
                        state[buf_key].extend(chunk[:keep])
                    if not state[over_key]:
                        state[over_key] = True
                        do_kill()          # keep draining afterwards; kill makes the pipe close
        except Exception:
            pass

    def writer() -> None:
        try:
            proc.stdin.write(stdin_bytes)
            proc.stdin.close()
        except Exception:
            pass

    threads = [
        threading.Thread(target=reader, args=(proc.stdout, "out", "out_over", stdout_cap), daemon=True),
        threading.Thread(target=reader, args=(proc.stderr, "err", "err_over", stderr_cap), daemon=True),
        threading.Thread(target=writer, daemon=True),
    ]
    for t in threads:
        t.start()

    timed_out = False
    try:
        proc.wait(timeout=wall_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        do_kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    for t in threads:
        t.join(timeout=5)
    for stream in (proc.stdout, proc.stderr, proc.stdin):
        try:
            stream.close()
        except Exception:
            pass

    return BoundedResult(
        returncode=proc.returncode,
        stdout=bytes(state["out"]),
        stderr=bytes(state["err"]),
        timed_out=timed_out,
        stdout_overflow=state["out_over"],
        stderr_overflow=state["err_over"],
        elapsed=time.monotonic() - start,
    )
