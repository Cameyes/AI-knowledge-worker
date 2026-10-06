#!/usr/bin/env python3
"""In-sandbox runner (stdlib only). Baked into the image at /opt/runner/runner.py.

Trust model:   Host  ->  Supervisor  ->  Worker  ->  user code

  Supervisor (this process, pid != 1 under `docker run --init`)
    * reads the stdin payload, prepares /work, redirects fd 0/1/2, applies the rlimits;
    * forks the Worker and is the ONLY process that holds the envelope fd and writes the envelope;
    * never executes user code.
  Worker (forked child, its own process group)
    * first closes every inherited fd except the result pipe (so it cannot reach the envelope fd);
    * runs the user code and sends ONE length-prefixed JSON frame to the Supervisor.
  Result frame channel
    * NOT authenticated: a frame is the user code's own claim about its outcome, with exactly the
      authority of solve()'s return value. The Supervisor only accepts kind ok/error/output_too_large;
      it never accepts `setup_error`, which only the Supervisor can produce (before any user code runs).
  Forked descendants of user code cannot emit envelopes: they lack the envelope fd, and a descendant
  that falls out of solve() is detected by pid and exits silently.
  Worker killed by a signal -> the Supervisor kills the worker's process group and dies by the SAME
  signal, so the exit status keeps its meaning for the host (152 = SIGXCPU, 137 = SIGKILL, ...).

Protocol with the host
  stdin  : one JSON payload {"code": str, "inputs": {name: base64}, "limits": {...}}
  stdout : exactly ONE JSON envelope, or none if the worker died by a signal.
  The runner is an output formatter and a privilege separator, not the security boundary.
  Isolation comes from the container.
"""
import base64
import importlib.util
import json
import os
import re
import select
import signal
import stat as stat_mod
import struct
import sys
import time
import traceback
from pathlib import Path

try:
    import resource
except ImportError:  # non-POSIX host: the runner must refuse to run user code (see _apply_limits)
    resource = None

RUNNER_VERSION = "3"   # the host refuses an image whose runner version differs (stale image)
WORKDIR = Path(os.environ.get("SANDBOX_WORKDIR", "/work"))
NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,99}$")
RESERVED = {"main.py"}
MAX_TRACEBACK_CHARS = 4000
FSIZE_LIMIT = 16 * 1024 * 1024

FRAME_HEADER = struct.Struct(">Q")                       # 8-byte big-endian body length
FRAME_KINDS = frozenset({"ok", "error", "output_too_large"})   # NEVER "setup_error"
FRAME_OVERHEAD = 16_000                                  # room for traceback + framing around max_result_bytes
POLL_SECONDS = 0.05


class _CappedStream:
    """Text stream writing to a raw fd, silently dropping everything past `cap` bytes."""

    encoding = "utf-8"
    errors = "replace"

    def __init__(self, fd: int, cap: int):
        self._fd, self._cap, self.written, self.truncated = fd, cap, 0, False

    def writable(self):
        return True

    def isatty(self):
        return False

    def fileno(self):
        return self._fd

    def flush(self):
        pass

    def write(self, text):
        data = str(text).encode("utf-8", "replace")
        room = self._cap - self.written
        if len(data) > room:
            self.truncated = True
            data = data[: max(room, 0)]
        view = memoryview(data)
        while view:
            n = os.write(self._fd, view)
            view = view[n:]
        self.written += len(data)
        return len(text)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        view = view[n:]


def _apply_limits(limits: dict) -> None:
    """Fail closed: if a limit cannot be applied, user code must not run."""
    if resource is None:
        raise RuntimeError("runner requires the POSIX 'resource' module (Linux container); refusing to run unlimited")
    cpu = int(limits.get("cpu_seconds", 15))
    for res, value in (
        (resource.RLIMIT_CPU, (cpu, cpu + 2)),
        (resource.RLIMIT_FSIZE, (FSIZE_LIMIT, FSIZE_LIMIT)),
        (resource.RLIMIT_CORE, (0, 0)),
    ):
        resource.setrlimit(res, value)


def _diagnostics() -> str:
    try:
        st = os.stat(WORKDIR)
        mode = oct(st.st_mode & 0o7777)
        owner = f"{st.st_uid}:{st.st_gid}"
    except OSError as exc:
        mode, owner = f"unstatable({exc})", "?"
    uid = getattr(os, "getuid", lambda: -1)()
    return f"uid={uid} workdir={WORKDIR} mode={mode} owner={owner}"


def _read_capture(path: Path, cap: int) -> tuple[str, bool]:
    """Read a capture file the WORKER could have tampered with: never follow symlinks, never block on a
    FIFO, accept only a regular file."""
    fd = None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
        if not stat_mod.S_ISREG(os.fstat(fd).st_mode):
            return "", False
        raw = b""
        while len(raw) <= cap:
            chunk = os.read(fd, cap + 1 - len(raw))
            if not chunk:
                break
            raw += chunk
    except OSError:
        return "", False
    finally:
        if fd is not None:
            os.close(fd)
    return raw[:cap].decode("utf-8", "replace"), len(raw) > cap


def _fail(kind: str, detail: str, error: str) -> dict:
    return {"kind": kind, "detail": detail, "error": error[-MAX_TRACEBACK_CHARS:], "result": None}


def _execute(payload: dict, max_result_bytes: int) -> dict:
    inputs_dir = WORKDIR / "in"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    records, extra = [], {}

    for name, b64 in (payload.get("inputs") or {}).items():
        if not isinstance(name, str) or not NAME_RE.match(name) or name in RESERVED or ".." in name:
            return _fail("error", "bad_input_name", f"invalid input name {name!r}")
        content = base64.b64decode(b64)
        (inputs_dir / name).write_bytes(content)
        try:
            if name == "records.jsonl":
                records = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
            elif name == "extra.json":
                extra = json.loads(content.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            return _fail("error", "bad_input_content", f"could not parse {name}: {exc}")

    code_path = WORKDIR / "main.py"
    code_path.write_text(payload["code"], encoding="utf-8")

    try:
        spec = importlib.util.spec_from_file_location("agent_main", code_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        solve = getattr(module, "solve", None)
        if not callable(solve):
            return _fail("error", "missing_solve", "main.py must define solve(records, extra)")
        out = solve(records, extra)
    except SyntaxError:
        return _fail("error", "SyntaxError", traceback.format_exc())
    except BaseException as exc:  # includes SystemExit raised by user code
        return _fail("error", type(exc).__name__, traceback.format_exc())

    if not isinstance(out, dict) or "result" not in out:
        return _fail("error", "bad_return_shape", "solve must return a dict containing the key 'result'")
    try:
        encoded = json.dumps(out, allow_nan=False, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        return _fail("error", "not_json_serializable", f"return value is not strict JSON: {exc}")
    if len(encoded) > max_result_bytes:
        return _fail("output_too_large", "result_too_large",
                     f"result is {len(encoded)} bytes; limit is {max_result_bytes}")
    return {"kind": "ok", "detail": "", "error": None, "result": out}


def _encode_frame(obj: dict) -> bytes:
    body = json.dumps(obj, allow_nan=False, ensure_ascii=False).encode("utf-8")
    return FRAME_HEADER.pack(len(body)) + body


def _parse_frame(buf: bytes, max_body: int):
    """-> ("need_more", None) | ("ok", obj) | ("bad", reason)"""
    if len(buf) < FRAME_HEADER.size:
        return "need_more", None
    (length,) = FRAME_HEADER.unpack_from(buf, 0)
    if length > max_body:
        return "bad", "frame_too_large"
    if len(buf) < FRAME_HEADER.size + length:
        return "need_more", None
    try:
        obj = json.loads(bytes(buf[FRAME_HEADER.size: FRAME_HEADER.size + length]).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return "bad", "frame_not_json"
    return "ok", obj


def _validate_outcome(obj, max_result: int):
    """Normalize a worker frame. Returns None unless it is a well-formed ok/error/output_too_large outcome."""
    if not isinstance(obj, dict) or obj.get("kind") not in FRAME_KINDS:
        return None
    kind, detail, error, result = obj["kind"], obj.get("detail", ""), obj.get("error"), obj.get("result")
    if not isinstance(detail, str) or not (error is None or isinstance(error, str)):
        return None
    if kind == "ok":
        if not isinstance(result, dict) or "result" not in result:
            return None
        try:
            if len(json.dumps(result, allow_nan=False, ensure_ascii=False).encode("utf-8")) > max_result:
                return None
        except (TypeError, ValueError):
            return None
    else:
        result = None
    return {
        "kind": kind, "detail": detail[:100],
        "error": error[-MAX_TRACEBACK_CHARS:] if error else error, "result": result,
        "stdout_truncated": obj.get("stdout_truncated") is True,
        "stderr_truncated": obj.get("stderr_truncated") is True,
    }


def _close_inherited_fds(keep: set) -> None:
    try:
        fds = [int(name) for name in os.listdir("/proc/self/fd")]
    except (OSError, ValueError):
        fds = list(range(3, 4096))
    for fd in fds:
        if fd >= 3 and fd not in keep:
            try:
                os.close(fd)
            except OSError:
                pass


def _run_worker(payload: dict, max_result: int, result_fd: int, out_stream, err_stream) -> None:
    """Worker side of the fork. NEVER returns: every path ends in os._exit, so neither this process nor any
    descendant that falls out of user code can run supervisor code."""
    my_pid = os.getpid()
    try:
        try:
            os.setpgid(0, 0)           # own process group, so the supervisor can kill all descendants
        except OSError:
            pass
        _close_inherited_fds({result_fd})   # drops the envelope fd and the saved stderr BEFORE user code
        try:
            outcome = _execute(payload, max_result)
        except BaseException:
            outcome = _fail("error", "runner_error", traceback.format_exc())
        if os.getpid() != my_pid:      # a forked descendant that fell out of user code has no authority
            os._exit(0)
        outcome["stdout_truncated"] = out_stream.truncated
        outcome["stderr_truncated"] = err_stream.truncated
        _write_all(result_fd, _encode_frame(outcome))
    except BaseException:
        pass
    finally:
        os._exit(0)


def _drain(fd: int, buf: bytearray, limit: int) -> None:
    os.set_blocking(fd, False)
    try:
        while len(buf) < limit:
            chunk = os.read(fd, 65536)
            if not chunk:
                return
            buf.extend(chunk)
    except (BlockingIOError, OSError):
        return


def _supervise(worker_pid: int, result_r: int, max_result: int):
    """Wait for ONE valid frame or for the worker to end. -> (outcome | None, wait_status | None, problem | None)"""
    max_body = max_result + FRAME_OVERHEAD
    buf_limit = FRAME_HEADER.size + max_body + 65536
    buf, status, eof = bytearray(), None, False

    def frame_state():
        state, obj = _parse_frame(buf, max_body)
        if state == "ok":
            outcome = _validate_outcome(obj, max_result)
            return ("done", outcome) if outcome else ("bad", "bad_worker_frame")
        return (state, obj)

    while True:
        if not eof:
            try:
                ready = select.select([result_r], [], [], POLL_SECONDS)[0]
            except InterruptedError:
                continue
            if ready:
                chunk = os.read(result_r, 65536)
                if chunk:
                    buf.extend(chunk)
                    state, value = frame_state()
                    if state == "done":
                        return value, status, None
                    if state == "bad" or len(buf) > buf_limit:
                        return None, status, value if state == "bad" else "frame_too_large"
                else:
                    eof = True
        else:
            time.sleep(POLL_SECONDS)
        if status is None:
            pid, wait_status = os.waitpid(worker_pid, os.WNOHANG)
            if pid:
                status = wait_status
        if status is not None:
            _drain(result_r, buf, buf_limit)       # the frame, if any, is already in the pipe buffer
            state, value = frame_state()
            if state == "done":
                return value, status, None
            return None, status, (value if state == "bad" else None)


def _kill_group_and_reap(worker_pid: int, status) -> None:
    try:
        os.killpg(worker_pid, signal.SIGKILL)
    except OSError:
        try:
            os.kill(worker_pid, signal.SIGKILL)
        except OSError:
            pass
    if status is None:
        try:
            os.waitpid(worker_pid, 0)
        except OSError:
            pass


def _die_like(sig: int, diag_fd: int) -> None:
    """Terminate this process by the same signal that killed the worker (never returns)."""
    try:
        name = signal.Signals(sig).name
    except ValueError:
        name = str(sig)
    try:
        _write_all(diag_fd, f"sandbox worker terminated by signal {sig} ({name})\n".encode())
    except OSError:
        pass
    try:
        signal.signal(sig, signal.SIG_DFL)
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {sig})
    except (OSError, ValueError, RuntimeError):
        pass
    os.kill(os.getpid(), sig)
    time.sleep(1)
    os._exit(128 + sig)


def _setup_error(envelope_fd: int, exc: Exception) -> None:
    _write_all(envelope_fd, json.dumps({
        "v": 1, "runner_version": RUNNER_VERSION, "kind": "setup_error", "detail": "sandbox_setup_failed",
        "error": f"{type(exc).__name__}: {exc} | {_diagnostics()}", "result": None,
        "stdout": "", "stderr": "", "exec_seconds": 0.0}).encode("utf-8"))
    os._exit(0)


def main() -> None:
    envelope_fd = os.dup(1)      # held by the supervisor ONLY: the worker closes it before any user code
    diag_fd = os.dup(2)          # the real stderr (docker's), likewise supervisor-only
    raw = sys.stdin.buffer.read()

    try:
        payload = json.loads(raw.decode("utf-8"))
        limits = payload.get("limits") or {}
        stream_cap = int(limits.get("max_stdout_bytes", 64000))
        max_result = int(limits.get("max_result_bytes", 1_000_000))
        code = payload["code"]
        assert isinstance(code, str)
    except Exception as exc:
        _write_all(envelope_fd, json.dumps({"v": 1, "runner_version": RUNNER_VERSION,
                                            **_fail("error", "bad_payload", str(exc)),
                                            "stdout": "", "stderr": "", "exec_seconds": 0.0}).encode())
        os._exit(0)

    out_path, err_path = WORKDIR / ".stdout", WORKDIR / ".stderr"
    try:
        WORKDIR.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        os.dup2(os.open(out_path, flags, 0o600), 1)
        os.dup2(os.open(err_path, flags, 0o600), 2)
        os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
        out_stream, err_stream = _CappedStream(1, stream_cap), _CappedStream(2, stream_cap)
        sys.stdout, sys.stderr = out_stream, err_stream
        _apply_limits(limits)             # applied here so the worker inherits them; failure => setup_error
        result_r, result_w = os.pipe()
        start = time.monotonic()
        worker_pid = os.fork()
    except Exception as exc:
        # Infrastructure failure BEFORE any user code ran. Only the supervisor can emit this kind.
        _setup_error(envelope_fd, exc)

    if worker_pid == 0:
        os.close(result_r)
        _run_worker(payload, max_result, result_w, out_stream, err_stream)   # never returns
    os.close(result_w)

    outcome, status, problem = _supervise(worker_pid, result_r, max_result)
    exec_seconds = time.monotonic() - start
    _kill_group_and_reap(worker_pid, status)

    if outcome is None:
        if problem:
            outcome = _fail("error", "bad_worker_frame", f"the worker sent an invalid result frame ({problem})")
        elif status is not None and os.WIFSIGNALED(status):
            _die_like(os.WTERMSIG(status), diag_fd)        # no envelope: the exit status carries the signal
        else:
            code_ = os.WEXITSTATUS(status) if status is not None else -1
            outcome = _fail("error", "no_result" if code_ == 0 else "worker_exit_nonzero",
                            f"the worker exited with status {code_} without reporting a result")

    stdout_text, out_big = _read_capture(out_path, stream_cap)
    stderr_text, err_big = _read_capture(err_path, stream_cap)
    envelope = {
        "v": 1,
        "runner_version": RUNNER_VERSION,
        "kind": outcome["kind"], "detail": outcome["detail"], "error": outcome["error"], "result": outcome["result"],
        "stdout": stdout_text,
        "stderr": stderr_text,
        "stdout_truncated": bool(outcome.get("stdout_truncated")) or out_big,
        "stderr_truncated": bool(outcome.get("stderr_truncated")) or err_big,
        "exec_seconds": exec_seconds,
    }
    _write_all(envelope_fd, json.dumps(envelope).encode("utf-8"))
    os.close(envelope_fd)
    os._exit(0)  # do not wait for threads the user code may have started


if __name__ == "__main__":
    main()
