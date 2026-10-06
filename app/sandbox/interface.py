"""Sandbox contract. The agent depends only on this module, never on a backend."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Status = Literal["ok", "error", "timeout", "oom", "output_too_large", "killed"]

# Logical input names become file names inside the sandbox.
INPUT_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,99}$")
RESERVED_INPUT_NAMES = frozenset({"main.py"})
RECORDS_INPUT = "records.jsonl"   # parsed and passed to solve() as `records`
EXTRA_INPUT = "extra.json"        # parsed and passed to solve() as `extra`

MAX_CODE_CHARS = 200_000
MAX_INPUT_BYTES_TOTAL = 50_000_000


class SandboxRequestError(ValueError):
    """The request itself is invalid. Nothing was executed."""


class SandboxInfrastructureError(RuntimeError):
    """The sandbox could not run (daemon down, image missing...). This is NOT a code failure
    and must never be shown to the agent as if its code had failed."""


@dataclass(frozen=True)
class SandboxLimits:
    wall_seconds: int = 20            # time allowed for the code (startup grace is added by the backend)
    cpu_seconds: int = 15
    memory_mb: int = 512
    pids: int = 64
    tmp_mb: int = 64
    max_result_bytes: int = 1_000_000
    max_stdout_bytes: int = 64_000    # per stream (stdout and stderr each)

    def validate(self) -> None:
        for name in ("wall_seconds", "cpu_seconds", "memory_mb", "pids", "tmp_mb",
                     "max_result_bytes", "max_stdout_bytes"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise SandboxRequestError(f"limit {name} must be a positive integer")


@dataclass(frozen=True)
class SandboxRequest:
    code: str                                         # full source of main.py; must define solve(records, extra)
    inputs: dict[str, bytes] = field(default_factory=dict)
    limits: SandboxLimits = SandboxLimits()

    def validate(self) -> None:
        self.limits.validate()
        if not isinstance(self.code, str) or not self.code.strip():
            raise SandboxRequestError("code must be a non-empty string")
        if len(self.code) > MAX_CODE_CHARS:
            raise SandboxRequestError("code too large")
        total = 0
        for name, content in self.inputs.items():
            if not INPUT_NAME_RE.match(name) or name in RESERVED_INPUT_NAMES or ".." in name:
                raise SandboxRequestError(f"invalid input name: {name!r}")
            if not isinstance(content, (bytes, bytearray)):
                raise SandboxRequestError(f"input {name!r} must be bytes")
            total += len(content)
        if total > MAX_INPUT_BYTES_TOTAL:
            raise SandboxRequestError("inputs too large")


@dataclass(frozen=True)
class SandboxResult:
    status: Status
    result_json: dict[str, Any] | None     # the dict returned by solve(); only when status == "ok"
    stdout: str                            # truncated to limits.max_stdout_bytes
    stderr: str                            # truncated; for status "error" includes the traceback
    duration_s: float                      # code execution time if reported by the runner, else host wall time
    peak_mem_mb: float | None              # not measured in v1
    code_sha256: str
    inputs_sha256: dict[str, str]
    detail: str = ""                       # short machine-readable reason, e.g. "SyntaxError", "cpu_limit"


class SandboxBackend(Protocol):
    def run(self, request: SandboxRequest) -> SandboxResult: ...
