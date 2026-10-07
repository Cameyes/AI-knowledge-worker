"""Phase 4 host orchestrator: pre-check -> sandbox -> contract -> metamorphic checks.

This module does not modify Sandbox v3. It uses the existing SandboxBackend as the
runtime containment layer and keeps all trust decisions on the host.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

from app.code_contract import (
    ContractValidationError,
    ValidatedContract,
    canonical_json,
    format_excluded_hint,
    select_records,
    validate_input_records,
    validate_output,
)
from app.static_precheck import PrecheckResult, precheck_code


Stage = Literal["precheck", "sandbox", "contract", "metamorphic"]


@dataclass(frozen=True)
class Phase4Failure:
    stage: Stage
    code: str
    message: str
    hint: str


@dataclass(frozen=True)
class Phase4Result:
    ok: bool
    contract: ValidatedContract | None
    failure: Phase4Failure | None
    sandbox_runs: int


class SandboxRequestFactory(Protocol):
    def __call__(self, code: str, records: Sequence[Mapping[str, Any]], extra: Mapping[str, Any], limits: Any) -> Any: ...


class SandboxBackend(Protocol):
    def run(self, request: Any) -> Any: ...


def default_request_factory(
    code: str,
    records: Sequence[Mapping[str, Any]],
    extra: Mapping[str, Any],
    limits: Any,
) -> Any:
    """Adapt this phase to the frozen SandboxRequest without changing Sandbox v3."""
    from app.sandbox.interface import SandboxRequest

    record_lines: list[str] = []
    for record in records:
        record_lines.append(
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    records_bytes = ("\n".join(record_lines) + ("\n" if record_lines else "")).encode("utf-8")
    extra_bytes = canonical_json(dict(extra)).encode("utf-8")

    return SandboxRequest(
        code=code,
        inputs={
            "records.jsonl": records_bytes,
            "extra.json": extra_bytes,
        },
        limits=limits,
    )


def _failure(stage: Stage, code: str, message: str, hint: str) -> Phase4Result:
    return Phase4Result(
        ok=False,
        contract=None,
        failure=Phase4Failure(stage=stage, code=code, message=message, hint=hint),
        sandbox_runs=0,
    )


class Phase4Executor:
    """Execute one generated computation under the frozen Phase 4 contract."""

    def __init__(
        self,
        backend: SandboxBackend,
        limits: Any,
        request_factory: SandboxRequestFactory = default_request_factory,
    ) -> None:
        self.backend = backend
        self.limits = limits
        self.request_factory = request_factory

    def _run_once(
        self,
        code: str,
        records: Sequence[Mapping[str, Any]],
        extra: Mapping[str, Any],
    ) -> Any:
        request = self.request_factory(code, records, extra, self.limits)
        return self.backend.run(request)

    @staticmethod
    def _sandbox_failure(result: Any, attempt: str) -> Phase4Result:
        status = getattr(result, "status", "unknown")
        detail = getattr(result, "detail", "") or "sandbox_failure"
        stderr = getattr(result, "stderr", "") or ""
        stderr = str(stderr)[-1500:]
        hint = f"Sandbox {attempt} failed with status={status}, detail={detail}. Regenerate code; do not retry infrastructure failures blindly."
        if stderr:
            hint += f" Sanitized stderr: {stderr}"
        return _failure("sandbox", str(detail), f"sandbox {attempt} did not succeed", hint)

    def execute(
        self,
        code: str,
        records: Sequence[Mapping[str, Any]],
        extra: Mapping[str, Any] | None = None,
    ) -> Phase4Result:
        extra = dict(extra or {})

        # 1. Cheap bounded static gate. Never compile or execute on the host.
        precheck: PrecheckResult = precheck_code(code)
        if not precheck.ok:
            location = ""
            if precheck.line is not None:
                location = f" at line {precheck.line}, column {precheck.column or 0}"
            return _failure(
                "precheck",
                precheck.code,
                f"generated code rejected{location}",
                precheck.message,
            )

        # 2. Validate input identity before any generated code runs.
        try:
            validate_input_records(records)
            # Strict JSON serialization here keeps replay inputs stable and prevents
            # non-JSON objects from reaching the sandbox input channel.
            for record in records:
                canonical_json(record)
            canonical_json(extra)
        except (ContractValidationError, TypeError, ValueError) as exc:
            return _failure(
                "contract",
                "invalid_input_records",
                "host-side input validation failed",
                str(exc),
            )

        sandbox_runs = 0

        # 3. Full execution #1.
        first = self._run_once(code, records, extra)
        sandbox_runs += 1
        if getattr(first, "status", None) != "ok":
            failed = self._sandbox_failure(first, "primary run")
            return Phase4Result(False, None, failed.failure, sandbox_runs)

        try:
            first_contract = validate_output(getattr(first, "result_json", None), records)
        except ContractValidationError as exc:
            return Phase4Result(
                False,
                None,
                Phase4Failure(
                    stage="contract",
                    code="invalid_return_contract",
                    message="primary sandbox result violated the host contract",
                    hint=str(exc),
                ),
                sandbox_runs,
            )

        # 4. Full execution #2: empirical determinism check.
        second = self._run_once(code, records, extra)
        sandbox_runs += 1
        if getattr(second, "status", None) != "ok":
            failed = self._sandbox_failure(second, "determinism run")
            return Phase4Result(False, None, failed.failure, sandbox_runs)

        try:
            second_contract = validate_output(getattr(second, "result_json", None), records)
        except ContractValidationError as exc:
            return Phase4Result(
                False,
                None,
                Phase4Failure(
                    stage="contract",
                    code="invalid_replay_contract",
                    message="determinism replay returned an invalid contract",
                    hint=str(exc),
                ),
                sandbox_runs,
            )

        if canonical_json(first_contract.result) != canonical_json(second_contract.result):
            return Phase4Result(
                False,
                None,
                Phase4Failure(
                    stage="metamorphic",
                    code="nondeterministic_result",
                    message="the same code and inputs produced different results",
                    hint="Run the computation without randomness, wall-clock state, environment state, unordered set iteration, or other external state.",
                ),
                sandbox_runs,
            )

        # 5. Full partition check: result must be a function of inputs_used only.
        used_ids = first_contract.inputs_used
        used_records = select_records(records, used_ids)

        third = self._run_once(code, used_records, extra)
        sandbox_runs += 1
        if getattr(third, "status", None) != "ok":
            failed = self._sandbox_failure(third, "inputs_used-only replay")
            return Phase4Result(False, None, failed.failure, sandbox_runs)

        try:
            third_contract = validate_output(getattr(third, "result_json", None), used_records)
        except ContractValidationError as exc:
            return Phase4Result(
                False,
                None,
                Phase4Failure(
                    stage="contract",
                    code="invalid_partition_replay_contract",
                    message="inputs_used-only replay returned an invalid contract",
                    hint=str(exc),
                ),
                sandbox_runs,
            )

        if canonical_json(first_contract.result) != canonical_json(third_contract.result):
            excluded_hint = format_excluded_hint(first_contract.excluded_ids())
            hint = (
                "Removing records declared as excluded changed the result. "
                "Move every record whose values influence the result into inputs_used, "
                "and declare its contributing fields in fields_used."
            )
            if excluded_hint:
                hint += " " + excluded_hint
            return Phase4Result(
                False,
                None,
                Phase4Failure(
                    stage="metamorphic",
                    code="exclusion_changed_result",
                    message="result is not a function of inputs_used only",
                    hint=hint,
                ),
                sandbox_runs,
            )

        return Phase4Result(True, first_contract, None, sandbox_runs)


__all__ = [
    "Phase4Executor",
    "Phase4Failure",
    "Phase4Result",
    "SandboxBackend",
    "default_request_factory",
]
