"""Host-side contract validation for generated computation results.

The generated program is untrusted. This module never trusts its audit metadata
without checking it against the actual input records. Semantic correctness is
left to the later verifier; this module enforces mechanically testable structure
and partition invariants.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


REASON_CODES = frozenset(
    {
        "MISSING_FIELD",
        "OUT_OF_SCOPE",
        "FILTERED_BY_CRITERIA",
        "UNPARSEABLE",
        "DUPLICATE",
    }
)

MAX_REASON_CHARS = 300
MAX_ASSUMPTION_CHARS = 500
MAX_FIELDS_USED_PER_RECORD = 100


class ContractValidationError(ValueError):
    """A generated computation violated the host-side result contract."""


@dataclass(frozen=True)
class ValidatedContract:
    result: Any
    inputs_used: tuple[str, ...]
    excluded: tuple[dict[str, Any], ...]
    fields_used: dict[str, tuple[str, ...]]
    assumptions: tuple[str, ...]

    def canonical_result(self) -> str:
        return canonical_json(self.result)

    def excluded_ids(self) -> tuple[str, ...]:
        return tuple(
            record_id
            for group in self.excluded
            for record_id in group["record_ids"]
        )


def canonical_json(value: Any) -> str:
    """Return strict, deterministic JSON suitable for exact comparison."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _validate_record_ids(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Mapping[str, Any]], tuple[str, ...]]:
    by_id: dict[str, Mapping[str, Any]] = {}
    ordered: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ContractValidationError(f"input record {index} is not an object")
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ContractValidationError(f"input record {index} has an invalid non-empty string id")
        if record_id in by_id:
            raise ContractValidationError(f"input record id '{record_id}' is duplicated")
        by_id[record_id] = record
        ordered.append(record_id)
    return by_id, tuple(ordered)


def validate_input_records(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Mapping[str, Any]], tuple[str, ...]]:
    """Validate record identity before any generated code is executed."""
    return _validate_record_ids(records)


def _validate_string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ContractValidationError(f"{field} must be a list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ContractValidationError(f"{field} must contain non-empty strings only")
    if len(set(value)) != len(value):
        raise ContractValidationError(f"{field} contains duplicate ids")
    return list(value)


def _validate_excluded(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ContractValidationError("excluded must be a list")

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ContractValidationError(f"excluded[{index}] must be an object")
        reason_code = item.get("reason_code")
        if reason_code not in REASON_CODES:
            raise ContractValidationError(
                f"excluded[{index}].reason_code must be one of {sorted(REASON_CODES)}"
            )
        reason = item.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ContractValidationError(f"excluded[{index}].reason must be a non-empty string")
        if len(reason) > MAX_REASON_CHARS:
            raise ContractValidationError(f"excluded[{index}].reason is too long")

        ids = item.get("record_ids")
        ids = _validate_string_list(ids, f"excluded[{index}].record_ids")
        if not ids:
            raise ContractValidationError(f"excluded[{index}].record_ids cannot be empty")
        overlap = seen.intersection(ids)
        if overlap:
            raise ContractValidationError(f"record ids appear in multiple excluded groups: {sorted(overlap)}")
        seen.update(ids)
        normalized.append(
            {
                "record_ids": tuple(sorted(ids)),
                "reason_code": reason_code,
                "reason": reason.strip(),
            }
        )

    normalized.sort(key=lambda group: (group["reason_code"], group["reason"], group["record_ids"]))
    return normalized


def _validate_fields_used(value: Any, record_map: Mapping[str, Mapping[str, Any]], inputs_used: set[str]) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        raise ContractValidationError("fields_used must be an object")

    normalized: dict[str, tuple[str, ...]] = {}
    for record_id, fields in value.items():
        if not isinstance(record_id, str) or not record_id.strip():
            raise ContractValidationError("fields_used keys must be non-empty strings")
        if record_id not in inputs_used:
            raise ContractValidationError(
                f"fields_used contains record '{record_id}' which is not in inputs_used"
            )
        if not isinstance(fields, list):
            raise ContractValidationError(f"fields_used['{record_id}'] must be a list")
        if len(fields) > MAX_FIELDS_USED_PER_RECORD:
            raise ContractValidationError(f"fields_used['{record_id}'] has too many fields")
        if any(not isinstance(field, str) or not field.strip() for field in fields):
            raise ContractValidationError(f"fields_used['{record_id}'] must contain non-empty strings only")
        if len(set(fields)) != len(fields):
            raise ContractValidationError(f"fields_used['{record_id}'] contains duplicate fields")

        record = record_map[record_id]
        for field in fields:
            if field not in record:
                raise ContractValidationError(
                    f"fields_used declares missing field '{field}' on record '{record_id}'"
                )
            if record[field] is None:
                raise ContractValidationError(
                    f"fields_used declares null field '{field}' on record '{record_id}'"
                )
        normalized[record_id] = tuple(sorted(fields))

    # Canonical mapping: records with no explicitly declared fields may remain absent.
    return dict(sorted(normalized.items()))


def validate_output(
    output: Any,
    records: Sequence[Mapping[str, Any]],
) -> ValidatedContract:
    """Validate a complete generated-code return value against supplied records."""
    record_map, supplied_ids = validate_input_records(records)
    supplied_set = set(supplied_ids)

    if not isinstance(output, Mapping):
        raise ContractValidationError("solve must return an object")

    expected_keys = {"result", "inputs_used", "excluded", "assumptions", "fields_used"}
    actual_keys = set(output.keys())
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        pieces = []
        if missing:
            pieces.append(f"missing keys: {missing}")
        if extra:
            pieces.append(f"unexpected keys: {extra}")
        raise ContractValidationError("invalid return shape (" + "; ".join(pieces) + ")")

    # Strict JSON validation of result and all audit values.
    try:
        canonical_json(output["result"])
        canonical_json(output["inputs_used"])
        canonical_json(output["excluded"])
        canonical_json(output["assumptions"])
        canonical_json(output["fields_used"])
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"return value is not strict JSON: {exc}") from exc

    inputs_used = _validate_string_list(output["inputs_used"], "inputs_used")
    unknown_used = sorted(set(inputs_used) - supplied_set)
    if unknown_used:
        raise ContractValidationError(f"inputs_used contains unknown record ids: {unknown_used}")

    excluded = _validate_excluded(output["excluded"])
    excluded_ids = {
        record_id
        for group in excluded
        for record_id in group["record_ids"]
    }
    unknown_excluded = sorted(excluded_ids - supplied_set)
    if unknown_excluded:
        raise ContractValidationError(f"excluded contains unknown record ids: {unknown_excluded}")

    used_set = set(inputs_used)
    overlap = sorted(used_set.intersection(excluded_ids))
    if overlap:
        raise ContractValidationError(f"records cannot be both inputs_used and excluded: {overlap}")

    accounted = used_set.union(excluded_ids)
    missing_accounting = [record_id for record_id in supplied_ids if record_id not in accounted]
    if missing_accounting:
        raise ContractValidationError(
            f"every supplied record must be accounted for; unaccounted ids: {missing_accounting}"
        )

    if not isinstance(output["assumptions"], list):
        raise ContractValidationError("assumptions must be a list")
    assumptions: list[str] = []
    for index, assumption in enumerate(output["assumptions"]):
        if not isinstance(assumption, str) or not assumption.strip():
            raise ContractValidationError(f"assumptions[{index}] must be a non-empty string")
        if len(assumption) > MAX_ASSUMPTION_CHARS:
            raise ContractValidationError(f"assumptions[{index}] is too long")
        assumptions.append(assumption.strip())

    fields_used = _validate_fields_used(output["fields_used"], record_map, used_set)

    canonical_inputs = tuple(sorted(inputs_used))
    if len(canonical_inputs) != len(inputs_used):
        raise ContractValidationError("inputs_used contains duplicate ids")

    canonical_excluded: list[dict[str, Any]] = []
    for group in excluded:
        canonical_excluded.append(
            {
                "record_ids": list(group["record_ids"]),
                "reason_code": group["reason_code"],
                "reason": group["reason"],
            }
        )

    return ValidatedContract(
        result=output["result"],
        inputs_used=canonical_inputs,
        excluded=tuple(canonical_excluded),
        fields_used=fields_used,
        assumptions=tuple(sorted(set(assumptions))),
    )


def select_records(records: Sequence[Mapping[str, Any]], record_ids: Iterable[str]) -> list[Mapping[str, Any]]:
    wanted = set(record_ids)
    return [record for record in records if record.get("id") in wanted]


def format_excluded_hint(excluded_ids: Sequence[str], max_ids: int = 20) -> str:
    ids = list(excluded_ids)
    if not ids:
        return ""
    shown = ids[:max_ids]
    suffix = "" if len(ids) <= max_ids else f" (+{len(ids) - max_ids} more)"
    return "Affected excluded record ids: " + ", ".join(shown) + suffix


__all__ = [
    "ContractValidationError",
    "REASON_CODES",
    "ValidatedContract",
    "canonical_json",
    "format_excluded_hint",
    "select_records",
    "validate_input_records",
    "validate_output",
]
