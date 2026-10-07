"""Host-side evidence records and verified evidence store.

This module deliberately has no LLM or database dependency. It turns a candidate
record into trusted evidence only after the host verifies every supporting quote
verbatim inside the declared document span.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha1
from typing import Any, Iterable


class EvidenceValidationError(ValueError):
    """The record is malformed or its evidence cannot be verified."""


class EvidenceConflictError(ValueError):
    """The same evidence id was already stored with different content."""


@dataclass(frozen=True)
class EvidenceRecord:
    id: str
    doc_id: str
    span: tuple[int, int]
    entity: str | None
    schema: str
    fields: dict[str, Any]
    quotes: dict[str, str]
    period: str | None
    extractor: str
    verified: bool = False

    def validate_structure(self) -> None:
        if not self.id:
            raise EvidenceValidationError("record id must be non-empty")
        if not self.doc_id:
            raise EvidenceValidationError("doc_id must be non-empty")
        if not self.schema:
            raise EvidenceValidationError("schema must be non-empty")
        if not self.extractor:
            raise EvidenceValidationError("extractor must be non-empty")
        if not isinstance(self.span, tuple) or len(self.span) != 2:
            raise EvidenceValidationError("span must be a (start, end) tuple")
        start, end = self.span
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
            raise EvidenceValidationError(f"invalid span: {self.span!r}")
        if not isinstance(self.fields, dict):
            raise EvidenceValidationError("fields must be a dict")
        if not isinstance(self.quotes, dict):
            raise EvidenceValidationError("quotes must be a dict")
        unknown_quotes = set(self.quotes) - set(self.fields)
        if unknown_quotes:
            raise EvidenceValidationError(
                f"quotes contain fields not present in fields: {sorted(unknown_quotes)!r}"
            )

        # Architecture invariant: every non-null value needs supporting evidence.
        for field_name, value in self.fields.items():
            if value is None:
                continue
            quote = self.quotes.get(field_name)
            if not isinstance(quote, str) or not quote:
                raise EvidenceValidationError(
                    f"non-null field {field_name!r} must have a non-empty supporting quote"
                )


def make_record_id(doc_id: str, span: tuple[int, int], schema: str, field: str) -> str:
    """Stable id following the architecture's documented identity formula.

    A record may contain multiple fields; the extraction layer supplies the field
    used as the record identity key. That keeps the documented id shape without
    adding another persisted field to the Record contract.
    """
    start, end = span
    raw = f"{doc_id}|{start}|{end}|{schema}|{field}".encode("utf-8")
    return sha1(raw).hexdigest()[:12]


def verify_record(record: EvidenceRecord, source_text: str) -> EvidenceRecord:
    """Verify all non-null field quotes inside the record's declared source span.

    Verification is exact substring matching; no normalization or fuzzy matching
    is allowed because the architecture requires a verbatim quote.
    """
    record.validate_structure()
    if not isinstance(source_text, str):
        raise EvidenceValidationError("source_text must be a string")

    start, end = record.span
    if end > len(source_text):
        raise EvidenceValidationError(
            f"record span {record.span!r} exceeds document length {len(source_text)}"
        )

    span_text = source_text[start:end]
    for field_name, value in record.fields.items():
        if value is None:
            continue
        quote = record.quotes[field_name]
        if quote not in span_text:
            raise EvidenceValidationError(
                f"quote for field {field_name!r} was not found verbatim inside span {record.span!r}"
            )

    # The host, not the model, sets verified=True.
    return replace(record, verified=True)


class EvidenceStore:
    """In-memory host-side store for verified evidence records.

    Persistence belongs above this small contract. The store's job is to enforce
    provenance/quote invariants and provide deterministic lookup operations.
    """

    def __init__(self, records: Iterable[EvidenceRecord] | None = None) -> None:
        self._records: dict[str, EvidenceRecord] = {}
        if records:
            for record in records:
                self.add_verified(record)

    def add(self, record: EvidenceRecord, source_text: str) -> EvidenceRecord:
        """Verify a candidate record against source_text, then store the verified copy."""
        verified = verify_record(record, source_text)
        return self.add_verified(verified)

    def add_verified(self, record: EvidenceRecord) -> EvidenceRecord:
        """Store only a record already marked verified by this host-side verifier."""
        record.validate_structure()
        if not record.verified:
            raise EvidenceValidationError("cannot store an unverified evidence record")

        existing = self._records.get(record.id)
        if existing is not None and existing != record:
            raise EvidenceConflictError(f"evidence id {record.id!r} already exists with different content")
        self._records[record.id] = record
        return record

    def get(self, record_id: str) -> EvidenceRecord | None:
        return self._records.get(record_id)

    def require(self, record_id: str) -> EvidenceRecord:
        record = self.get(record_id)
        if record is None:
            raise KeyError(record_id)
        return record

    def for_doc(self, doc_id: str) -> tuple[EvidenceRecord, ...]:
        return tuple(r for r in self._records.values() if r.doc_id == doc_id)

    def all(self) -> tuple[EvidenceRecord, ...]:
        # Dict insertion order is deterministic for the lifetime of the store.
        return tuple(self._records.values())

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, record_id: str) -> bool:
        return record_id in self._records
