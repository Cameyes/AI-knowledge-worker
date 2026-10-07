"""Deterministic coverage tracking for agentic enumeration tasks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CoverageState = Literal["pending", "extracted", "no_match", "failed", "skipped"]
CandidateBasis = Literal["exhaustive", "search_selected"]
TERMINAL_STATES = frozenset({"extracted", "no_match", "failed", "skipped"})


class CoverageError(ValueError):
    """The coverage ledger operation is invalid."""


@dataclass(frozen=True)
class CoverageEntry:
    doc_id: str
    state: CoverageState = "pending"
    reason: str | None = None

    def validate(self) -> None:
        if not self.doc_id:
            raise CoverageError("doc_id must be non-empty")
        if self.state not in {"pending", "extracted", "no_match", "failed", "skipped"}:
            raise CoverageError(f"invalid coverage state: {self.state!r}")
        if self.state == "pending" and self.reason is not None:
            raise CoverageError("pending entries cannot carry a reason")
        if self.state in {"no_match", "failed", "skipped"} and not self.reason:
            raise CoverageError(f"state {self.state!r} requires a reason")


@dataclass
class CoverageLedger:
    candidate_basis: CandidateBasis
    basis_detail: str
    entries: list[CoverageEntry]

    def __post_init__(self) -> None:
        if self.candidate_basis not in {"exhaustive", "search_selected"}:
            raise CoverageError(f"invalid candidate basis: {self.candidate_basis!r}")
        if not self.basis_detail.strip():
            raise CoverageError("basis_detail must be non-empty")
        seen: set[str] = set()
        normalized: list[CoverageEntry] = []
        for entry in self.entries:
            entry.validate()
            if entry.doc_id in seen:
                raise CoverageError(f"duplicate doc_id in ledger: {entry.doc_id!r}")
            seen.add(entry.doc_id)
            normalized.append(entry)
        self.entries = normalized

    @classmethod
    def for_documents(
        cls,
        doc_ids: list[str],
        *,
        candidate_basis: CandidateBasis,
        basis_detail: str,
    ) -> "CoverageLedger":
        return cls(
            candidate_basis=candidate_basis,
            basis_detail=basis_detail,
            entries=[CoverageEntry(doc_id=doc_id) for doc_id in doc_ids],
        )

    def _index(self, doc_id: str) -> int:
        for i, entry in enumerate(self.entries):
            if entry.doc_id == doc_id:
                return i
        raise CoverageError(f"doc_id {doc_id!r} is not registered in the candidate set")

    def get(self, doc_id: str) -> CoverageEntry:
        return self.entries[self._index(doc_id)]

    def mark(self, doc_id: str, state: CoverageState, reason: str | None = None) -> CoverageEntry:
        """Set the latest state for a registered candidate document.

        Re-marking is allowed so a failed/no-match candidate can be retried after
        the agent changes its retrieval or extraction strategy. The trace layer is
        responsible for preserving the history of those attempts.
        """
        updated = CoverageEntry(doc_id=doc_id, state=state, reason=reason)
        updated.validate()
        idx = self._index(doc_id)
        self.entries[idx] = updated
        return updated

    def mark_extracted(self, doc_id: str) -> CoverageEntry:
        return self.mark(doc_id, "extracted")

    def mark_no_match(self, doc_id: str, reason: str) -> CoverageEntry:
        return self.mark(doc_id, "no_match", reason)

    def mark_failed(self, doc_id: str, reason: str) -> CoverageEntry:
        return self.mark(doc_id, "failed", reason)

    def mark_skipped(self, doc_id: str, reason: str) -> CoverageEntry:
        return self.mark(doc_id, "skipped", reason)

    def mark_pending(self, doc_id: str) -> CoverageEntry:
        return self.mark(doc_id, "pending")

    def complete(self) -> bool:
        """Architecture definition: no pending or failed entries remain."""
        return all(entry.state not in {"pending", "failed"} for entry in self.entries)

    def fraction_examined(self) -> float:
        """Fraction of candidate documents actually examined.

        `extracted` and `no_match` count as examined. `pending`, `failed`, and
        `skipped` do not. An empty candidate set is treated as 1.0.
        """
        if not self.entries:
            return 1.0
        examined = sum(entry.state in {"extracted", "no_match"} for entry in self.entries)
        return examined / len(self.entries)

    def counts(self) -> dict[str, int]:
        return {
            state: sum(entry.state == state for entry in self.entries)
            for state in ("pending", "extracted", "no_match", "failed", "skipped")
        }

    def failed_or_pending(self) -> tuple[CoverageEntry, ...]:
        return tuple(entry for entry in self.entries if entry.state in {"pending", "failed"})
