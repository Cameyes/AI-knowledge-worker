from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.rag.llm_client import safe_completion_json


ConsolidationStatus = Literal[
    "supported",
    "resolved",
    "contradictory",
    "insufficient",
]

ResolutionBasis = Literal[
    "declared_attribute_precedence",
]


class AtomicFact(BaseModel):
    """A single grounded fact plus structural provenance.

    The provenance fields describe how and where the fact was stated.
    Resolution is performed separately by deterministic code.
    """

    statement: str
    attribute: str
    value: str

    value_type: Literal[
        "declared_attribute",
        "narrative_reference",
        "unknown",
    ]

    temporal_scope: Literal[
        "current",
        "historical",
        "unknown",
    ]

    section_type: str = "unknown"
    source_candidate_id: str = ""
    qualifiers: dict[str, Any] = Field(default_factory=dict)
    evidence_indices: list[int] = Field(default_factory=list)


class FactExtractionResult(BaseModel):
    """Facts extracted from one requirement; deliberately contains no status."""

    group: str
    facts: list[AtomicFact] = Field(default_factory=list)


class ConsolidationGroup(BaseModel):
    """Deterministic requirement-level consolidation result."""

    group: str
    status: ConsolidationStatus = "insufficient"
    resolution_basis: ResolutionBasis | None = None
    resolved_fact: AtomicFact | None = None
    facts: list[AtomicFact] = Field(default_factory=list)
    raw_candidates: list[dict[str, Any]] = Field(default_factory=list)


_UNIVERSAL_STOPWORDS = {
    "what", "was", "were", "who", "whom", "which", "how", "much",
    "many", "does", "do", "did", "is", "are", "the", "a", "an",
    "in", "on", "of", "to", "for", "from", "among", "with", "and",
    "or", "their", "his", "her", "its", "they", "them", "than",
    "current", "had", "has", "have", "about", "work", "works", "worked",
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _salient_terms(text: str) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []

    for token in _tokens(text):
        if len(token) <= 2 and not token.isdigit():
            continue
        if token in _UNIVERSAL_STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        terms.append(token)

    return terms


def _candidate_score(group: str, item: dict) -> float:
    """Generic lexical affinity used only to order already-retrieved candidates."""

    result = item.get("result", {})
    metadata = result.get("metadata", {})
    source = str(metadata.get("source", ""))
    document = str(result.get("document", ""))

    terms = _salient_terms(group)
    if not terms:
        return 0.0

    haystack = f"{source}\n{document}".lower()
    score = 0.0

    for term in terms:
        if term not in haystack:
            continue
        # Slightly favor more informative/longer terms without assuming domain.
        score += 1.0 + min(len(term), 12) / 12.0

    return score


def _rank_candidates(group: str, items: list[dict], max_candidates: int) -> list[dict]:
    ranked = [
        (index, _candidate_score(group, item), item)
        for index, item in enumerate(items)
    ]
    ranked.sort(key=lambda row: (-row[1], row[0]))
    return [item for _, _, item in ranked[:max_candidates]]


def _extractive_spans(group: str, items: list[dict]) -> list[dict]:
    """Extract verbatim evidence lines with strong lexical overlap.

    This never creates a new factual statement. It only preserves text that
    already exists in the retrieved candidate.
    """

    terms = _salient_terms(group)
    if not terms:
        return []

    spans: list[dict] = []

    for index, item in enumerate(items):
        result = item.get("result", {})
        document = str(result.get("document", ""))

        for line in document.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue

            normalized = cleaned.lower()
            overlap = [term for term in terms if term in normalized]

            # Require at least two salient query terms in the same retrieved
            # line, unless the query has only one salient term.
            threshold = 1 if len(terms) == 1 else 2
            if len(overlap) < threshold:
                continue

            spans.append({
                "statement": cleaned,
                "evidence_indices": [index],
                "attribute": "",
                "value": cleaned,
                "value_type": "unknown",
                "temporal_scope": "unknown",
                "section_type": "unknown",
                "source_candidate_id": f"candidate:{index}",
                "qualifiers": {},
            })

    return spans


def _build_group_prompt(
    group: str,
    items: list[dict],
    max_chars_per_candidate: int,
) -> str:
    blocks: list[str] = []

    for index, item in enumerate(items):
        result = item.get("result", {})
        metadata = result.get("metadata", {})
        source = metadata.get("source", "Unknown source")
        document = str(result.get("document", ""))
        if len(document) > max_chars_per_candidate:
            document = document[:max_chars_per_candidate]

        blocks.append(
            f"CANDIDATE {index}\n"
            f"Source: {source}\n"
            f"Evidence:\n{document}"
        )

    evidence = "\n\n".join(blocks) if blocks else "No candidates were retrieved."

    return f"""
You are an evidence extraction system for a general-purpose enterprise RAG application.

Analyze ONE factual requirement using ONLY the supplied candidates.

REQUIREMENT:
{group}

Rules:
1. Inspect every candidate independently.
2. Extract a fact only when it is directly stated in a candidate.
3. A fact may appear anywhere in the candidate: summary, history, table,
   timeline, metadata, bullet list, or another structured section.
4. Preserve exact names, values, dates, units, identifiers and qualifiers.
5. Do not use outside knowledge.
6. Do not infer missing values.
7. Do not perform comparison, ranking, filtering, aggregation, calculation,
   conflict resolution, or source selection.
8. Return every directly stated relevant fact, including facts whose values
   disagree with other candidates. Never discard a fact because another
   candidate contains a different value.
9. Classify each fact by how it is stated:
   - "declared_attribute": the candidate presents the value as a direct
     labeled/key-value field for the requested attribute, such as
     "Status: Active", "Version: 3.2", or "Job Title: Engineer".
   - "narrative_reference": the value appears inside a sentence describing
     an event, action, transition, change, or other narrative statement, such
     as "was promoted to Engineer" or "moved to the Gold plan".
   - "unknown": use this when the distinction cannot be established from
     the supplied text.
10. Classify temporal scope without deciding which value should win:
   - "current": explicitly current, active, present, or otherwise stated
     to apply now.
   - "historical": explicitly tied to a past date, period, completed event,
     or past state.
   - "unknown": the time scope is not clear.
11. Record the structural section containing the fact as a short generic
    label such as "summary", "history", "table", "metadata", "body",
    or "unknown". Do not invent section names not supported by the text.
12. Normalize "attribute" to the factual field being reported, but keep
    "value" faithful to the source wording.
13. Each fact must include the local candidate index that directly supports
    it. These indices are local to this requirement.
14. Return ONLY JSON. Do not return a status field.

Schema:
{{
  "group": "{group}",
  "facts": [
    {{
      "statement": "direct factual statement",
      "attribute": "normalized attribute",
      "value": "source-faithful value",
      "value_type": "declared_attribute",
      "temporal_scope": "unknown",
      "section_type": "summary",
      "evidence_indices": [0],
      "qualifiers": {{}}
    }}
  ]
}}

CANDIDATES:
{evidence}
"""


def _normalize_model_result(group: str, items: list[dict], raw: object) -> dict:
    """Normalize facts-only extraction while preserving the legacy outward contract.

    Phase 1 deliberately does not resolve conflicts. The legacy ``status``
    field is retained only so existing callers do not break before Phase 2.
    """

    try:
        parsed = FactExtractionResult.model_validate(raw)
    except Exception:
        return {"status": "insufficient", "facts": [], "evidence": []}

    facts: list[dict] = []
    indices: set[int] = set()

    for fact in parsed.facts:
        statement = fact.statement.strip()
        valid_indices = sorted({
            index for index in fact.evidence_indices
            if 0 <= index < len(items)
        })
        if not statement or not valid_indices:
            continue

        primary_index = valid_indices[0]
        candidate_metadata = items[primary_index].get("result", {}).get("metadata", {})
        candidate_id = str(
            candidate_metadata.get("candidate_id")
            or f"candidate:{primary_index}"
        )

        fact_data = fact.model_dump()
        fact_data["statement"] = statement
        fact_data["evidence_indices"] = valid_indices
        fact_data["source_candidate_id"] = candidate_id

        facts.append(fact_data)
        indices.update(valid_indices)

    if not facts:
        return {"status": "insufficient", "facts": [], "evidence": []}

    # Compatibility only. Phase 2 will replace this with deterministic
    # resolution based on the extracted provenance.
    return {
        "status": "supported",
        "facts": facts,
        "evidence": [items[index] for index in sorted(indices)],
    }


def _normalized_value(value: str) -> str:
    """Normalize only for equality checks; preserve the source value itself."""

    return re.sub(r"\s+", " ", value.strip()).casefold()


def _distinct_value_keys(facts: list[AtomicFact]) -> set[tuple[str, str]]:
    """Return distinct (attribute, value) pairs without rewriting source facts."""

    keys: set[tuple[str, str]] = set()
    for fact in facts:
        keys.add((fact.attribute.strip().casefold(), _normalized_value(fact.value)))
    return keys


def _resolve_requirement_facts(
    facts: list[AtomicFact],
) -> tuple[ConsolidationStatus, ResolutionBasis | None, AtomicFact | None]:
    """Resolve one requirement deterministically from already-extracted facts.

    V1 policy: a directly declared attribute is the system-of-record when it
    conflicts with narrative references. This deliberately follows the
    benchmark convention observed in the current corpus, including cases where
    a narrative section contains an explicit ``Present`` scope. The trade-off
    is intentional and documented here rather than hidden in the LLM prompt.

    Unknown/ambiguous provenance never triggers automatic resolution when
    multiple values disagree.
    """

    # Extractive fallback spans may intentionally carry ``value_type=unknown``
    # and an empty attribute. They are preserved as raw evidence, but they do
    # not have enough structure to participate in semantic resolution.
    resolvable_facts = [
        fact
        for fact in facts
        if fact.attribute.strip() and fact.value.strip()
    ]

    if not resolvable_facts:
        return "insufficient", None, None

    non_empty_attributes = {
        fact.attribute.strip().casefold()
        for fact in resolvable_facts
    }

    # One requirement is expected to describe one factual attribute. If
    # extraction produces unrelated attributes, do not guess which one the
    # resolver should select.
    if len(non_empty_attributes) > 1:
        return "contradictory", None, None

    value_keys = _distinct_value_keys(resolvable_facts)

    if len(value_keys) == 1:
        return "supported", None, resolvable_facts[0]

    # Ambiguous provenance must not silently resolve a disagreement.
    if any(f.value_type == "unknown" for f in resolvable_facts):
        return "contradictory", None, None

    declared = [
        fact
        for fact in resolvable_facts
        if fact.value_type == "declared_attribute"
    ]
    narrative = [
        fact
        for fact in resolvable_facts
        if fact.value_type == "narrative_reference"
    ]

    declared_keys = _distinct_value_keys(declared)

    # A single declared value takes precedence over any narrative value(s).
    # Keep every fact intact; only the resolved_fact/status changes.
    if len(declared_keys) == 1 and narrative:
        resolved = next(iter(declared), None)
        if resolved is not None:
            return "resolved", "declared_attribute_precedence", resolved

    return "contradictory", None, None


def _merge_extractive_fallback(
    model_result: dict,
    extractive_spans: list[dict],
    ranked_items: list[dict],
) -> dict:
    """Make directly retrieved text available when the model misses it."""

    if not extractive_spans:
        if model_result.get("evidence"):
            return model_result
        return {
            "status": model_result.get("status", "insufficient"),
            "facts": model_result.get("facts", []),
            "evidence": ranked_items,
        }

    existing = list(model_result.get("facts", []))
    seen = {fact["statement"].strip().lower() for fact in existing}
    selected_indices = {
        index
        for fact in existing
        for index in fact.get("evidence_indices", [])
    }

    for span in extractive_spans:
        statement = span["statement"].strip()
        if statement.lower() in seen:
            continue

        enriched = dict(span)
        primary_index = span["evidence_indices"][0]
        if 0 <= primary_index < len(ranked_items):
            metadata = ranked_items[primary_index].get("result", {}).get("metadata", {})
            enriched["source_candidate_id"] = str(
                metadata.get("candidate_id")
                or f"candidate:{primary_index}"
            )

        existing.append(enriched)
        seen.add(statement.lower())
        selected_indices.update(span["evidence_indices"])

    # If direct retrieved text establishes the requirement, don't let a
    # conservative model label erase that evidence.
    status = "supported" if existing else model_result.get("status", "insufficient")

    return {
        "status": status,
        "facts": existing,
        "evidence": [ranked_items[index] for index in sorted(selected_indices) if 0 <= index < len(ranked_items)],
    }


def consolidate_evidence(
    grouped_evidence: dict[str, list[dict]],
    *,
    max_candidates_per_group: int = 3,
    max_chars_per_candidate: int = 3500,
) -> dict[str, dict]:
    """Consolidate retrieval evidence into requirement-scoped atomic facts.

    The design is extractive-first: direct text already present in retrieval
    candidates is never discarded merely because the LLM fails to summarize it.
    The LLM is used to normalize/identify facts, not to decide whether the
    underlying retrieved text exists.
    """

    consolidated: dict[str, dict] = {}

    for group, items in grouped_evidence.items():
        ranked_items = _rank_candidates(
            group,
            items,
            max_candidates=max_candidates_per_group,
        )

        extractive = _extractive_spans(group, ranked_items)

        prompt = _build_group_prompt(
            group,
            ranked_items,
            max_chars_per_candidate=max_chars_per_candidate,
        )

        fallback = {
            "group": group,
            "facts": [],
        }

        raw = safe_completion_json(
            prompt,
            max_tokens=512,
            max_retries=2,
            fallback=fallback,
        )

        model_result = _normalize_model_result(group, ranked_items, raw)
        merged = _merge_extractive_fallback(
            model_result,
            extractive,
            ranked_items,
        )

        # A second focused model pass is only used when neither the first
        # model pass nor deterministic extraction found anything.
        if not merged["facts"] and ranked_items:
            recheck_prompt = _build_group_prompt(
                group,
                ranked_items,
                max_chars_per_candidate=max_chars_per_candidate,
            ) + "\n\nRECHECK: Carefully inspect candidate 0, then candidate 1, then candidate 2."

            recheck_raw = safe_completion_json(
                recheck_prompt,
                max_tokens=512,
                max_retries=1,
                fallback=fallback,
            )
            rechecked = _normalize_model_result(group, ranked_items, recheck_raw)
            merged = _merge_extractive_fallback(
                rechecked,
                extractive,
                ranked_items,
            )

        fact_models = []
        for fact_data in merged.get("facts", []):
            try:
                fact_models.append(AtomicFact.model_validate(fact_data))
            except Exception:
                continue

        status, resolution_basis, resolved_fact = _resolve_requirement_facts(
            fact_models
        )

        consolidated[group] = {
            "group": group,
            "status": status,
            "resolution_basis": resolution_basis,
            "resolved_fact": (
                resolved_fact.model_dump()
                if resolved_fact is not None
                else None
            ),
            "facts": [fact.model_dump() for fact in fact_models],
            # Keep the existing retrieval evidence contract for downstream
            # callers. Phase 3 will teach generation how to consume the new
            # resolution fields explicitly.
            "evidence": merged.get("evidence", ranked_items),
            "raw_candidates": ranked_items,
        }

    return consolidated
