from __future__ import annotations

import json
import logging
import os
import re
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.rag.llm_client import safe_completion_json


logger = logging.getLogger(__name__)

_PROVENANCE_VERIFIER_MODES = {"off", "shadow", "enforce"}
_DEFAULT_PROVENANCE_VERIFIER_MODE = "shadow"


def _provenance_verifier_mode() -> str:
    mode = os.getenv("RAG_PROVENANCE_VERIFIER_MODE", _DEFAULT_PROVENANCE_VERIFIER_MODE)
    mode = mode.strip().lower()
    return mode if mode in _PROVENANCE_VERIFIER_MODES else _DEFAULT_PROVENANCE_VERIFIER_MODE


ConsolidationStatus = Literal[
    "supported",
    "resolved",
    "contradictory",
    "insufficient",
]

ResolutionBasis = Literal[
    "declared_attribute_precedence",
    "temporal_current_precedence",
    "temporal_latest_state",
    "llm_ambiguity_resolution",
]


class AtomicFact(BaseModel):
    """A single grounded fact plus structural provenance.

    The provenance fields describe how and where the fact was stated.
    Resolution is performed separately by deterministic code.
    """

    statement: str
    attribute: str
    value: str
    subject: str = ""

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
    effective_start: str = ""
    effective_end: str = ""
    effective_end_open: bool = False

    section_type: str = "unknown"
    source_candidate_id: str = ""
    source_line: str = ""
    declaration_label: str = ""
    qualifiers: dict[str, Any] = Field(default_factory=dict)
    evidence_indices: list[int] = Field(default_factory=list)


class FactExtractionResult(BaseModel):
    """Facts extracted from one requirement; deliberately contains no status."""

    group: str
    facts: list[AtomicFact] = Field(default_factory=list)


class ProvenanceVerdict(BaseModel):
    """Semantic provenance verdict for one declaration candidate."""

    fact_index: int
    verdict: Literal["attribute_label", "contextual_label", "unknown"]


class ProvenanceVerificationResult(BaseModel):
    """Closed-set provenance verdicts returned by the semantic verifier."""

    verdicts: list[ProvenanceVerdict] = Field(default_factory=list)


class TemporalStateVerdict(BaseModel):
    """Semantic verdict about whether a dated statement establishes a continuing state."""

    fact_index: int
    verdict: Literal["state_assignment", "event_only", "unknown"]


class TemporalStateVerificationResult(BaseModel):
    """Closed-set temporal-state verdicts returned by the semantic verifier."""

    verdicts: list[TemporalStateVerdict] = Field(default_factory=list)


class EntityScopeVerdict(BaseModel):
    """Whether a retrieved candidate is about the requested target entity."""

    verdict: Literal["match", "mismatch", "unknown"]


class AmbiguityResolution(BaseModel):
    """Closed-set semantic resolution for genuinely ambiguous extracted facts."""

    verdict: Literal["resolved", "ambiguous", "insufficient"]
    selected_fact_index: int | None = None
    reason: str = ""


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


def _normalize_identity(text: str) -> str:
    """Normalize an entity/document identity for exact generic matching."""
    text = str(text or "").casefold()
    text = re.sub(r"\.[a-z0-9]{1,8}$", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _source_basename(source: str) -> str:
    cleaned = str(source or "").replace("\\", "/").strip()
    return cleaned.rsplit("/", 1)[-1] if cleaned else ""


def _candidate_identity_aliases(item: dict) -> list[str]:
    """Return generic identity aliases from metadata, source name, and document heading."""
    result = item.get("result", {})
    metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
    metadata = metadata if isinstance(metadata, dict) else {}

    aliases: list[str] = []
    seen: set[str] = set()

    def add(value: object) -> None:
        if not isinstance(value, str) or not value.strip():
            return
        normalized = _normalize_identity(value)
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        aliases.append(value.strip())

    for key in ("entity", "subject", "document_entity", "entity_name", "document_name"):
        add(metadata.get(key))

    source = metadata.get("source") or metadata.get("filename") or ""
    filename = _source_basename(str(source))
    if filename:
        add(re.sub(r"\.[^.]+$", "", filename))

    document = str(result.get("document", "")) if isinstance(result, dict) else ""
    for line in document.splitlines()[:20]:
        match = re.match(r"^\s*#\s+(.+?)\s*$", line)
        if match:
            add(match.group(1).strip(" #"))

    return aliases


def _candidate_identity(item: dict) -> str:
    """Return the strongest available identity alias for display/fallback use."""
    aliases = _candidate_identity_aliases(item)
    return aliases[0] if aliases else ""


def _infer_target_entities(group: str, items: list[dict]) -> list[str]:
    """Infer explicit targets only when a candidate identity alias is named in the requirement."""
    group_norm = _normalize_identity(group)
    targets: list[str] = []
    seen: set[str] = set()
    for item in items:
        for identity in _candidate_identity_aliases(item):
            normalized = _normalize_identity(identity)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            if re.search(rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])", group_norm):
                targets.append(identity)
                break
    return targets


def _source_matches_targets(item: dict, target_entities: list[str]) -> str:
    if not target_entities:
        return "unscoped"

    target_keys = {_normalize_identity(value) for value in target_entities}
    aliases = _candidate_identity_aliases(item)
    alias_keys = {_normalize_identity(value) for value in aliases}

    # Explicit entity metadata is a hard identity assertion.
    result = item.get("result", {})
    metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
    metadata = metadata if isinstance(metadata, dict) else {}
    explicit = []
    for key in ("entity", "subject", "document_entity", "entity_name"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            explicit.append(_normalize_identity(value))

    if explicit:
        return "match" if any(value in target_keys for value in explicit) else "mismatch"

    # Source filename/header is an identity anchor when it exactly matches the target.
    if alias_keys & target_keys:
        return "match"

    # A non-matching filename is not automatically a mismatch: generic RAG sources
    # may use opaque IDs. Let the narrow semantic verifier decide in that case.
    return "unknown"


def _build_entity_scope_verifier_prompt(group: str, target_entities: list[str], item: dict) -> str:
    result = item.get("result", {})
    metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
    metadata = metadata if isinstance(metadata, dict) else {}
    source = metadata.get("source") or metadata.get("filename") or "Unknown source"
    document = str(result.get("document", "")) if isinstance(result, dict) else ""
    aliases = ", ".join(_candidate_identity_aliases(item)) or "none"
    return f"""
You are an entity-scope verification component for a general-purpose enterprise RAG system.

Decide whether this candidate can be used for the explicitly requested target entity/entities.

Requirement:
{group}

Target entity/entities:
{', '.join(target_entities)}

Source:
{source}

Candidate identity aliases:
{aliases}

Candidate evidence:
{document[:1800]}

Allowed verdicts:
- match: the candidate contains the requested fact about a target entity.
- mismatch: the candidate is about a different entity/entity set.
- unknown: identity cannot be established reliably.

Do not answer the requirement. Do not resolve values. Return JSON only:
{{"verdict": "match"}}
"""


def _run_entity_scope_verifier(group: str, target_entities: list[str], item: dict) -> str:
    raw = safe_completion_json(
        _build_entity_scope_verifier_prompt(group, target_entities, item),
        max_tokens=128,
        max_retries=1,
        fallback={"verdict": "unknown"},
    )
    try:
        return EntityScopeVerdict.model_validate(raw).verdict
    except Exception:
        return "unknown"


def _scope_candidates(group: str, items: list[dict], target_entities: list[str] | None = None) -> tuple[list[dict], list[dict], list[str], str]:
    """Fail-closed entity firewall. Rejected candidates never reach resolution or generation."""
    targets = [str(v).strip() for v in (target_entities or []) if str(v).strip()]
    if not targets:
        targets = _infer_target_entities(group, items)

    if not targets:
        return list(items), [], [], "unscoped"

    accepted: list[dict] = []
    rejected: list[dict] = []
    unknown: list[dict] = []

    for item in items:
        classification = _source_matches_targets(item, targets)
        if classification == "match":
            accepted.append(item)
        elif classification == "mismatch":
            rejected.append(item)
        else:
            unknown.append(item)

    # Ambiguous sources are the only candidates that receive semantic identity verification.
    for item in unknown:
        if _run_entity_scope_verifier(group, targets, item) == "match":
            accepted.append(item)
        else:
            rejected.append(item)

    return accepted, rejected, targets, "explicit"


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
                "subject": _candidate_identity(items[index]),
                "evidence_indices": [index],
                "attribute": "",
                "value": cleaned,
                "value_type": "unknown",
                "temporal_scope": "unknown",
                "section_type": "unknown",
                "source_candidate_id": f"candidate:{index}",
                "source_line": "",
                "declaration_label": "",
                "qualifiers": {},
            })

    return spans




def _requested_years(requirement: str) -> set[str]:
    """Return explicit four-digit years requested by the requirement."""
    return set(re.findall(r"\b(?:19|20)\d{2}\b", requirement))


def _label_matches_requirement(label: str, requirement: str) -> bool:
    """Use conservative lexical overlap to identify a requested field label."""
    label_terms = set(_salient_terms(label))
    requirement_terms = set(_salient_terms(requirement))
    return bool(label_terms & requirement_terms)


def _label_has_explicit_current_marker(label: str) -> bool:
    """Return whether a field label explicitly denotes a current state/value."""
    text = _normalized_compact(label)
    return bool(re.search(r"\b(?:current|present|presently|ongoing|now)\b", text))


def _structured_field_spans(group: str, items: list[dict]) -> list[dict]:
    """Recover simple grounded key/value fields missed by model extraction.

    The extractor is domain-neutral. It only emits an atomic fact when a
    retrieved line visibly contains a field label whose terms overlap the
    requested requirement. Dated nested labels are accepted only when the
    requested year is explicit, preventing historical rows from being treated
    as the current value merely because they contain a familiar field label.
    """
    spans: list[dict] = []
    requested_years = _requested_years(group)

    for index, item in enumerate(items):
        result = item.get("result", {})
        document = str(result.get("document", "")) if isinstance(result, dict) else ""
        subject = _candidate_identity(item)

        for raw_line in document.splitlines():
            cleaned = _clean_markdown_text(raw_line)
            if not cleaned:
                continue

            # Split nested forms such as:
            #   Current Salary: $285,000
            #   2023: Rating: 4.7/5
            segments = [
                part.strip()
                for part in re.split(r"\s*:\s*", cleaned)
                if part.strip()
            ]
            if len(segments) < 2 and "|" in cleaned:
                segments = [
                    part.strip()
                    for part in re.split(r"\s*\|\s*", cleaned)
                    if part.strip()
                ]
            if len(segments) < 2:
                continue

            selected: tuple[str, str] | None = None
            selected_year = ""
            has_date_prefix = False

            for position in range(len(segments) - 1):
                label = segments[position]
                value = ":".join(segments[position + 1:]).strip()
                if not value:
                    continue

                # A date prefix is temporal context, not the requested field.
                # Only use it when that exact year was requested by the user.
                date_match = re.fullmatch(_DATE_TOKEN_RE, label, re.IGNORECASE)
                if date_match:
                    has_date_prefix = True
                    normalized_date = _normalize_date_token(label)
                    if normalized_date and normalized_date[:4] in requested_years:
                        selected_year = normalized_date
                    continue

                if _label_matches_requirement(label, group):
                    selected = (label, value)

            if selected is None:
                continue

            label, value = selected

            # For current/latest questions, a plain static field such as
            # "Job Title" must not be promoted into an additional current-state
            # fact because a later history entry may supersede it. Explicit
            # current/present labels remain eligible (for example,
            # "Current Salary: $285,000").
            temporal_intent = _query_temporal_intent(group)
            if (
                temporal_intent in {"current", "latest"}
                and not _label_has_explicit_current_marker(label)
            ):
                continue

            # A date-prefixed line is historical/scoped evidence. Do not use it
            # as a fallback for an undated/current request unless the requested
            # year is explicitly the same year carried by that line.
            if has_date_prefix and not selected_year:
                continue

            if selected_year and selected_year[:4] not in requested_years:
                continue

            # A simple two-part field is an actual key/value declaration. A
            # nested dated field is kept provenance-neutral because its date
            # prefix scopes the value rather than declaring the attribute.
            direct_field = len(segments) == 2 and not selected_year
            value_type = "declared_attribute" if direct_field else "unknown"

            qualifiers: dict[str, Any] = {"structured_rescue": True}
            if selected_year:
                qualifiers["effective_start"] = selected_year

            spans.append({
                "statement": cleaned,
                "subject": subject,
                "evidence_indices": [index],
                "attribute": label,
                "value": value,
                "value_type": value_type,
                "temporal_scope": "historical" if selected_year else "unknown",
                "effective_start": selected_year,
                "effective_end": "",
                "effective_end_open": False,
                "section_type": "unknown",
                "source_candidate_id": f"candidate:{index}",
                "source_line": cleaned if direct_field else "",
                "declaration_label": label if direct_field else "",
                "qualifiers": qualifiers,
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
3. For every fact, record `subject` as the entity the fact is actually about.
   If the subject is not established, leave it empty; never copy the target from the requirement.
4. A fact may appear anywhere in the candidate: summary, history, table,
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
   - "declared_attribute": use only when the candidate explicitly presents
     the requested attribute as a labeled/key-value field. A declaration must
     provide `source_line` (the verbatim source line containing the claim) and
     `declaration_label` (the verbatim field label introducing the value).
     The label must identify the requested attribute, not merely provide a
     date, sequence, scope, phase, location, event, or other context.
   - "narrative_reference": use when the value appears in a sentence, event,
     timeline entry, transition, history/log entry, contextual statement, or
     other text that does not explicitly declare the requested attribute.
   - "unknown": use when the provenance classification cannot be established
     reliably from the supplied text.
   Never classify something as a declaration merely because it has a colon,
   equals sign, pipe, bullet, table-like layout, or another key/value-looking
   shape.
10. Extract temporal applicability separately from the broad temporal scope:
   - "temporal_scope": "current" when the value is explicitly current, present,
     ongoing, or otherwise stated to apply now.
   - "temporal_scope": "historical" when it is explicitly tied to a past
     period/event or a completed state.
   - "temporal_scope": "unknown" when unclear.
   - "effective_start": the normalized date when the value became effective,
     using YYYY, YYYY-MM, or YYYY-MM-DD when supported; otherwise empty.
   - "effective_end": the normalized date when the value stopped being effective,
     using the same formats; otherwise empty. Use "present" only when the source
     explicitly says present/current/ongoing.
   - "effective_end_open": true when the source establishes that the value takes
     effect at a stated time and remains in force until a later change, without an
     explicit end date. This is especially applicable to state-changing assignments
     such as a value becoming a new state/role/version/plan at a dated event. Do not
     mark a completed one-time event as open-ended.
   Important: distinguish the date of an event from the effective period of the
   extracted attribute value. For a state-changing statement such as "promoted to
   X in August 2019", X can have effective_start=2019-08 and effective_end_open=true
   when the text establishes that X becomes the new state/attribute value and gives
   no later end. A later state-changing fact for the same attribute supersedes it.
11. Record the structural section containing the fact as a short generic
    label such as "summary", "history", "table", "metadata", "body",
    or "unknown". Do not invent section names not supported by the text.
12. Normalize "attribute" to the factual field being reported, but keep
    "value" faithful to the source wording.
13. Each fact must include the local candidate index that directly supports
    it. These indices are local to this requirement.
14. Return ONLY JSON. Do not return a status field.
15. Return only facts whose attribute directly answers the REQUIREMENT.
    Do not return unrelated names, compensation, performance, awards,
    locations, or other facts merely because they appear in the same candidate.
16. Prefer the smallest complete set of facts needed to resolve this one
    requirement. Do not repeat equivalent facts.
17. For current/latest requirements, inspect history/timeline/change entries
    for the requested attribute and preserve dated state-changing facts; do
    not stop after finding a summary field.

Schema:
{{
  "group": "{group}",
  "facts": [
    {{
      "statement": "direct factual statement",
      "subject": "entity the fact is about or empty if unknown",
      "attribute": "normalized attribute",
      "value": "source-faithful value",
      "value_type": "declared_attribute",
      "temporal_scope": "unknown",
      "effective_start": "",
      "effective_end": "",
      "effective_end_open": false,
      "section_type": "summary",
      "source_line": "verbatim source line",
      "declaration_label": "verbatim field label or empty string",
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
        _enrich_temporal_fields(fact_data)

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



_MONTHS = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}

_DATE_TOKEN_RE = (
    r"(?:"
    r"(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+(?:\d{1,2},?\s+)?\d{4}"
    r"|"
    r"\d{4}-\d{2}(?:-\d{2})?"
    r"|"
    r"\d{4}"
    r")"
)


def _normalize_date_token(value: str) -> str:
    """Normalize common date tokens to YYYY / YYYY-MM / YYYY-MM-DD."""
    text = re.sub(r"\s+", " ", value.strip().lower())
    if re.fullmatch(r"\d{4}(?:-\d{2}){0,2}", text):
        return text

    match = re.fullmatch(
        r"(january|february|march|april|may|june|july|august|september|"
        r"october|november|december)\s+(?:(\d{1,2}),?\s+)?(\d{4})",
        text,
    )
    if not match:
        return ""
    month = _MONTHS[match.group(1)]
    year = match.group(3)
    day = match.group(2)
    return f"{year}-{month}-{int(day):02d}" if day else f"{year}-{month}"


def _infer_effective_dates_from_text(text: str) -> tuple[str, str]:
    """Recover generic effective date/range metadata from grounded text."""
    cleaned = _clean_markdown_text(text)
    if not cleaned:
        return "", ""

    date_token = _DATE_TOKEN_RE
    range_pattern = re.compile(
        rf"(?P<start>{date_token})\s*(?:-|–|—|\bto\b)\s*"
        rf"(?P<end>{date_token}|present|current|ongoing)",
        re.IGNORECASE,
    )
    range_match = range_pattern.search(cleaned)
    if range_match:
        start = _normalize_date_token(range_match.group("start"))
        raw_end = range_match.group("end").strip().lower()
        end = "present" if raw_end in {"present", "current", "ongoing"} else _normalize_date_token(raw_end)
        return start, end

    # A date prefix before a colon is a common generic timeline/event form.
    prefix_pattern = re.compile(
        rf"^\s*(?P<start>{date_token})\s*:",
        re.IGNORECASE,
    )
    prefix_match = prefix_pattern.search(cleaned)
    if prefix_match:
        return _normalize_date_token(prefix_match.group("start")), ""

    # A change/event sentence can express the effective date inline.
    inline_pattern = re.compile(
        rf"\b(?:in|on|from|since)\s+(?P<start>{date_token})\b",
        re.IGNORECASE,
    )
    inline_match = inline_pattern.search(cleaned)
    if inline_match:
        return _normalize_date_token(inline_match.group("start")), ""

    return "", ""


def _enrich_temporal_fields(fact_data: dict[str, Any]) -> None:
    """Normalize temporal metadata from every available representation.

    Extraction models may place a date in canonical fields, qualifiers, the
    source line, or the statement. The resolver must see one canonical view
    regardless of where the model happened to put that information.
    """
    source = str(fact_data.get("source_line", "")).strip()
    statement = str(fact_data.get("statement", "")).strip()
    qualifiers = fact_data.get("qualifiers")
    qualifiers = qualifiers if isinstance(qualifiers, dict) else {}

    start = str(fact_data.get("effective_start", "")).strip()
    end = str(fact_data.get("effective_end", "")).strip()

    # First honor already-canonical fields. Then inspect common generic
    # temporal qualifier names emitted by extraction models.
    if not start:
        for key in ("effective_start", "start_date", "start", "effective_date", "date", "year"):
            candidate = str(qualifiers.get(key, "")).strip()
            if candidate:
                normalized = _normalize_date_token(candidate)
                if normalized:
                    start = normalized
                    fact_data["effective_start"] = normalized
                    break
    if not end:
        for key in ("effective_end", "end_date", "end"):
            candidate = str(qualifiers.get(key, "")).strip()
            if candidate:
                normalized = _normalize_date_token(candidate)
                if normalized:
                    end = normalized
                    fact_data["effective_end"] = normalized
                    break

    # Source text is the strongest fallback because it is grounded in the
    # retrieved evidence rather than being model-generated metadata.
    for text in (source, statement):
        if not text:
            continue
        inferred_start, inferred_end = _infer_effective_dates_from_text(text)
        if not start and inferred_start:
            start = inferred_start
            fact_data["effective_start"] = inferred_start
        if not end and inferred_end:
            end = inferred_end
            fact_data["effective_end"] = inferred_end
        if inferred_end == "present":
            fact_data["effective_end_open"] = True
        if start:
            break


def _clean_markdown_text(text: str) -> str:
    """Normalize common lightweight markdown for structural comparison."""
    cleaned = str(text).strip()
    cleaned = re.sub(r"^\s*(?:[-*+]\s+|>\s+)+", "", cleaned)
    cleaned = re.sub(r"^\s*#{1,6}\s+", "", cleaned)
    cleaned = cleaned.replace("**", "").replace("__", "").replace("`", "")
    return re.sub(r"\s+", " ", cleaned).strip()


def _normalized_compact(text: str) -> str:
    return re.sub(r"\s+", " ", _clean_markdown_text(text)).strip().casefold()


def _find_source_line(source_line: str, evidence_texts: list[str]) -> bool:
    """Return whether the claimed line is actually present in cited evidence."""
    target = _normalized_compact(source_line)
    if not target:
        return False
    for document in evidence_texts:
        for line in str(document).splitlines():
            if _normalized_compact(line) == target:
                return True
    return False


def _verify_declaration_structure(
    fact_data: dict[str, Any],
    items: list[dict],
) -> tuple[bool, str]:
    """Verify structural declaration evidence deterministically."""
    source_line = str(fact_data.get("source_line", "")).strip()
    declaration_label = str(fact_data.get("declaration_label", "")).strip()
    value = str(fact_data.get("value", "")).strip()
    evidence_indices = fact_data.get("evidence_indices", [])

    if not source_line or not declaration_label or not value:
        return False, "missing_source_line_label_or_value"

    evidence_texts: list[str] = []
    for index in evidence_indices:
        if not isinstance(index, int) or not 0 <= index < len(items):
            continue
        result = items[index].get("result", {})
        document = result.get("document", "") if isinstance(result, dict) else ""
        if document:
            evidence_texts.append(str(document))

    if not _find_source_line(source_line, evidence_texts):
        return False, "source_line_not_grounded"

    cleaned_line = _clean_markdown_text(source_line)
    cleaned_label = _clean_markdown_text(declaration_label)
    cleaned_value = _normalized_compact(value)

    # Structural only: do not infer whether the label semantically denotes the
    # requested attribute. That is intentionally deferred to the verifier.
    pattern = re.compile(
        rf"(?:^|\|)\s*{re.escape(cleaned_label)}\s*[:=]\s*(?P<rest>.+)$",
        re.IGNORECASE,
    )
    match = pattern.search(cleaned_line)
    if not match:
        return False, "label_not_found_as_field_prefix"

    remainder = _normalized_compact(match.group("rest"))
    if not remainder or cleaned_value not in remainder:
        return False, "value_not_in_declared_value_region"

    return True, "verified_structural_declaration"


def _build_provenance_verifier_prompt(
    group: str,
    candidates: list[dict[str, Any]],
) -> str:
    blocks: list[str] = []
    for candidate in candidates:
        blocks.append(
            f"FACT INDEX: {candidate['fact_index']}\n"
            f"Requested requirement: {group}\n"
            f"Attribute: {candidate['attribute']}\n"
            f"Value: {candidate['value']}\n"
            f"Declaration label: {candidate['declaration_label']}\n"
            f"Source line: {candidate['source_line']}\n"
            f"Statement: {candidate['statement']}"
        )

    evidence = "\n\n".join(blocks)
    return f"""
You are a provenance verification component for a general-purpose enterprise RAG system.

Do not answer the requirement and do not resolve any conflict.
For each supplied fact, decide only whether its explicit declaration label
represents the requested attribute or is merely contextual/scoping information.

Allowed verdicts:
- "attribute_label": the declaration label explicitly names the requested attribute.
- "contextual_label": the label scopes, dates, versions, phases, events,
  locations, sequence markers, or otherwise contextualizes the value instead of
  declaring the requested attribute.
- "unknown": the distinction cannot be established reliably.

Return exactly one verdict per FACT INDEX. Do not use outside knowledge,
do not infer a hidden schema, and when uncertain return "unknown".

REQUIREMENT:
{group}

FACTS TO VERIFY:
{evidence}

Return JSON only:
{{
  "verdicts": [
    {{"fact_index": 0, "verdict": "attribute_label"}}
  ]
}}
"""


def _run_provenance_verifier(
    group: str,
    candidates: list[dict[str, Any]],
) -> dict[int, str]:
    """Return semantic provenance verdicts; any failure becomes unknown."""
    if not candidates:
        return {}

    fallback = {
        "verdicts": [
            {"fact_index": candidate["fact_index"], "verdict": "unknown"}
            for candidate in candidates
        ]
    }

    raw = safe_completion_json(
        _build_provenance_verifier_prompt(group, candidates),
        max_tokens=384,
        max_retries=1,
        fallback=fallback,
    )

    try:
        parsed = ProvenanceVerificationResult.model_validate(raw)
    except Exception:
        return {candidate["fact_index"]: "unknown" for candidate in candidates}

    allowed_indices = {candidate["fact_index"] for candidate in candidates}
    result: dict[int, str] = {index: "unknown" for index in allowed_indices}
    for verdict in parsed.verdicts:
        if verdict.fact_index in allowed_indices:
            result[verdict.fact_index] = verdict.verdict
    return result


def _build_temporal_state_verifier_prompt(
    group: str,
    candidates: list[dict[str, Any]],
) -> str:
    blocks: list[str] = []
    for candidate in candidates:
        blocks.append(
            f"FACT INDEX: {candidate['fact_index']}\n"
            f"Requested requirement: {group}\n"
            f"Attribute: {candidate['attribute']}\n"
            f"Value: {candidate['value']}\n"
            f"Effective start: {candidate['effective_start']}\n"
            f"Effective end: {candidate['effective_end']}\n"
            f"Temporal scope: {candidate['temporal_scope']}\n"
            f"Source line: {candidate['source_line']}\n"
            f"Statement: {candidate['statement']}"
        )

    evidence = "\n\n".join(blocks)
    return f"""
You are a temporal-state classification component for a general-purpose enterprise RAG system.

Do not answer the requirement and do not choose the latest value.
For each supplied fact, decide only whether the dated statement establishes
a continuing value for the requested attribute starting at the stated date.

Allowed verdicts:
- "state_assignment": the statement establishes that the attribute became
  the supplied value at the stated time. Unless an explicit end/replacement
  is given, this state can remain effective until a later state change.
- "event_only": the statement records a one-time event, observation, or action
  without establishing that the supplied value becomes the continuing state
  of the requested attribute.
- "unknown": the distinction cannot be established reliably.

Important:
- A transition such as a value being promoted, changed, moved, assigned,
  switched, upgraded, downgraded, activated, deactivated, or otherwise set
  to a new value may establish a continuing state when the statement clearly
  identifies the requested attribute value.
- Do not infer a continuing state from a one-time event such as recognition,
  measurement, payment, completion, or observation unless the text explicitly
  changes the requested attribute.
- Do not use outside knowledge.
- Return "unknown" when uncertain.

REQUIREMENT:
{group}

FACTS TO CLASSIFY:
{evidence}

Return JSON only:
{{
  "verdicts": [
    {{"fact_index": 0, "verdict": "state_assignment"}}
  ]
}}
"""


def _run_temporal_state_verifier(
    group: str,
    candidates: list[dict[str, Any]],
) -> dict[int, str]:
    """Return semantic temporal-state verdicts; any failure becomes unknown."""
    if not candidates:
        return {}

    fallback = {
        "verdicts": [
            {"fact_index": candidate["fact_index"], "verdict": "unknown"}
            for candidate in candidates
        ]
    }

    raw = safe_completion_json(
        _build_temporal_state_verifier_prompt(group, candidates),
        max_tokens=512,
        max_retries=1,
        fallback=fallback,
    )

    try:
        parsed = TemporalStateVerificationResult.model_validate(raw)
    except Exception:
        return {candidate["fact_index"]: "unknown" for candidate in candidates}

    allowed_indices = {candidate["fact_index"] for candidate in candidates}
    result: dict[int, str] = {index: "unknown" for index in allowed_indices}
    for verdict in parsed.verdicts:
        if verdict.fact_index in allowed_indices:
            result[verdict.fact_index] = verdict.verdict
    return result


def _same_label_conflict_indices(facts: list[dict[str, Any]]) -> set[int]:
    """Protect conflicting same-label declarations from verifier demotion."""
    values_by_label: dict[str, set[str]] = {}
    indices_by_label: dict[str, list[int]] = {}

    for index, fact in enumerate(facts):
        if fact.get("value_type") != "declared_attribute":
            continue
        label = _normalized_compact(str(fact.get("declaration_label", "")))
        if not label:
            continue
        values_by_label.setdefault(label, set()).add(
            _normalized_value(str(fact.get("value", "")))
        )
        indices_by_label.setdefault(label, []).append(index)

    forced: set[int] = set()
    for label, values in values_by_label.items():
        if len(values) > 1:
            forced.update(indices_by_label[label])
    return forced


def _looks_like_temporal_event_line(source_line: str) -> bool:
    """Detect date-prefixed timeline/event syntax without using domain terms."""
    cleaned = _clean_markdown_text(source_line)
    if not cleaned:
        return False
    date_token = _DATE_TOKEN_RE
    return bool(
        re.match(
            rf"^\s*{date_token}\s*(?:-|–|—|\bto\b)\s*(?:{date_token}|present|current|ongoing)\s*:",
            cleaned,
            re.IGNORECASE,
        )
        or re.match(rf"^\s*{date_token}\s*:", cleaned, re.IGNORECASE)
    )


def _verify_provenance(
    group: str,
    facts: list[dict[str, Any]],
    items: list[dict],
    mode: str,
) -> list[dict[str, Any]]:
    """Ground declaration claims and optionally verify their semantic role."""
    if mode == "off":
        return facts

    working = [dict(fact) for fact in facts]

    for fact in working:
        qualifiers = dict(fact.get("qualifiers") or {})
        claimed_type = str(fact.get("value_type", "unknown"))

        # A date-prefixed timeline/event row is not a field declaration merely
        # because it uses a colon or table-like syntax. Keep the temporal fact
        # available to the resolver, but prevent it from masquerading as a
        # static/current declaration.
        if claimed_type == "declared_attribute" and _looks_like_temporal_event_line(
            str(fact.get("source_line", ""))
        ):
            fact["value_type"] = "narrative_reference"
            fact["declaration_label"] = ""
            claimed_type = "narrative_reference"

        qualifiers["claimed_value_type"] = claimed_type

        if claimed_type == "declared_attribute":
            grounded, reason = _verify_declaration_structure(fact, items)
            qualifiers["provenance_structural_verdict"] = "verified" if grounded else "failed"
            qualifiers["provenance_structural_reason"] = reason
            if not grounded and mode == "enforce":
                fact["value_type"] = "unknown"
        else:
            qualifiers["provenance_structural_verdict"] = "not_applicable"

        fact["qualifiers"] = qualifiers

    # Only semantic disagreements need an additional LLM call. This keeps
    # single-value groups on the existing path.
    resolvable = [
        AtomicFact.model_validate(fact)
        for fact in working
        if fact.get("attribute", "").strip() and fact.get("value", "").strip()
    ]
    value_keys = _distinct_value_keys(resolvable)

    declared_candidates = [
        {
            "fact_index": index,
            "attribute": str(fact.get("attribute", "")),
            "value": str(fact.get("value", "")),
            "declaration_label": str(fact.get("declaration_label", "")),
            "source_line": str(fact.get("source_line", "")),
            "statement": str(fact.get("statement", "")),
        }
        for index, fact in enumerate(working)
        if fact.get("value_type") == "declared_attribute"
        and fact.get("declaration_label")
    ]

    should_verify = len(value_keys) > 1 and bool(declared_candidates)
    forced_declared = _same_label_conflict_indices(working)

    verdicts: dict[int, str] = {}
    if should_verify:
        candidates_to_verify = [
            candidate
            for candidate in declared_candidates
            if candidate["fact_index"] not in forced_declared
        ]
        verdicts = _run_provenance_verifier(group, candidates_to_verify)

    # Current/latest questions also need to distinguish a dated state-changing
    # fact from a dated one-time event. Without this, a promotion/change entry
    # can remain marked historical even though it establishes the state that
    # persists until superseded. The verifier is gated to disagreement cases
    # with dated facts so ordinary supported groups do not pay for another call.
    temporal_intent = _query_temporal_intent(group)
    temporal_candidates: list[dict[str, Any]] = []
    if temporal_intent in {"current", "latest"} and len(value_keys) > 1:
        for index, fact in enumerate(working):
            if not fact.get("attribute", "").strip() or not fact.get("value", "").strip():
                continue
            if not str(fact.get("effective_start", "")).strip():
                continue
            if str(fact.get("effective_end", "")).strip() or bool(fact.get("effective_end_open")):
                continue
            temporal_candidates.append({
                "fact_index": index,
                "attribute": str(fact.get("attribute", "")),
                "value": str(fact.get("value", "")),
                "effective_start": str(fact.get("effective_start", "")),
                "effective_end": str(fact.get("effective_end", "")),
                "temporal_scope": str(fact.get("temporal_scope", "unknown")),
                "source_line": str(fact.get("source_line", "")),
                "statement": str(fact.get("statement", "")),
            })

    temporal_verdicts = _run_temporal_state_verifier(
        group, temporal_candidates
    ) if temporal_candidates else {}

    temporal_events: list[dict[str, Any]] = []
    for index, fact in enumerate(working):
        if index not in temporal_verdicts:
            continue
        qualifiers = dict(fact.get("qualifiers") or {})
        verifier_verdict = temporal_verdicts[index]
        qualifiers["temporal_verifier_verdict"] = verifier_verdict

        if (
            mode == "enforce"
            and verifier_verdict == "state_assignment"
            and str(fact.get("effective_end", "")).strip() == ""
            and not bool(fact.get("effective_end_open"))
        ):
            fact["effective_end_open"] = True
            qualifiers["temporal_verified_open_ended_state"] = True

        fact["qualifiers"] = qualifiers
        temporal_events.append({
            "fact_index": index,
            "value": fact.get("value", ""),
            "effective_start": fact.get("effective_start", ""),
            "verifier": verifier_verdict,
            "would_be_open_ended": verifier_verdict == "state_assignment",
        })

    if mode == "shadow" and temporal_events:
        logger.info(
            "Temporal-state shadow verification group=%r events=%s",
            group,
            temporal_events,
        )

    shadow_events: list[dict[str, Any]] = []
    for index, fact in enumerate(working):
        qualifiers = dict(fact.get("qualifiers") or {})
        if qualifiers.get("claimed_value_type") != "declared_attribute":
            continue

        structural = qualifiers.get("provenance_structural_verdict")

        if structural != "verified":
            would_be = "unknown"
            verifier_verdict = "not_run"
        elif index in forced_declared:
            would_be = "declared_attribute"
            verifier_verdict = "same_label_conflict_guard"
        elif should_verify:
            verifier_verdict = verdicts.get(index, "unknown")
            would_be = {
                "attribute_label": "declared_attribute",
                "contextual_label": "narrative_reference",
                "unknown": "unknown",
            }.get(verifier_verdict, "unknown")
        else:
            would_be = "declared_attribute"
            verifier_verdict = "not_needed"

        qualifiers["provenance_verifier_verdict"] = verifier_verdict
        qualifiers["provenance_verified_type"] = would_be
        fact["qualifiers"] = qualifiers

        event = {
            "fact_index": index,
            "value": fact.get("value", ""),
            "label": fact.get("declaration_label", ""),
            "claimed_type": "declared_attribute",
            "would_be_type": would_be,
            "structural": structural,
            "verifier": verifier_verdict,
        }
        shadow_events.append(event)

        if mode == "enforce":
            fact["value_type"] = would_be

    if mode == "shadow" and shadow_events:
        logger.info("Provenance shadow verification group=%r events=%s", group, shadow_events)

    return working




_CURRENT_QUERY_PATTERNS = (
    r"\bcurrent(?:ly)?\b",
    r"\bnow\b",
    r"\bat present\b",
    r"\bas of now\b",
    r"\bpresently\b",
)
_LATEST_QUERY_PATTERNS = (
    r"\blatest\b",
    r"\bmost recent\b",
)


def _query_temporal_intent(requirement: str) -> str:
    """Detect current/latest state intent without domain assumptions."""
    text = _normalized_compact(requirement)
    if any(re.search(pattern, text) for pattern in _CURRENT_QUERY_PATTERNS):
        return "current"
    if any(re.search(pattern, text) for pattern in _LATEST_QUERY_PATTERNS):
        return "latest"
    return "none"


def _temporal_key(value: str, *, end: bool = False) -> tuple[int, int, int] | None:
    """Normalize YYYY / YYYY-MM / YYYY-MM-DD values for ordering."""
    text = str(value).strip().casefold()
    if not text or text == "present":
        return None
    match = re.fullmatch(r"(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?", text)
    if not match:
        return None
    year = int(match.group(1))
    month = int(match.group(2) or (12 if end else 1))
    day = int(match.group(3) or (31 if end else 1))
    return year, month, day


def _fact_effective_start(fact: AtomicFact) -> tuple[int, int, int] | None:
    """Return the normalized start date of a fact's effective state."""
    return _temporal_key(fact.effective_start)


def _fact_effective_end(fact: AtomicFact) -> tuple[int, int, int] | None:
    """Return the normalized end date of a fact's effective state."""
    return _temporal_key(fact.effective_end, end=True)


def _fact_is_current_candidate(fact: AtomicFact, today: date | None = None) -> bool:
    """Return whether a fact can describe the current state.

    A dated state assignment is considered effective until an explicit end or
    a later fact supersedes it. Explicit current/present scope is also current.
    Future-starting facts are excluded.
    """
    today = today or date.today()
    today_key = (today.year, today.month, today.day)

    start_key = _fact_effective_start(fact)
    if start_key is not None and start_key > today_key:
        return False

    if fact.temporal_scope == "current":
        return True

    if fact.effective_end_open:
        return True

    if fact.effective_end.strip().casefold() == "present":
        return True

    end_key = _fact_effective_end(fact)
    if end_key is not None:
        return end_key >= today_key

    # A dated state assignment with no explicit end remains a candidate until
    # superseded by a later state assignment for the same attribute.
    return start_key is not None


def _latest_dated_facts(facts: list[AtomicFact]) -> list[AtomicFact]:
    """Return facts sharing the latest known effective start date."""
    dated = [fact for fact in facts if _fact_effective_start(fact) is not None]
    if not dated:
        return []
    latest_key = max(
        _fact_effective_start(fact)
        for fact in dated
        if _fact_effective_start(fact) is not None
    )
    return [fact for fact in dated if _fact_effective_start(fact) == latest_key]


def _resolve_current_state(
    facts: list[AtomicFact],
    *,
    latest_only: bool = False,
) -> tuple[ConsolidationStatus, ResolutionBasis | None, AtomicFact | None]:
    """Resolve the latest applicable state from temporal facts."""
    if latest_only:
        candidates = _latest_dated_facts(facts)
    else:
        candidates = [fact for fact in facts if _fact_is_current_candidate(fact)]
        dated_candidates = [fact for fact in candidates if _fact_effective_start(fact) is not None]
        if dated_candidates:
            candidates = _latest_dated_facts(dated_candidates)

    if not candidates:
        return "insufficient", None, None

    value_keys = _distinct_value_keys(candidates)
    overall_value_keys = _distinct_value_keys(facts)
    if len(value_keys) == 1:
        status = "resolved" if len(overall_value_keys) > 1 else "supported"
        return status, (
            "temporal_latest_state" if latest_only else "temporal_current_precedence"
        ), candidates[0]

    # If several facts share the same effective start and disagree, there is
    # no deterministic basis for selecting one.
    return "contradictory", None, None


def _resolve_requirement_facts(
    facts: list[AtomicFact],
    requirement: str = "",
) -> tuple[ConsolidationStatus, ResolutionBasis | None, AtomicFact | None]:
    """Resolve one requirement deterministically from already-extracted facts.

    For current/latest questions, explicit temporal applicability is resolved
    first. An undated/static declaration is only a fallback when no temporal
    state can be established. For non-temporal disagreements, a directly
    declared attribute may still take precedence over narrative references.

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

    temporal_intent = _query_temporal_intent(requirement)

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

    # A current-state question must distinguish an explicit *current temporal
    # state* from a merely dated event. An open-ended/present interval is
    # direct evidence of the state that applies now and therefore outranks a
    # static declaration. A dated event without an explicit ongoing/current
    # interval does not silently overwrite a declaration. This is based only
    # on temporal semantics, never on a domain-specific record format.
    if temporal_intent == "current":
        explicit_current_states = [
            fact
            for fact in resolvable_facts
            if fact.temporal_scope == "current"
            or fact.effective_end_open
            or fact.effective_end.strip().casefold() in {"present", "current", "ongoing"}
        ]

        if explicit_current_states:
            current_status, current_basis, current_fact = _resolve_current_state(
                explicit_current_states, latest_only=False
            )
            if current_status in {"supported", "resolved"}:
                return current_status, current_basis, current_fact
            if current_status == "contradictory":
                return "contradictory", None, None

        if declared:
            if len(declared_keys) == 1:
                resolved = declared[0]
                if len(_distinct_value_keys(resolvable_facts)) == 1:
                    return "supported", None, resolved
                return "resolved", "declared_attribute_precedence", resolved
            return "contradictory", None, None

    # If there is no explicit declaration/current temporal state, current-state
    # questions can still be answered from dated state evidence that is
    # applicable now.
    if temporal_intent == "current":
        temporal_status, temporal_basis, temporal_fact = _resolve_current_state(
            resolvable_facts, latest_only=False
        )
        if temporal_status in {"supported", "resolved"}:
            return temporal_status, temporal_basis, temporal_fact
        if temporal_status == "contradictory":
            return "contradictory", None, None

    # "Latest" explicitly asks for the most recent effective state, so dated
    # state evidence is authoritative when it can be established.
    if temporal_intent == "latest":
        temporal_status, temporal_basis, temporal_fact = _resolve_current_state(
            resolvable_facts, latest_only=True
        )
        if temporal_status in {"supported", "resolved"}:
            return temporal_status, temporal_basis, temporal_fact
        if temporal_status == "contradictory":
            return "contradictory", None, None

    # For non-temporal disagreements, ambiguous provenance must not silently
    # resolve the conflict.
    if any(f.value_type == "unknown" for f in resolvable_facts):
        return "contradictory", None, None

    # A single declared value takes precedence over narrative references when
    # there is no explicit temporal intent.
    if len(declared_keys) == 1 and narrative:
        return "resolved", "declared_attribute_precedence", declared[0]

    return "contradictory", None, None


def _build_ambiguity_resolution_prompt(
    requirement: str,
    target_entities: list[str],
    facts: list[AtomicFact],
) -> str:
    """Build a closed-set resolver prompt over already-extracted facts."""
    serialized = []
    for index, fact in enumerate(facts):
        serialized.append(
            {
                "index": index,
                "subject": fact.subject,
                "attribute": fact.attribute,
                "value": fact.value,
                "value_type": fact.value_type,
                "temporal_scope": fact.temporal_scope,
                "effective_start": fact.effective_start,
                "effective_end": fact.effective_end,
                "effective_end_open": fact.effective_end_open,
                "section_type": fact.section_type,
                "statement": fact.statement,
                "source_candidate_id": fact.source_candidate_id,
            }
        )

    return f"""
You are a closed-set ambiguity resolver for a general-purpose enterprise RAG system.

Your job is ONLY to determine whether one of the already-extracted candidate facts
resolves the requirement. You are NOT an answer generator and you MUST NOT invent,
rewrite, combine, calculate, or infer a new factual value.

REQUIREMENT:
{requirement}

TARGET ENTITY/ENTITIES:
{', '.join(target_entities) if target_entities else 'not explicitly established'}

CANDIDATE FACTS:
{json.dumps(serialized, indent=2, ensure_ascii=False)}

Rules:
1. You may select ONLY one of the supplied fact indices.
2. Never invent a value, subject, attribute, date, or qualification.
3. The selected fact must answer the requested attribute and belong to the target
   entity when a target entity is established.
4. For current/latest requirements, distinguish historical facts from facts that
   establish a continuing/current state.
5. A dated state-changing fact can establish a continuing state when its text and
   temporal metadata show that the value became effective and remained in force
   until a later change.
6. Historical values that ended before a later state are not current conflicts.
7. A declared attribute can resolve an ambiguity when it is the strongest explicit
   representation of the requested attribute and no stronger temporal fact conflicts.
8. If two or more facts remain genuinely incompatible for the requested temporal
   scope, return "ambiguous" rather than guessing.
9. If none of the supplied facts answers the requirement, return "insufficient".
10. Do not use outside knowledge.

Return ONLY this JSON shape:
{{
  "verdict": "resolved | ambiguous | insufficient",
  "selected_fact_index": integer_or_null,
  "reason": "one short explanation"
}}
"""


def _run_ambiguity_resolver(
    requirement: str,
    target_entities: list[str],
    facts: list[AtomicFact],
) -> AmbiguityResolution:
    """Resolve ambiguity using only the closed set of already-extracted facts."""
    if not facts:
        return AmbiguityResolution(
            verdict="insufficient",
            selected_fact_index=None,
            reason="No extracted facts are available for ambiguity resolution.",
        )

    raw = safe_completion_json(
        _build_ambiguity_resolution_prompt(requirement, target_entities, facts),
        max_tokens=512,
        max_retries=2,
        fallback={
            "verdict": "ambiguous",
            "selected_fact_index": None,
            "reason": "Ambiguity resolver failed.",
        },
    )

    try:
        resolution = AmbiguityResolution.model_validate(raw)
    except Exception:
        return AmbiguityResolution(
            verdict="ambiguous",
            selected_fact_index=None,
            reason="Ambiguity resolver returned an invalid structured result.",
        )

    if resolution.verdict != "resolved":
        return resolution

    index = resolution.selected_fact_index
    if index is None or not 0 <= index < len(facts):
        return AmbiguityResolution(
            verdict="ambiguous",
            selected_fact_index=None,
            reason="Ambiguity resolver selected an invalid fact index.",
        )

    selected = facts[index]
    target_keys = {_normalize_identity(value) for value in target_entities if value.strip()}
    if target_keys and _normalize_identity(selected.subject) not in target_keys:
        return AmbiguityResolution(
            verdict="ambiguous",
            selected_fact_index=None,
            reason="Selected fact does not match the requested target entity.",
        )

    if not selected.attribute.strip() or not selected.value.strip():
        return AmbiguityResolution(
            verdict="ambiguous",
            selected_fact_index=None,
            reason="Selected fact lacks a usable attribute or value.",
        )

    return resolution


def _should_run_ambiguity_resolver(
    status: ConsolidationStatus,
    facts: list[AtomicFact],
    requirement: str,
) -> bool:
    """Gate the semantic resolver to unresolved cases with usable candidates."""
    if status not in {"contradictory", "insufficient"}:
        return False
    if not facts:
        return False
    if not any(fact.attribute.strip() and fact.value.strip() for fact in facts):
        return False
    return True


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



def _has_temporal_signals(items: list[dict]) -> bool:
    """Return True when retrieved text contains generic temporal markers."""
    date_pattern = re.compile(
        rf"(?:{_DATE_TOKEN_RE})|\bpresent\b|\bcurrent\b|\bongoing\b",
        re.IGNORECASE,
    )
    for item in items:
        result = item.get("result", {})
        document = str(result.get("document", "")) if isinstance(result, dict) else ""
        if date_pattern.search(document):
            return True
    return False


def _build_compact_extraction_rescue_prompt(
    group: str,
    items: list[dict],
    max_chars_per_candidate: int,
) -> str:
    """Build a deliberately small recovery prompt for failed/weak extraction."""
    blocks: list[str] = []
    for index, item in enumerate(items):
        result = item.get("result", {})
        document = str(result.get("document", "")) if isinstance(result, dict) else ""
        if len(document) > max_chars_per_candidate:
            document = document[:max_chars_per_candidate]
        blocks.append(f"CANDIDATE {index}:\n{document}")

    evidence = "\n\n".join(blocks)
    return f"""
You are a recovery extractor for a general-purpose enterprise RAG system.

Requirement: {group}

Extract ONLY facts about the attribute requested by the requirement.
For a current/latest question, include both:
1. an explicitly declared current/static field if present, and
2. any dated event or timeline entry that changes the requested attribute,
   preserving its date and value.

Do not include unrelated facts. Do not resolve or choose between values.
Return at most 6 facts. Preserve the exact source line for every fact.

Each fact must contain:
- statement
- attribute
- value
- value_type: declared_attribute | narrative_reference | unknown
- temporal_scope: current | historical | unknown
- effective_start
- effective_end
- effective_end_open
- section_type
- source_line
- declaration_label
- evidence_indices
- qualifiers

For a declaration, source_line and declaration_label must be present.
For narrative/history facts, declaration_label must be empty.
If a fact cannot be classified reliably, use unknown.

Return JSON only:
{{"group": {group!r}, "facts": [...]}}

{evidence}
"""


def _needs_temporal_rescue(group: str, facts: list[dict], items: list[dict]) -> bool:
    """Decide whether a current/latest requirement may have incomplete extraction."""
    if _query_temporal_intent(group) not in {"current", "latest"}:
        return False
    if not _has_temporal_signals(items):
        return False
    # Rescue when extraction failed completely, or found only one fact while
    # retrieved material contains explicit temporal information. The latter
    # protects against a summary-only extraction that misses a state change.
    return not facts or len(facts) == 1


def _run_extraction_rescue(
    group: str,
    items: list[dict],
    max_chars_per_candidate: int,
) -> dict:
    """Run a compact recovery extraction; failures remain safely empty."""
    fallback = {"group": group, "facts": []}
    raw = safe_completion_json(
        _build_compact_extraction_rescue_prompt(
            group,
            items,
            max_chars_per_candidate,
        ),
        max_tokens=768,
        max_retries=1,
        fallback=fallback,
    )
    return _normalize_model_result(group, items, raw)


def consolidate_evidence(
    grouped_evidence: dict[str, list[dict]],
    *,
    max_candidates_per_group: int | None = None,
    max_chars_per_candidate: int = 3500,
    provenance_verifier_mode: str | None = None,
    target_entities_by_group: dict[str, list[str]] | None = None,
) -> dict[str, dict]:
    """Consolidate retrieval evidence into requirement-scoped atomic facts.

    The design is extractive-first: direct text already present in retrieval
    candidates is never discarded merely because the LLM fails to summarize it.
    The LLM is used to normalize/identify facts, not to decide whether the
    underlying retrieved text exists.
    """

    consolidated: dict[str, dict] = {}
    verifier_mode = (
        provenance_verifier_mode.strip().lower()
        if isinstance(provenance_verifier_mode, str)
        else _provenance_verifier_mode()
    )
    if verifier_mode not in _PROVENANCE_VERIFIER_MODES:
        verifier_mode = _DEFAULT_PROVENANCE_VERIFIER_MODE

    for group, items in grouped_evidence.items():
        explicit_targets = None
        if isinstance(target_entities_by_group, dict):
            explicit_targets = target_entities_by_group.get(group)

        scoped_items, rejected_items, target_entities, scope_status = _scope_candidates(
            group, list(items), explicit_targets
        )

        rank_limit = len(scoped_items) if max_candidates_per_group is None else max_candidates_per_group
        ranked_items = _rank_candidates(
            group,
            scoped_items,
            max_candidates=max(0, rank_limit),
        )

        extractive = _extractive_spans(group, ranked_items)
        structured_rescue = _structured_field_spans(group, ranked_items)

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
            max_tokens=1024,
            max_retries=2,
            fallback=fallback,
        )

        model_result = _normalize_model_result(group, ranked_items, raw)
        merged = _merge_extractive_fallback(
            model_result,
            extractive + structured_rescue,
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
                max_tokens=768,
                max_retries=1,
                fallback=fallback,
            )
            rechecked = _normalize_model_result(group, ranked_items, recheck_raw)
            merged = _merge_extractive_fallback(
                rechecked,
                extractive,
                ranked_items,
            )

        if _needs_temporal_rescue(
            group,
            list(merged.get("facts", [])),
            ranked_items,
        ):
            rescue_result = _run_extraction_rescue(
                group,
                ranked_items,
                max_chars_per_candidate=max_chars_per_candidate,
            )
            rescue_merged = _merge_extractive_fallback(
                rescue_result,
                extractive,
                ranked_items,
            )
            # Prefer the rescue only when it actually found additional facts.
            if len(rescue_merged.get("facts", [])) > len(merged.get("facts", [])):
                merged = rescue_merged

        scoped_fact_data = []
        target_keys = {_normalize_identity(value) for value in target_entities}
        for fact_data in list(merged.get("facts", [])):
            subject = _normalize_identity(str(fact_data.get("subject", "")))
            if subject:
                if subject in target_keys:
                    scoped_fact_data.append(fact_data)
                continue

            # Legacy/extractive fact without a subject is accepted only from an
            # exact source identity match. This is fail-closed for unknown sources.
            indices = fact_data.get("evidence_indices", [])
            trusted = any(
                isinstance(index, int)
                and 0 <= index < len(ranked_items)
                and _normalize_identity(_candidate_identity(ranked_items[index])) in target_keys
                for index in indices
            ) if isinstance(indices, list) else False
            if not target_entities or trusted:
                scoped_fact_data.append(fact_data)

        verified_fact_data = _verify_provenance(
            group,
            scoped_fact_data,
            ranked_items,
            verifier_mode,
        )

        fact_models = []
        for fact_data in verified_fact_data:
            try:
                fact_models.append(AtomicFact.model_validate(fact_data))
            except Exception:
                continue

        status, resolution_basis, resolved_fact = _resolve_requirement_facts(
            fact_models,
            requirement=group,
        )

        # Deterministic rules remain authoritative whenever they can resolve the
        # requirement. The LLM is a closed-set fallback only for genuine ambiguity
        # or conservative insufficiency with at least one usable extracted fact.
        if _should_run_ambiguity_resolver(status, fact_models, group):
            ambiguity = _run_ambiguity_resolver(
                group,
                target_entities,
                fact_models,
            )
            if ambiguity.verdict == "resolved" and ambiguity.selected_fact_index is not None:
                resolved_fact = fact_models[ambiguity.selected_fact_index]
                status = "resolved"
                resolution_basis = "llm_ambiguity_resolution"
                llm_qualifiers = dict(resolved_fact.qualifiers or {})
                llm_qualifiers["ambiguity_resolver"] = {
                    "verdict": ambiguity.verdict,
                    "selected_fact_index": ambiguity.selected_fact_index,
                    "reason": ambiguity.reason,
                }
                resolved_fact = resolved_fact.model_copy(update={"qualifiers": llm_qualifiers})
            else:
                logger.info(
                    "Ambiguity resolver did not resolve group=%r verdict=%s reason=%s",
                    group,
                    ambiguity.verdict,
                    ambiguity.reason,
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
            # Only identity-scoped evidence is exposed downstream.
            "evidence": merged.get("evidence", ranked_items),
            "raw_candidates": list(items),
            "rejected_candidates": rejected_items,
            "target_entities": target_entities,
            "entity_scope_status": scope_status,
        }

    return consolidated
