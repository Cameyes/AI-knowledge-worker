from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.rag.llm_client import safe_completion_json


ConsolidationStatus = Literal["supported", "contradictory", "insufficient"]


class AtomicFact(BaseModel):
    """A concise factual statement grounded in one or more candidates."""

    statement: str
    evidence_indices: list[int] = Field(default_factory=list)


class ConsolidationGroup(BaseModel):
    """Structured evidence for one independent factual requirement."""

    group: str
    status: ConsolidationStatus = "insufficient"
    facts: list[AtomicFact] = Field(default_factory=list)



def _build_group_prompt(
    group: str,
    items: list[dict],
    max_candidates: int,
    max_chars_per_candidate: int,
    recheck: bool = False,
) -> str:
    """Build a focused prompt for exactly one evidence requirement."""

    candidates = items[:max_candidates]
    blocks: list[str] = []

    if not candidates:
        blocks.append("No candidates were retrieved.")
    else:
        for index, item in enumerate(candidates):
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

    evidence = "\n\n".join(blocks)

    recheck_instruction = (
        "This is a semantic recheck. The first pass may have been too "
        "conservative. Re-inspect EVERY candidate carefully and extract "
        "any directly stated fact that satisfies the requirement. "
        "Do not return insufficient merely because the fact appears in "
        "a different section of a candidate.\n\n"
        if recheck
        else ""
    )

    return f"""
You are an evidence extraction system for a general-purpose
enterprise RAG application.

You must consolidate evidence for ONE independent factual requirement.
Do not answer the overall user question.

{recheck_instruction}

REQUIREMENT:
{group}

RULES:
1. Use ONLY the candidates shown below.
2. Carefully inspect EVERY candidate before deciding whether the
   requirement is supported.
3. A requirement is supported when a candidate directly states the
   requested fact, even if the fact appears in a section such as a
   summary, history, table, timeline, metadata block, or another
   structured section of the document.
4. Extract the smallest useful atomic fact: subject, requested
   attribute, and necessary date/time/qualifiers/value.
5. Preserve exact names, dates, values, units, statuses, titles,
   versions, identifiers, and other relevant qualifiers.
6. Do NOT use outside knowledge.
7. Do NOT infer a missing value merely because another value is nearby.
8. Do NOT compare, rank, calculate, aggregate, filter, or answer the
   overall question.
9. If multiple candidates directly support the same fact, include all
   supporting candidate indices.
10. If candidates directly disagree about the same factual attribute,
    preserve BOTH facts and set status to "contradictory".
11. If the candidates do not directly establish the requirement,
    return no facts and status "insufficient".
12. Candidate indices are local to this requirement and start at 0.
13. Return ONLY JSON matching the schema below.

Schema:
{{
  "group": "{group}",
  "status": "supported",
  "facts": [
    {{
      "statement": "atomic factual statement",
      "evidence_indices": [0]
    }}
  ]
}}

Candidates:
{evidence}
"""


def _normalize_group_result(group: str, items: list[dict], raw: object, max_candidates: int) -> dict:
    """Validate model output and map valid evidence indices back to candidates."""

    try:
        parsed = ConsolidationGroup.model_validate(raw)
    except Exception:
        return {"status": "insufficient", "facts": [], "evidence": []}

    candidate_limit = min(len(items), max_candidates)
    valid_facts: list[dict] = []
    selected_indices: set[int] = set()

    for fact in parsed.facts:
        indices = sorted({
            index
            for index in fact.evidence_indices
            if 0 <= index < candidate_limit
        })
        statement = fact.statement.strip()
        if not statement or not indices:
            continue
        valid_facts.append({
            "statement": statement,
            "evidence_indices": indices,
        })
        selected_indices.update(indices)

    if not valid_facts:
        return {"status": "insufficient", "facts": [], "evidence": []}

    if parsed.status == "contradictory" and len(valid_facts) >= 2:
        status = "contradictory"
    else:
        status = "supported"

    return {
        "status": status,
        "facts": valid_facts,
        "evidence": [items[index] for index in sorted(selected_indices)],
    }


def consolidate_evidence(
    grouped_evidence: dict[str, list[dict]],
    *,
    max_candidates_per_group: int = 3,
    max_chars_per_candidate: int = 3500,
) -> dict[str, dict]:
    """
    Convert retrieved chunks into compact atomic evidence.

    IMPORTANT: each requirement is consolidated in its own LLM call.
    This prevents evidence from one requirement from competing for
    attention with evidence from another requirement.
    """

    consolidated: dict[str, dict] = {}

    for group, items in grouped_evidence.items():
        prompt = _build_group_prompt(
            group,
            items,
            max_candidates=max_candidates_per_group,
            max_chars_per_candidate=max_chars_per_candidate,
        )

        fallback = {
            "group": group,
            "status": "insufficient",
            "facts": [],
        }

        raw = safe_completion_json(
            prompt,
            max_tokens=512,
            max_retries=3,
            fallback=fallback,
        )

        normalized = _normalize_group_result(
            group,
            items,
            raw,
            max_candidates_per_group,
        )

        # Semantic retry: a model can conservatively miss a directly
        # stated fact even when the correct candidate is present.
        if normalized["status"] == "insufficient" and items:
            recheck_prompt = _build_group_prompt(
                group,
                items,
                max_candidates=max_candidates_per_group,
                max_chars_per_candidate=max_chars_per_candidate,
                recheck=True,
            )

            recheck_raw = safe_completion_json(
                recheck_prompt,
                max_tokens=512,
                max_retries=2,
                fallback=fallback,
            )

            rechecked = _normalize_group_result(
                group,
                items,
                recheck_raw,
                max_candidates_per_group,
            )

            if rechecked["status"] != "insufficient":
                normalized = rechecked
            else:
                # Never discard retrieved candidates merely because
                # consolidation was unable to extract a fact. They are
                # retained for the downstream fallback inspection path.
                normalized["evidence"] = items[:max_candidates_per_group]

        consolidated[group] = normalized

    return consolidated
