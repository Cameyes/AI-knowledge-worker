from app.rag.llm_client import safe_completion_text


def _is_consolidated_group(value: object) -> bool:
    """Return True when a value matches the Phase 2 consolidation contract."""
    return isinstance(value, dict) and any(
        key in value
        for key in (
            "status",
            "resolved_fact",
            "resolution_basis",
            "raw_candidates",
        )
    )


def _format_fact(fact: dict) -> str:
    """Format one normalized fact without inventing information."""
    subject = fact.get("subject", "")
    attribute = fact.get("attribute", "")
    value = fact.get("value", "")
    statement = fact.get("statement", "")
    value_type = fact.get("value_type", "unknown")
    temporal_scope = fact.get("temporal_scope", "unknown")
    section_type = fact.get("section_type", "unknown")

    return (
        f"Subject: {subject}\n"
        f"Attribute: {attribute}\n"
        f"Value: {value}\n"
        f"Statement: {statement}\n"
        f"Value type: {value_type}\n"
        f"Temporal scope: {temporal_scope}\n"
        f"Section type: {section_type}"
    )


def _format_consolidated_group(
    group_key: str,
    result: dict,
) -> tuple[str, list[str]]:
    """Build generation context from a Phase 2 consolidated requirement."""
    lines = [f"GROUP: {group_key}"]
    sources: list[str] = []

    status = result.get("status", "insufficient")
    resolution_basis = result.get("resolution_basis")
    target_entities = result.get("target_entities") or []

    lines.append(f"STATUS: {status}")
    if target_entities:
        lines.append("VERIFIED TARGET ENTITIES: " + ", ".join(str(x) for x in target_entities))
        lines.append("ENTITY SCOPE: Only evidence already admitted for these targets is usable.")

    if resolution_basis:
        lines.append(f"RESOLUTION BASIS: {resolution_basis}")

    resolved_fact = result.get("resolved_fact")

    if isinstance(resolved_fact, dict):
        lines.append("RESOLVED FACT (PRIMARY ANSWER VALUE):")
        lines.append(_format_fact(resolved_fact))

    facts = result.get("facts", [])

    # Once consolidation has resolved a requirement, the resolved fact is the
    # application-level source of truth. Do not expose competing normalized
    # alternatives to the generator because that invites the model to
    # re-resolve a conflict that Python has already resolved.
    if status != "resolved" and facts:
        lines.append("ALL NORMALIZED FACTS:")
        for index, fact in enumerate(facts):
            if not isinstance(fact, dict):
                continue
            lines.append(f"FACT {index}:")
            lines.append(_format_fact(fact))

    evidence = result.get("evidence", result.get("raw_candidates", []))

    # Raw evidence is useful for unresolved/supported requirements, but a
    # resolved requirement already has an application-selected truth. Exposing
    # competing raw evidence here would invite the generator to re-resolve the
    # conflict and potentially replace the authoritative value.
    expose_raw_evidence = status not in {"insufficient", "resolved"}

    if evidence and expose_raw_evidence:
        lines.append("RAW SUPPORTING EVIDENCE:")

    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            continue

        nested = item.get("result", item)
        if not isinstance(nested, dict):
            continue

        metadata = nested.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        source = metadata.get("source") or metadata.get("filename") or "Unknown source"

        if source not in sources:
            sources.append(source)

        if expose_raw_evidence:
            document = nested.get("document", "")
            lines.append(
                f"Candidate {index} | Source: {source}\n"
                f"Evidence:\n{document}"
            )

    return "\n".join(lines), sources


def _format_raw_group(
    group_key: str,
    items: list[dict],
) -> tuple[str, list[str]]:
    """Backward-compatible formatting for the pre-Phase-2 evidence shape."""
    lines = [f"GROUP: {group_key}"]
    sources: list[str] = []

    for item in items:
        subquery = item.get("subquery", "")
        result = item.get("result", {})
        if not isinstance(result, dict):
            continue

        metadata = result.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        source = metadata.get("source", "Unknown source")
        document = result.get("document", "")

        lines.append(
            f"Subquery:\n{subquery}\n\n"
            f"Source:\n{source}\n\n"
            f"Evidence:\n{document}"
        )

        if source not in sources:
            sources.append(source)

    return "\n".join(lines), sources



def _normalized_text(value: object) -> str:
    """Normalize text for conservative answer-consistency checks."""
    import re

    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _resolved_facts(evidence: dict) -> list[dict]:
    """Return authoritative resolved facts from consolidated groups."""
    if not isinstance(evidence, dict):
        return []

    facts: list[dict] = []
    for group_value in evidence.values():
        if not isinstance(group_value, dict):
            continue
        if group_value.get("status") != "resolved":
            continue
        fact = group_value.get("resolved_fact")
        if isinstance(fact, dict) and str(fact.get("value", "")).strip():
            facts.append(fact)
    return facts


def _answer_preserves_resolved_facts(
    answer: str,
    resolved_facts: list[dict],
) -> bool:
    """
    Check a single resolved fact against a generated direct-answer response.

    Validation is intentionally limited to the one-resolved-fact case. Complex
    multi-group questions may legitimately summarize, compare, aggregate, or
    rank several resolved facts without spelling out every value verbatim.
    """
    if not resolved_facts:
        return True

    if len(resolved_facts) != 1:
        return True

    value = _normalized_text(resolved_facts[0].get("value", ""))
    if not value:
        return True

    return value in _normalized_text(answer)


def _deterministic_resolved_answer(resolved_facts: list[dict]) -> str:
    """Produce a grounded fallback using only application-resolved facts."""
    statements = []
    for fact in resolved_facts:
        statement = str(fact.get("statement", "")).strip()
        if statement:
            statements.append(statement)

    if not statements:
        return "I could not generate an answer due to a temporary service issue."

    if len(statements) == 1:
        return statements[0]

    return " ".join(statements)


def generate_answer(
    query: str,
    evidence: dict,
) -> dict:
    """
    Generate an answer from retrieved/consolidated evidence.

    Phase 3 behavior:
      - The evidence consolidator owns factual resolution.
      - The generator consumes `status`, `resolved_fact`, and
        `resolution_basis`; it does not re-resolve conflicting facts.
      - Raw evidence remains available for grounding.

    The function also accepts the pre-Phase-2 raw evidence shape for
    compatibility with callers that have not yet migrated.
    """

    if not evidence:
        return {
            "answer": "I could not find sufficient evidence to answer the question.",
            "sources": [],
        }

    evidence_context: list[str] = []
    sources: list[str] = []
    consolidated_mode = all(
        _is_consolidated_group(value)
        for value in evidence.values()
    )

    for group_key, group_value in evidence.items():
        if consolidated_mode and isinstance(group_value, dict):
            formatted, group_sources = _format_consolidated_group(
                group_key,
                group_value,
            )
        else:
            formatted, group_sources = _format_raw_group(
                group_key,
                group_value if isinstance(group_value, list) else [],
            )

        evidence_context.append(formatted)

        for source in group_sources:
            if source not in sources:
                sources.append(source)

    context = "\n\n".join(evidence_context)

    if consolidated_mode:
        resolution_instructions = """
IMPORTANT EVIDENCE-RESOLUTION CONTRACT:
The evidence consolidation layer has already performed deterministic fact
resolution. Treat STATUS, RESOLVED FACT, and RESOLUTION BASIS as application
outputs, not suggestions to reconsider.

- If STATUS is `resolved`, use the RESOLVED FACT as the primary answer value.
- If STATUS is `supported`, answer from the supported fact(s).
- If STATUS is `contradictory`, explicitly report the conflicting facts and do
  not choose one yourself.
- If STATUS is `insufficient`, state that the available evidence is
  insufficient; do not invent or infer a value.
- Do not independently re-resolve a `resolved` requirement using raw evidence.
- Raw supporting evidence is provided for grounding and context only. It has
  already passed entity scoping; never re-introduce or infer facts from rejected
  or omitted candidates. For a `resolved` group, raw evidence MUST NOT replace
  or override the RESOLVED FACT.
"""
    else:
        resolution_instructions = """
This evidence has not necessarily been passed through deterministic
consolidation. Do not invent facts, and do not silently resolve genuine
conflicts in raw evidence.
"""

    prompt = f"""
You are an enterprise knowledge assistant.

Answer the user's question using ONLY the provided evidence.
Evidence is grouped by independently retrievable requirement.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. Every factual claim must be supported by the provided evidence.
4. Carefully synthesize evidence from different groups only when the user's
   question requires combining independently supported facts.
5. Keep different requirement groups distinct; do not use one group's fact to
   satisfy another group's requirement.
6. Be concise and directly answer the user's question.
7. Do not generate citations or source references.

{resolution_instructions}

User question:
{query}

Retrieved and consolidated evidence:
{context}

Now provide the final answer.
"""

    answer = safe_completion_text(
        prompt,
        max_tokens=256,
        fallback="I could not generate an answer due to a temporary service issue.",
    )

    authoritative_facts = _resolved_facts(evidence) if consolidated_mode else []

    if (
        len(authoritative_facts) == 1
        and not _answer_preserves_resolved_facts(answer, authoritative_facts)
    ):
        # One corrective pass is allowed, using ONLY authoritative resolved
        # facts. This keeps the LLM responsible for wording while preventing it
        # from replacing application-resolved values with stale alternatives.
        authoritative_context = "\n\n".join(
            _format_fact(fact)
            for fact in authoritative_facts
        )

        correction_prompt = f"""
You are an enterprise knowledge assistant correcting a generated answer.

Answer the user's question using ONLY the authoritative application-resolved
facts below. Do not re-resolve, reinterpret, combine, replace, or contradict
these values. Do not use any discarded or alternative value from earlier
reasoning. Every factual claim must be supported by these authoritative facts.

Authoritative resolved facts:
{authoritative_context}

User question:
{query}

Previous generated answer:
{answer}

Return a concise corrected answer whose factual values exactly preserve the
authoritative resolved facts. Do not add unsupported details or mention this
correction process.
"""

        corrected = safe_completion_text(
            correction_prompt,
            max_tokens=256,
            max_retries=2,
            fallback="",
        )

        if corrected.strip() and _answer_preserves_resolved_facts(
            corrected,
            authoritative_facts,
        ):
            answer = corrected.strip()
        else:
            answer = _deterministic_resolved_answer(authoritative_facts)

    return {
        "answer": answer,
        "sources": sources,
    }
