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
    attribute = fact.get("attribute", "")
    value = fact.get("value", "")
    statement = fact.get("statement", "")
    value_type = fact.get("value_type", "unknown")
    temporal_scope = fact.get("temporal_scope", "unknown")
    section_type = fact.get("section_type", "unknown")

    return (
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

    lines.append(f"STATUS: {status}")

    if resolution_basis:
        lines.append(f"RESOLUTION BASIS: {resolution_basis}")

    resolved_fact = result.get("resolved_fact")

    if isinstance(resolved_fact, dict):
        lines.append("RESOLVED FACT (PRIMARY ANSWER VALUE):")
        lines.append(_format_fact(resolved_fact))

    facts = result.get("facts", [])

    if facts:
        lines.append("ALL NORMALIZED FACTS:")
        for index, fact in enumerate(facts):
            if not isinstance(fact, dict):
                continue
            lines.append(f"FACT {index}:")
            lines.append(_format_fact(fact))

    evidence = result.get("evidence", result.get("raw_candidates", []))

    # An insufficient consolidated requirement has no verified facts. Do not
    # expose unvalidated raw candidate text to the generator, because doing so
    # can cause the LLM to invent or merge identities from unrelated retrieval
    # candidates. Sources are still collected for the structured return value.
    expose_raw_evidence = status != "insufficient"

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
- Raw supporting evidence is provided for grounding and context only.
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

    return {
        "answer": answer,
        "sources": sources,
    }
