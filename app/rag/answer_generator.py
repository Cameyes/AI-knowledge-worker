from __future__ import annotations

from app.rag.llm_client import safe_completion_text


def generate_answer(
    query: str,
    evidence: dict[str, dict],
) -> dict:
    """
    Generate an answer from a structured, domain-agnostic evidence layer.

    Each requirement contains atomic facts produced by evidence
    consolidation plus the retrieved candidates that support those facts.
    """

    if not evidence:
        return {
            "answer": (
                "I could not find sufficient evidence "
                "to answer the question."
            ),
            "sources": [],
        }

    evidence_context: list[str] = []
    sources: list[str] = []

    for requirement, group_data in evidence.items():
        status = group_data.get("status", "insufficient")
        facts = group_data.get("facts", [])
        supporting_evidence = group_data.get("evidence", [])

        evidence_context.append(
            f"REQUIREMENT:\n{requirement}\n"
            f"STATUS: {status.upper()}"
        )

        if facts:
            evidence_context.append("ATOMIC FACTS:")

            for fact in facts:
                evidence_context.append(
                    f"- {fact['statement']} "
                    f"[evidence indices: {fact['evidence_indices']}]"
                )

        if status == "contradictory":
            evidence_context.append(
                "CONFLICT: multiple retrieved facts disagree for "
                "this requirement. Do not silently resolve the conflict."
            )

        # Include only selected supporting evidence, and keep it bounded.
        # This is for traceability; the atomic facts are the primary input
        # to the final reasoning step.
        for index, item in enumerate(supporting_evidence):
            result = item.get("result", {})
            metadata = result.get("metadata", {})
            source = metadata.get("source", "Unknown source")
            document = str(result.get("document", ""))

            evidence_context.append(
                f"SUPPORTING EVIDENCE {index}: {source}\n"
                f"{document[:1800]}"
            )

            if source not in sources:
                sources.append(source)

    context = "\n\n".join(evidence_context)

    prompt = f"""
You are an enterprise knowledge assistant.

Answer the user's question using ONLY the structured evidence below.

The evidence has already been consolidated into atomic facts. Treat
an atomic fact as usable only for the requirement where it appears.
The supporting evidence is included for traceability and conflict
inspection.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. Do not silently replace, merge, or reinterpret an atomic fact.
4. Every factual claim in the final answer must be supported by the
   atomic facts or directly by their supporting evidence.
5. Each REQUIREMENT is independent. Do not use one requirement to
   satisfy another.
6. For comparison, ranking, filtering, aggregation, or calculation,
   use the atomic facts across the relevant requirements only after
   all required facts are supported.
7. You MAY perform the arithmetic/comparison explicitly requested by
   the user when all required underlying facts are present.
8. If any required requirement is INSUFFICIENT, first inspect its
   retained supporting candidates. If a candidate directly establishes
   the requirement, you may use that directly supported fact; otherwise
   say that the requested conclusion is not fully established from the evidence.
9. If any required requirement is CONTRADICTORY, explicitly state the
   conflicting facts and do not choose one arbitrarily.
10. Do not invent a missing value for an entity, date, attribute, or
    other requirement merely to complete a comparison.
11. Do not generate citations or source references.
12. Be concise and answer only what was asked.

User question:
{query}

Structured evidence:
{context}

Now provide the final answer.
"""

    answer = safe_completion_text(
        prompt,
        max_tokens=384,
        fallback=(
            "I could not generate an answer "
            "due to a temporary service issue."
        ),
    )

    return {
        "answer": answer,
        "sources": sources,
    }
