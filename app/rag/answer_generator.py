from app.rag.llm_client import safe_completion_text


def generate_answer(
    query: str,
    evidence: dict[str, list[dict]],
) -> dict:
    """
    Generate an answer using retrieved evidence.

    `evidence` is a dict mapping an arbitrary grouping key
    (e.g. entity, document, employee — whatever the retrieval
    layer grouped by) to a list of items shaped like:
        {"subquery": str, "result": {"document": str, "metadata": {...}, "distance": ...}}
    """

    if not evidence:
        return {
            "answer": "I could not find sufficient evidence to answer the question.",
            "sources": [],
        }

    evidence_context = []
    sources = []

    for group_key, items in evidence.items():
        evidence_context.append(f"GROUP: {group_key}")

        for item in items:
            subquery = item["subquery"]
            result = item["result"]

            source = result["metadata"].get(
                "source",
                "Unknown source",
            )

            document = result["document"]

            evidence_context.append(
                f"""
Subquery:
{subquery}

Source:
{source}

Evidence:
{document}
"""
            )

            if source not in sources:
                sources.append(source)

    context = "\n".join(evidence_context)

    prompt = f"""
You are an enterprise knowledge assistant.

Answer the user's question using ONLY the provided evidence.
Evidence is grouped — pay attention to which group each piece
of evidence belongs to when the question involves comparing or
combining information across groups.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. Every factual claim must be supported by the provided evidence.
4. Carefully reason across evidence from different subqueries when necessary.
5. For multi-condition questions, make sure all conditions are satisfied before giving a final answer.
6. If the evidence is insufficient or contradictory, clearly say so.
7. Do not generate citations or source references.
8. Give a concise answer.

User question:
{query}

Retrieved evidence:
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