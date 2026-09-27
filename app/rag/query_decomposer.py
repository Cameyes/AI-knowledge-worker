
# def decompose_query(query: str):
#     return [
#         "Which employee had a 2023 performance rating of 4.7/5?",
#         "Who received recognition for customer-related performance?"
#     ]


from app.rag.llm_client import safe_completion_json


def decompose_query_structured(query: str) -> list[dict]:
    """
    Decompose a query into independent, domain-agnostic evidence
    requirements while preserving explicitly requested entities.

    Each item has:
        - requirement: the complete retrievable factual requirement
        - entities: explicitly identified entities/concepts needed
          for that requirement
    """

    prompt = f"""
You are a query decomposition system for a general-purpose
enterprise RAG application.

Decompose the user's question into the smallest set of independent
factual evidence requirements needed to answer it.

The decomposition must be completely domain-agnostic. An entity may
be a person, product, project, company, location, document, metric,
organization, object, or anything else explicitly identified by the user.

RULES:
1. Create one requirement for each independently retrievable fact.
2. If multiple explicitly identified entities each require the same
   type of fact, create one requirement per entity.
3. Preserve the complete information requirement in every requirement,
   including dates, metrics, attributes, categories, locations,
   statuses, and other constraints.
4. Do not perform calculations, comparisons, ranking, filtering,
   aggregation, or answer the question.
5. Do not invent entities or facts.
6. Each requirement must stand alone without the original query.
7. `entities` must contain only entities explicitly identified in the
   requirement/query that are useful for targeted retrieval.
8. If no explicit entity is present, return an empty entities list.
9. Preserve the user's terminology whenever possible.

EXAMPLE:
User: "What are the release dates of Product A, Product B, and Product C?"
Output:
{{
  "requirements": [
    {{"requirement": "What is the release date of Product A?", "entities": ["Product A"]}},
    {{"requirement": "What is the release date of Product B?", "entities": ["Product B"]}},
    {{"requirement": "What is the release date of Product C?", "entities": ["Product C"]}}
  ]
}}

Return ONLY valid JSON in exactly this structure:
{{
  "requirements": [
    {{"requirement": "...", "entities": ["..."]}}
  ]
}}

User query:
{query}
"""

    data = safe_completion_json(
        prompt,
        max_tokens=768,
        fallback={"requirements": [{"requirement": query, "entities": []}]},
    )

    requirements = data.get("requirements", [])

    if not isinstance(requirements, list) or not requirements:
        return [{"requirement": query, "entities": []}]

    cleaned = []

    for item in requirements:
        if not isinstance(item, dict):
            continue

        requirement = item.get("requirement")
        entities = item.get("entities", [])

        if not isinstance(requirement, str) or not requirement.strip():
            continue

        if not isinstance(entities, list):
            entities = []

        cleaned_entities = []
        for entity in entities:
            if isinstance(entity, str) and entity.strip():
                value = entity.strip()
                if value not in cleaned_entities:
                    cleaned_entities.append(value)

        cleaned.append({
            "requirement": requirement.strip(),
            "entities": cleaned_entities,
        })

    return cleaned or [{"requirement": query, "entities": []}]


def decompose_query(query: str) -> list[str]:
    """Backward-compatible string-only decomposition API."""
    return [item["requirement"] for item in decompose_query_structured(query)]
