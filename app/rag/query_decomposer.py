
# def decompose_query(query: str):
#     return [
#         "Which employee had a 2023 performance rating of 4.7/5?",
#         "Who received recognition for customer-related performance?"
#     ]


from app.rag.llm_client import safe_completion_json


def decompose_query(query: str) -> list[str]:
    """
    Decompose a user query into the smallest set of independent
    factual evidence requirements needed to answer it.

    The decomposition is domain-agnostic. It does not assume
    anything about the type of entities, documents, or data
    being queried.
    """

    prompt = f"""
You are a query decomposition system for a general-purpose
enterprise RAG application.

Your task is to decompose the user's question into the smallest
set of independent factual evidence requirements needed to
answer it.

The requirements will be used independently for document
retrieval and evidence verification.

IMPORTANT PRINCIPLE:

Each subquery should represent ONE independently retrievable
and verifiable piece of evidence.

The decomposition must be completely domain-agnostic.
Do not assume the query is about employees, companies,
finance, products, projects, contracts, or any particular
domain.

GENERAL RULES:

1. Identify every factual piece of evidence required to answer
   the user's question.

2. If multiple explicitly identified entities are each being
   asked for the same type of information, create one subquery
   per entity.

3. Preserve the complete information requirement in every
   subquery, including:
   - entity
   - date or time period
   - metric or attribute
   - category
   - location
   - status
   - other constraints specified by the user

4. If a question requires comparing, ranking, filtering,
   aggregating, or calculating across multiple entities,
   decompose it into the independent factual requirements
   needed to perform that operation.

5. Do NOT perform calculations, comparisons, ranking,
   aggregation, or filtering yourself.

6. Do NOT answer the user's question.

7. Do NOT invent facts or requirements that are not implied
   by the user's question.

8. Do NOT create unnecessary subqueries.

9. If one piece of evidence is sufficient to answer the
   question, return exactly one subquery.

10. If multiple pieces of evidence are independently required,
    return one subquery for each piece.

11. When multiple entities are explicitly named and each entity
    requires independent evidence, do not keep those entities
    together in one subquery.

12. When a question asks for a comparison or calculation,
    retrieve the underlying facts separately rather than
    attempting to encode the comparison itself into every
    retrieval query.

13. Each subquery must be understandable on its own without
    relying on the original user query.

14. Preserve the user's terminology whenever possible.

15. Do not rewrite the request into a broader or more general
    question.

DOMAIN-NEUTRAL EXAMPLE 1:

User query:
"What are the release dates of Product A, Product B,
and Product C?"

Correct decomposition:

[
    "What is the release date of Product A?",
    "What is the release date of Product B?",
    "What is the release date of Product C?"
]

DOMAIN-NEUTRAL EXAMPLE 2:

User query:
"Which of Product A, Product B, and Product C was released
first?"

Correct decomposition:

[
    "What is the release date of Product A?",
    "What is the release date of Product B?",
    "What is the release date of Product C?"
]

The decomposition retrieves the facts required for the
downstream comparison. Do not perform the comparison.

DOMAIN-NEUTRAL EXAMPLE 3:

User query:
"How did Metric X change between 2022 and 2023?"

Correct decomposition:

[
    "What was Metric X in 2022?",
    "What was Metric X in 2023?"
]

DOMAIN-NEUTRAL EXAMPLE 4:

User query:
"Which locations had a value above 100 in 2023?"

Correct decomposition:

[
    "What was the value for each relevant location in 2023?"
]

Do not attempt to determine which locations satisfy the
condition. Retrieve the underlying evidence needed for the
downstream filtering operation.

DOMAIN-NEUTRAL EXAMPLE 5:

User query:
"Who is responsible for Project A and what is its current
status?"

Correct decomposition:

[
    "Who is responsible for Project A?",
    "What is the current status of Project A?"
]

DOMAIN-NEUTRAL EXAMPLE 6:

User query:
"Summarize Project A."

Correct decomposition:

[
    "What information is available about Project A?"
]

Do not unnecessarily split a simple request into many
subqueries.

OUTPUT REQUIREMENTS:

Return ONLY valid JSON.

The JSON must have exactly this structure:

{{
    "subqueries": [
        "subquery 1",
        "subquery 2"
    ]
}}

User query:

{query}
"""

    data = safe_completion_json(
        prompt,
        max_tokens=512,
        fallback={"subqueries": [query]},
    )

    subqueries = data.get("subqueries", [])

    if not isinstance(subqueries, list) or not subqueries:
        subqueries = [query]

    cleaned_subqueries = []

    for subquery in subqueries:
        if not isinstance(subquery, str):
            continue

        subquery = subquery.strip()

        if subquery:
            cleaned_subqueries.append(subquery)

    return cleaned_subqueries or [query]