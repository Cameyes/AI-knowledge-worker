
# def decompose_query(query: str):
#     return [
#         "Which employee had a 2023 performance rating of 4.7/5?",
#         "Who received recognition for customer-related performance?"
#     ]


import re

from app.rag.llm_client import safe_completion_json


def _normalize_entity(text: str) -> str:
    """Normalize an entity for duplicate/explicitness checks."""
    return re.sub(
        r"\s+",
        " ",
        re.sub(r"[^a-z0-9]+", " ", str(text).lower()),
    ).strip()


def _entity_is_explicit(entity: str, requirement: str, original_query: str) -> bool:
    """Accept an entity only when its complete phrase occurs in user input."""
    entity_norm = _normalize_entity(entity)
    if not entity_norm:
        return False

    for text in (requirement, original_query):
        text_norm = _normalize_entity(text)
        if re.search(
            rf"(?<![a-z0-9]){re.escape(entity_norm)}(?![a-z0-9])",
            text_norm,
        ):
            return True

    return False


def _validate_entity_roles(
    query: str,
    requirements: list[dict],
) -> list[dict]:
    """Semantically validate model-emitted entities without domain-specific rules.

    The decomposer may return phrases that literally occur in the query but are
    actually attributes, values, dates, actions, or other constraints. This
    closed-set validation keeps only phrases the model identifies as true
    retrieval targets for their requirement.
    """
    validated: list[dict] = []

    for item in requirements:
        requirement = item.get("requirement", "")
        entities = item.get("entities", [])

        if not requirement or not isinstance(entities, list) or not entities:
            validated.append(item)
            continue

        candidates = [
            value.strip()
            for value in entities
            if isinstance(value, str) and value.strip()
        ]
        if not candidates:
            validated.append({
                "requirement": requirement,
                "entities": [],
            })
            continue

        prompt = f"""
You are validating entity roles for a general-purpose query decomposition system.
This is a closed-set classification task. Do not invent, rewrite, merge, or add
entities.

User query:
{query}

Requirement:
{requirement}

Candidate phrases:
{candidates}

For each candidate, classify whether it is a true retrieval TARGET ENTITY for
this requirement. A target entity is a referential thing the requirement is
about and that could be independently identified/retrieved, such as a person,
organization, product, project, document, location, object, or other explicit
target.

Do NOT classify a candidate as an entity when it is functioning only as an
attribute, field name, date/time, number/value, measure, category, action,
relationship word, or other constraint on the target.

Return ONLY JSON in exactly this form:
{{
  "verdicts": [
    {{"candidate": "EXACT CANDIDATE TEXT", "is_entity": true}}
  ]
}}

Rules:
1. Use only the supplied candidates.
2. Preserve candidate text exactly.
3. `is_entity` must be true only for candidates functioning as retrieval targets
   in this requirement.
4. Do not infer a missing entity from context.
"""

        data = safe_completion_json(
            prompt,
            max_tokens=384,
            max_retries=1,
            fallback={"verdicts": []},
        )

        verdicts = data.get("verdicts", []) if isinstance(data, dict) else []
        allowed = set(candidates)
        verdict_map: dict[str, bool] = {}

        if isinstance(verdicts, list):
            for verdict in verdicts:
                if not isinstance(verdict, dict):
                    continue
                candidate = verdict.get("candidate")
                is_entity = verdict.get("is_entity")
                if (
                    isinstance(candidate, str)
                    and candidate in allowed
                    and isinstance(is_entity, bool)
                ):
                    verdict_map[candidate] = is_entity

        filtered = [
            candidate
            for candidate in candidates
            if verdict_map.get(candidate, False)
        ]

        validated.append({
            "requirement": requirement,
            "entities": filtered,
        })

    return validated


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
        seen_entity_keys = set()

        for entity in entities:
            if not isinstance(entity, str) or not entity.strip():
                continue

            value = entity.strip()
            entity_key = _normalize_entity(value)

            # Fail closed if the model invents or silently changes an entity.
            if not entity_key or entity_key in seen_entity_keys:
                continue
            if not _entity_is_explicit(value, requirement.strip(), query):
                continue

            cleaned_entities.append(value)
            seen_entity_keys.add(entity_key)

        cleaned.append({
            "requirement": requirement.strip(),
            "entities": cleaned_entities,
        })

    cleaned = cleaned or [{"requirement": query, "entities": []}]

    # Semantically distinguish retrieval targets from attributes, values, dates,
    # actions, and other constraints. This is intentionally domain-agnostic.
    cleaned = _validate_entity_roles(query, cleaned)

    # Fail closed on LLM grouping errors: independent target entities must not
    # silently share one retrieval/consolidation group.
    return _split_multi_entity_requirements(cleaned)



_MULTI_ENTITY_COMPARISON_PATTERNS = (
    r"\bamong\b",
    r"\bhighest\b",
    r"\blowest\b",
    r"\bmaximum\b",
    r"\bminimum\b",
    r"\bmost\b",
    r"\bleast\b",
    r"\bfirst\b",
    r"\blast\b",
    r"\btop\b",
    r"\bbottom\b",
    r"\bwinner\b",
    r"\bwinners\b",
    r"\bdifference\b",
    r"\bcompare\b",
    r"\bcomparison\b",
)


_RELATIONSHIP_PATTERNS = (
    r"\brelationship\s+between\b",
    r"\bconnection\s+between\b",
    r"\bassociation\s+between\b",
    r"\binteraction\s+between\b",
    r"\bdependency\s+between\b",
    r"\bintegration\s+between\b",
    r"\bcollaboration\s+between\b",
    r"\bcommunication\s+between\b",
)


def _is_relationship_requirement(requirement: str) -> bool:
    text = requirement.casefold()
    return any(re.search(pattern, text) for pattern in _RELATIONSHIP_PATTERNS)


def _should_split_multi_entity_requirement(
    requirement: str,
    entities: list[str],
) -> bool:
    """Identify multi-entity requests that need independent retrieval groups.

    Relationship-style questions remain intact. Collection/comparison requests
    are split because retrieval and consolidation must resolve each entity
    independently before a downstream answer step can compare or combine them.
    """
    if len(entities) <= 1:
        return False

    if _is_relationship_requirement(requirement):
        return False

    text = requirement.casefold()

    if any(re.search(pattern, text) for pattern in _MULTI_ENTITY_COMPARISON_PATTERNS):
        return True

    collection_pattern = re.compile(
        r"\b(?:what|which|where|when)\b\s+"
        r"[^?]*\b(?:are|were|is|was)\b[^?]*\bof\b",
        re.IGNORECASE,
    )
    if collection_pattern.search(requirement):
        return True

    behavior_pattern = re.compile(
        r"\b(?:what|which)\b[^?]*\b(?:do|does|use|manage|hold|have|has)\b",
        re.IGNORECASE,
    )
    return bool(behavior_pattern.search(requirement))


def _split_multi_entity_requirements(
    requirements: list[dict],
) -> list[dict]:
    """Prevent an otherwise valid multi-entity item from becoming one group."""
    expanded: list[dict] = []

    for item in requirements:
        requirement = str(item.get("requirement", "")).strip()
        entities = item.get("entities", [])
        if not requirement or not isinstance(entities, list):
            continue

        clean_entities = [
            str(entity).strip()
            for entity in entities
            if isinstance(entity, str) and entity.strip()
        ]

        if not _should_split_multi_entity_requirement(requirement, clean_entities):
            expanded.append({
                "requirement": requirement,
                "entities": clean_entities,
            })
            continue

        for entity in clean_entities:
            expanded.append({
                "requirement": (
                    f"{requirement}\n"
                    f"Target entity: {entity}. Retrieve only the requested factual "
                    "value for this entity; do not compare or combine it with the "
                    "other named entities."
                ),
                "entities": [entity],
            })

    return expanded

def decompose_query(query: str) -> list[str]:
    """Backward-compatible string-only decomposition API."""
    return [item["requirement"] for item in decompose_query_structured(query)]
