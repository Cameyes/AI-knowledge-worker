def merge_evidence(
    evidence: list[dict],
) -> list[dict]:
    """
    Combine evidence retrieved for multiple subqueries
    into a single evidence collection.

    The merger is domain-agnostic.

    It does not assume anything about the type of entity
    mentioned in the documents.
    """

    merged_evidence = []

    for evidence_group in evidence:

        subquery = evidence_group["subquery"]

        for result in evidence_group["results"]:

            merged_evidence.append(
                {
                    "subquery": subquery,
                    "result": result,
                }
            )

    return merged_evidence