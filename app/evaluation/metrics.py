import math


# ---------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------

def _document_contains_keyword(keyword: str, document: dict) -> bool:
    keyword = keyword.lower()

    content = document.get("document", "").lower()

    metadata = document.get("metadata", {}) or {}
    source = metadata.get("source", "").lower()

    return keyword in content or keyword in source


def _subquery_keywords(
    subquery: str,
    keywords: list[str],
) -> list[str]:
    """
    Select evaluation keywords that are explicitly present
    in the generated subquery.

    This gives us a deterministic target for the current
    benchmark without inventing entity-specific logic.
    """

    subquery_lower = subquery.lower()

    return [
        keyword
        for keyword in keywords
        if keyword.lower() in subquery_lower
    ]


# ---------------------------------------------------------
# Existing retrieval metrics
# ---------------------------------------------------------

def calculate_mrr(
    keyword: str,
    retrieved_docs: list[dict],
) -> float:
    keyword = keyword.lower()

    for rank, doc in enumerate(
        retrieved_docs,
        start=1,
    ):
        content = doc.get("document", "")

        if keyword in content.lower():
            return 1.0 / rank

    return 0.0


def calculate_average_mrr(
    keywords: list[str],
    retrieved_docs: list[dict],
) -> float:
    if not keywords:
        return 0.0

    scores = [
        calculate_mrr(
            keyword,
            retrieved_docs,
        )
        for keyword in keywords
    ]

    return sum(scores) / len(scores)


def calculate_dcg(
    relevances: list[int],
    k: int,
) -> float:
    dcg = 0.0

    for i, relevance in enumerate(
        relevances[:k]
    ):
        dcg += relevance / math.log2(i + 2)

    return dcg


def calculate_ndcg(
    keyword: str,
    retrieved_docs: list[dict],
    k: int = 10,
) -> float:
    keyword = keyword.lower()

    relevances = [
        1
        if keyword in doc.get("document", "").lower()
        else 0
        for doc in retrieved_docs[:k]
    ]

    dcg = calculate_dcg(
        relevances,
        k,
    )

    ideal_relevances = sorted(
        relevances,
        reverse=True,
    )

    idcg = calculate_dcg(
        ideal_relevances,
        k,
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


def calculate_average_ndcg(
    keywords: list[str],
    retrieved_docs: list[dict],
    k: int = 10,
) -> float:
    if not keywords:
        return 0.0

    scores = [
        calculate_ndcg(
            keyword,
            retrieved_docs,
            k,
        )
        for keyword in keywords
    ]

    return sum(scores) / len(scores)


def calculate_keyword_coverage(
    keywords: list[str],
    retrieved_docs: list[dict],
) -> float:
    if not keywords:
        return 0.0

    found = 0

    for keyword in keywords:

        keyword_lower = keyword.lower()

        if any(
            keyword_lower in doc.get(
                "document",
                "",
            ).lower()
            for doc in retrieved_docs
        ):
            found += 1

    return (
        found / len(keywords)
    ) * 100


def calculate_recall_at_k(
    keywords: list[str],
    retrieved_docs: list[dict],
    k: int = 10,
) -> float:
    if not keywords:
        return 0.0

    top_k_docs = retrieved_docs[:k]

    found = 0

    for keyword in keywords:

        keyword_lower = keyword.lower()

        if any(
            keyword_lower in doc.get(
                "document",
                "",
            ).lower()
            for doc in top_k_docs
        ):
            found += 1

    return found / len(keywords)


def calculate_precision_at_k(
    keywords: list[str],
    retrieved_docs: list[dict],
    k: int = 10,
) -> float:
    if not keywords or not retrieved_docs:
        return 0.0

    top_k_docs = retrieved_docs[:k]

    relevant_docs = 0

    for doc in top_k_docs:

        content = doc.get(
            "document",
            "",
        ).lower()

        if any(
            keyword.lower() in content
            for keyword in keywords
        ):
            relevant_docs += 1

    return (
        relevant_docs / len(top_k_docs)
    )


# ---------------------------------------------------------
# Subquery-level metrics - becuase in Test 1 , multi-entity level questions failed
# ---------------------------------------------------------

def calculate_subquery_mrr(
    subquery: str,
    keywords: list[str],
    retrieved_docs: list[dict],
) -> float:
    """
    Calculate MRR for keywords explicitly represented
    in the generated subquery.

    Matching is provenance-aware:
    - document content
    - metadata.source
    """

    target_keywords = _subquery_keywords(
        subquery,
        keywords,
    )

    if not target_keywords:
        return 0.0

    scores = []

    for keyword in target_keywords:
        score = 0.0

        for rank, doc in enumerate(
            retrieved_docs,
            start=1,
        ):
            if _document_contains_keyword(
                keyword,
                doc,
            ):
                score = 1.0 / rank
                break

        scores.append(score)

    return sum(scores) / len(scores)


def calculate_subquery_recall_at_k(
    subquery: str,
    keywords: list[str],
    retrieved_docs: list[dict],
    k: int = 10,
) -> float:
    """
    Recall@K for keywords explicitly represented
    in the generated subquery.

    Matching is provenance-aware:
    - document content
    - metadata.source
    """

    target_keywords = _subquery_keywords(
        subquery,
        keywords,
    )

    if not target_keywords:
        return 0.0

    top_k_docs = retrieved_docs[:k]

    found = 0

    for keyword in target_keywords:
        if any(
            _document_contains_keyword(
                keyword,
                doc,
            )
            for doc in top_k_docs
        ):
            found += 1

    return found / len(target_keywords)


def calculate_subquery_coverage(
    subqueries: list[dict],
) -> float:
    """
    Percentage of generated subqueries for which
    at least one target keyword was retrieved.

    Matching is provenance-aware:
    - document content
    - metadata.source
    """

    if not subqueries:
        return 0.0

    covered = 0

    for subquery_data in subqueries:
        target_keywords = subquery_data.get(
            "target_keywords",
            [],
        )

        retrieved_docs = subquery_data.get(
            "retrieved_docs",
            [],
        )

        if not target_keywords:
            continue

        found = any(
            _document_contains_keyword(
                keyword,
                doc,
            )
            for keyword in target_keywords
            for doc in retrieved_docs
        )

        if found:
            covered += 1

    return covered / len(subqueries)

# ---------------------------------------------------------
# Semantic evidence metrics
# ---------------------------------------------------------

def calculate_semantic_mrr(
    evidence_judgments,
) -> float:
    """
    MRR based on the first retrieved chunk judged to support the answer.
    """

    if not evidence_judgments:
        return 0.0

    for rank, judgment in enumerate(
        evidence_judgments,
        start=1,
    ):
        if judgment.supports_answer:
            return 1.0 / rank

    return 0.0


def calculate_semantic_recall_at_k(
    evidence_judgments,
    k: int = 10,
) -> float:
    """
    Recall@K based on whether answer-supporting evidence appears in top-K.
    """

    if not evidence_judgments:
        return 0.0

    top_k = evidence_judgments[:k]

    if any(
        judgment.supports_answer
        for judgment in top_k
    ):
        return 1.0

    return 0.0


def calculate_semantic_ndcg(
    evidence_judgments,
    k: int = 10,
) -> float:
    """
    nDCG@K using Jev support probabilities as graded relevance.
    """

    if not evidence_judgments:
        return 0.0

    top_k = evidence_judgments[:k]

    relevances = [
        judgment.support_probability
        for judgment in top_k
    ]

    dcg = calculate_dcg(
        relevances,
        k,
    )

    ideal_relevances = sorted(
        relevances,
        reverse=True,
    )

    idcg = calculate_dcg(
        ideal_relevances,
        k,
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


def calculate_evidence_support_rate(
    evidence_judgments,
    k: int = 10,
) -> float:
    """
    Percentage of top-K retrieved chunks that support the answer.
    """

    if not evidence_judgments:
        return 0.0

    top_k = evidence_judgments[:k]

    supported = sum(
        1
        for judgment in top_k
        if judgment.supports_answer
    )

    return supported / len(top_k)