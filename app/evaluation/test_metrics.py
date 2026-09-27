from app.evaluation.metrics import (
    calculate_mrr,
    calculate_average_mrr,
    calculate_ndcg,
    calculate_average_ndcg,
    calculate_keyword_coverage,
    calculate_recall_at_k,
    calculate_precision_at_k,
)


retrieved_docs = [
    {
        "document": "Kevin Zhang is a Software Engineer.",
        "metadata": {"source": "Kevin Zhang.md"},
    },
    {
        "document": "David Kim is a DevOps Engineer with a 2022 bonus of $8,000.",
        "metadata": {"source": "David Kim.md"},
    },
    {
        "document": "Robert Chen received a bonus of $25,000 in 2022.",
        "metadata": {"source": "Robert Chen.md"},
    },
]


keywords = [
    "David Kim",
    "DevOps Engineer",
    "$8,000",
]


print("MRR")
print(calculate_average_mrr(keywords, retrieved_docs))

print("\nnDCG")
print(calculate_average_ndcg(keywords, retrieved_docs, k=3))

print("\nKeyword Coverage")
print(calculate_keyword_coverage(keywords, retrieved_docs))

print("\nRecall@3")
print(calculate_recall_at_k(keywords, retrieved_docs, k=3))

print("\nPrecision@3")
print(calculate_precision_at_k(keywords, retrieved_docs, k=3))