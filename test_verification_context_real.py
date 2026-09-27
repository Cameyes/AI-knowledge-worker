from app.rag.retriever import retrieve
from app.rag.reranker import rerank
from app.rag.verification_context import compact_evidence


query = "What was David Kim's 2023 performance rating?"

print("=" * 80)
print("REAL PIPELINE VERIFICATION-CONTEXT TEST")
print("=" * 80)

# 1. Retrieve 10 candidates
retrieved = retrieve(
    query,
    top_k=10,
)

# 2. Rerank -> keep 8
reranked = rerank(
    query,
    retrieved,
    top_k=8,
)

print(f"\nRetrieved: {len(retrieved)}")
print(f"After reranking: {len(reranked)}")

# 3. Compact each real reranked result
print("\n" + "=" * 80)
print("RERANKED RESULTS + COMPACTION")
print("=" * 80)

for rank, result in enumerate(reranked, start=1):

    source = result["metadata"].get(
        "source",
        "Unknown source",
    )

    original = result["document"]

    compacted = compact_evidence(
        query=query,
        document=original,
    )

    original_chars = len(original)
    compacted_chars = len(compacted)

    reduction = (
        1 - compacted_chars / original_chars
    ) * 100 if original_chars else 0

    print("\n" + "-" * 80)
    print(f"RERANK RANK: {rank}")
    print(f"SOURCE: {source}")
    print(f"RERANK SCORE: {result['rerank_score']:.4f}")
    print(f"ORIGINAL CHARS: {original_chars}")
    print(f"COMPACTED CHARS: {compacted_chars}")
    print(f"REDUCTION: {reduction:.1f}%")

    print("\nORIGINAL:")
    print(original)

    print("\nCOMPACTED:")
    print(compacted)