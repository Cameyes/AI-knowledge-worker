# from app.rag.retriever import retrieve


# QUERY = "What was Marcus's rating in 2023?"

# results = retrieve(
#     QUERY,
#     top_k=50,
# )

# print(f"\nQUERY: {QUERY}")
# print(f"Retrieved: {len(results)} chunks\n")

# for rank, result in enumerate(results, start=1):
#     metadata = result.get("metadata", {})
#     source = metadata.get("source", "")
#     document = result.get("document", "")
#     distance = result.get("distance")

#     if "marcus" in source.lower() or "marcus" in document.lower():
#         print("=" * 100)
#         print(f"RANK: {rank}")
#         print(f"DISTANCE: {distance}")
#         print(f"SOURCE: {source}")
#         print("-" * 100)
#         print(document)
#         print()

from app.rag.retriever import retrieve
from app.rag.reranker import rerank


QUERY = "What was Marcus's rating in 2023?"

results = retrieve(
    QUERY,
    top_k=30,
)

reranked = rerank(
    QUERY,
    results,
    top_k=30,
)

print("\n" + "=" * 100)
print("MARCUS RESULTS AFTER RERANKING")
print("=" * 100)

for rank, result in enumerate(reranked, start=1):
    source = result.get("metadata", {}).get("source", "")
    
    if "marcus" in source.lower():
        print("\n" + "-" * 80)
        print(f"RANK: {rank}")
        print(f"SCORE: {result.get('rerank_score')}")
        print(f"SOURCE: {source}")
        print("-" * 80)
        print(result["document"])