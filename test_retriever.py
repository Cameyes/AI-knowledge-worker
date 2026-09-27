# from app.rag.retriever import retrieve


# def main():

#     query = "Which employee had a 2023 performance rating of 4.7/5?"

#     results = retrieve(
#         query,
#         top_k=10,
#     )

#     for i, result in enumerate(
#         results,
#         start=1,
#     ):
#         print(f"\n{'=' * 80}")
#         print(f"RESULT #{i}")
#         print(f"{'=' * 80}")

#         print(f"Distance: {result['distance']}")
#         print(f"Source: {result['metadata']['source']}")

#         print("\nDocument:")
#         print(result["document"])


# if __name__ == "__main__":
#     main()


from app.rag.retriever import retrieve
from app.rag.reranker import rerank


def main():
    query = "What was David Kim's 2023 performance rating?"

    print("=" * 80)
    print("RAW RETRIEVAL")
    print("=" * 80)

    results = retrieve(query, top_k=10)

    for i, result in enumerate(results, start=1):
        print(f"\n--- RESULT {i} ---")
        print("Source:", result["metadata"].get("source"))
        print("Distance:", result["distance"])
        print(result["document"])

    print("\n")
    print("=" * 80)
    print("RERANKED RESULTS")
    print("=" * 80)

    reranked = rerank(query, results, top_k=10)

    for i, result in enumerate(reranked, start=1):
        print(f"\n--- RESULT {i} ---")
        print("Source:", result["metadata"].get("source"))
        print("Rerank score:", result["rerank_score"])
        print(result["document"])


if __name__ == "__main__":
    main()