from app.rag.retriever import retrieve
from app.rag.reranker import rerank


QUERIES = [
    "Which employee had a 2023 performance rating of 4.7/5?",
    "who received recognition for customer-related performance?"
]


def main():

    for query in QUERIES:

        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        # First-stage retrieval
        results = retrieve(
            query,
            top_k=20,
        )


        # print("\n========== CHROMA TOP 10 ==========")

        # for i, result in enumerate(results, start=1):
        #     print(f"\n--- Chroma Result {i} ---")
        #     print(f"Distance: {result['distance']}")
        #     print(f"Source: {result['metadata']['source']}")
        #     print(result["document"][:500])

        print("\n========== RERANKED TOP 10 ==========")

        # Second-stage reranking
        reranked_results = rerank(
            query,
            results,
            top_k=10,
        )

        for i, result in enumerate(
            reranked_results,
            start=1,
        ):
            print(f"\n--- Result {i} ---")
            print(
                f"Rerank score: "
                f"{result['rerank_score']}"
            )
            print(
                f"Chroma distance: "
                f"{result['distance']}"
            )
            print(
                f"Source: "
                f"{result['metadata']['source']}"
            )
            print(result["document"])


if __name__ == "__main__":
    main()