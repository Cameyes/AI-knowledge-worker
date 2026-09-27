from app.evaluation.judge import judge_subquery_evidence
from app.rag.retriever import retrieve
from app.rag.reranker import rerank


def main():
    query = (
        "What was the 2023 performance rating "
        "of Samuel Trenton?"
    )

    print("Retrieving...")
    results = retrieve(
        query,
        top_k=30,
    )

    print("Reranking...")
    reranked = rerank(
        query,
        results,
        top_k=10,
    )

    print("Running Jev...")
    judgments = judge_subquery_evidence(
        query,
        reranked,
    )

    print("\nJev results:\n")

    for rank, (result, judgment) in enumerate(
        zip(reranked, judgments),
        start=1,
    ):
        document = result.get("document", "")
        source = judgment.source

        print("=" * 100)
        print(f"RANK: {rank}")
        print(f"SOURCE: {source}")
        print(f"JEV SUPPORT PROBABILITY: {judgment.support_probability:.4f}")
        print(f"JEV SUPPORTS ANSWER: {judgment.supports_answer}")
        print("\nFULL CHUNK:")
        print(document)
        print("=" * 100)


if __name__ == "__main__":
    main()