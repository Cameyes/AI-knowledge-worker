import json
from pathlib import Path

from app.evaluation.judge import judge_subquery_evidence
from app.rag.evidence_retriever import multi_evidence_retrieve


TEST_FILE = Path(__file__).resolve().parent / "tests.jsonl"


def load_tests():
    tests = []

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tests.append(json.loads(line))

    return tests


def select_smoke_tests(tests):
    selected = []

    categories = [
        "direct_fact",
        "temporal",
        "numerical",
        "relationship",
        "spanning",
        "multi_entity",
        "comparative",
    ]

    # First test from each category.
    for category in categories:
        for test in tests:
            if test["category"] == category:
                selected.append(test)
                break

    # Add three more diverse cases.
    for test in tests:
        if test["category"] in {"multi_entity", "comparative", "spanning"}:
            if test not in selected:
                selected.append(test)

        if len(selected) == 10:
            break

    return selected[:10]


def main():
    tests = load_tests()
    smoke_tests = select_smoke_tests(tests)

    print(f"Loaded {len(tests)} tests.")
    print(f"Running Jev smoke test on {len(smoke_tests)} questions.\n")

    for index, test in enumerate(smoke_tests, start=1):
        question = test["question"]

        print("=" * 100)
        print(f"SMOKE TEST {index}/10")
        print(f"Category: {test['category']}")
        print(f"Question: {question}")
        print("=" * 100)

        try:
            evidence = multi_evidence_retrieve(
                question,
                retrieval_top_k=30,
                rerank_top_k=10,
            )

            total_supported = 0

            for subquery_data in evidence:
                subquery = subquery_data["subquery"]
                results = subquery_data["results"]

                judgments = judge_subquery_evidence(
                    subquery,
                    results,
                )

                print(f"\nSubquery: {subquery}")

                for rank, (result, judgment) in enumerate(
                    zip(results, judgments),
                    start=1,
                ):
                    print(
                        f"  Rank {rank}: "
                        f"{judgment.source} | "
                        f"support={judgment.support_probability:.2f} | "
                        f"supports={judgment.supports_answer}"
                    )

                    if judgment.supports_answer:
                        total_supported += 1

            print(
                f"\nSupported evidence chunks: {total_supported}"
            )

        except Exception as exc:
            print(f"\nERROR: {exc}")

        print()


if __name__ == "__main__":
    main()