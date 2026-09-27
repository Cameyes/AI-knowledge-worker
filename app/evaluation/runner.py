import argparse
import json
import time
from pathlib import Path

from app.evaluation.metrics import (
    calculate_average_mrr,
    calculate_average_ndcg,
    calculate_keyword_coverage,
    calculate_recall_at_k,
    calculate_precision_at_k,
    calculate_subquery_mrr,
    calculate_subquery_recall_at_k,
    calculate_semantic_mrr,
    calculate_semantic_recall_at_k,
    calculate_semantic_ndcg,
    calculate_evidence_support_rate,
    _document_contains_keyword,
)

from app.evaluation.schemas import (
    EvaluationTestCase,
    EvaluationResult,
    RetrievalMetrics,
    RetrievedEvidence,
    SubqueryEvaluation,
)
from app.rag.evidence_retriever import multi_evidence_retrieve
from app.evaluation.judge import judge_subquery_evidence


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

EVALUATION_DIR = Path(__file__).resolve().parent

TEST_FILE = EVALUATION_DIR / "tests.jsonl"

RESULTS_DIR = EVALUATION_DIR / "results"
RESULTS_FILE = RESULTS_DIR / "retrieval_results.json"


# ---------------------------------------------------------
# Test loading
# ---------------------------------------------------------

def load_tests() -> list[EvaluationTestCase]:
    """
    Load evaluation questions from tests.jsonl.
    """

    tests = []

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            tests.append(
                EvaluationTestCase(
                    question=data["question"],
                    keywords=data.get("keywords", []),
                    reference_answer=data["reference_answer"],
                    category=data["category"],
                )
            )

    return tests


# ---------------------------------------------------------
# Cache
# ---------------------------------------------------------

def load_cached_results() -> dict[int, dict]:
    """
    Load previously completed evaluation results.

    The dictionary key is the test index.
    """

    if not RESULTS_FILE.exists():
        return {}

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        return {
            int(key): value
            for key, value in json.load(f).items()
        }


def save_cached_results(results: dict[int, dict]):
    """
    Persist evaluation results to disk.
    """

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ---------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------

def flatten_retrieved_results(
    evidence: list[dict],
) -> list[dict]:
    """
    Flatten results returned by multi_evidence_retrieve()
    while removing duplicate chunks.

    The first occurrence of each unique chunk is preserved.
    """

    retrieved_docs = []
    seen = set()

    for evidence_group in evidence:
        results = evidence_group.get("results", [])

        for result in results:
            document = result.get("document", "")
            metadata = result.get("metadata", {})
            source = metadata.get("source", "")

            key = (source, document)

            if key in seen:
                continue

            seen.add(key)
            retrieved_docs.append(result)

    return retrieved_docs


def build_evidence(
    retrieved_docs: list[dict],
) -> list[RetrievedEvidence]:
    """
    Convert raw retrieval results into evaluation evidence objects.
    """

    evidence = []

    for result in retrieved_docs:
        metadata = result.get("metadata", {})

        evidence.append(
            RetrievedEvidence(
                document=result.get("document", ""),
                source=metadata.get("source", ""),
                distance=result.get("distance"),
            )
        )

    return evidence


# ---------------------------------------------------------
# Single test evaluation
# ---------------------------------------------------------

def evaluate_retrieval(
    test: EvaluationTestCase,
    retrieval_top_k: int = 30,
    rerank_top_k: int = 10,
    metric_k: int = 10,
) -> EvaluationResult:

    start_time = time.perf_counter()

    evidence = multi_evidence_retrieve(
    test.question,
    retrieval_top_k=retrieval_top_k,
    rerank_top_k=rerank_top_k,
)

    subquery_evaluations = build_subquery_evaluations(
        evidence,
        test.keywords,
    )

    if subquery_evaluations:
        subquery_mrr = sum(
            item.mrr
            for item in subquery_evaluations
        ) / len(subquery_evaluations)

        subquery_recall_at_k = sum(
            item.recall_at_k
            for item in subquery_evaluations
        ) / len(subquery_evaluations)

        subquery_coverage = sum(
            1
            for item in subquery_evaluations
            if item.target_keywords
            and item.matched_keywords
        ) / len(subquery_evaluations)
    else:
        subquery_mrr = 0.0
        subquery_recall_at_k = 0.0
        subquery_coverage = 0.0

    if subquery_evaluations:
        semantic_mrr = sum(
            item.semantic.mrr
            for item in subquery_evaluations
        ) / len(subquery_evaluations)

        semantic_recall_at_k = sum(
            item.semantic.recall_at_k
            for item in subquery_evaluations
        ) / len(subquery_evaluations)

        semantic_ndcg = sum(
            item.semantic.ndcg
            for item in subquery_evaluations
        ) / len(subquery_evaluations)

        evidence_support_rate = sum(
            item.semantic.evidence_support_rate
            for item in subquery_evaluations
        ) / len(subquery_evaluations)
    else:
        semantic_mrr = 0.0
        semantic_recall_at_k = 0.0
        semantic_ndcg = 0.0
        evidence_support_rate = 0.0

    retrieved_docs = flatten_retrieved_results(evidence)

    print("\nSubquery diagnostics:")

    for i, subquery_eval in enumerate(
        subquery_evaluations,
        start=1,
    ):
        print(f"\n  Subquery {i}:")
        print(f"    {subquery_eval.subquery}")

        print(
            f"    Target keywords: "
            f"{subquery_eval.target_keywords}"
        )

        print(
            f"    Matched keywords: "
            f"{subquery_eval.matched_keywords}"
        )

        print(
            f"    Subquery MRR: "
            f"{subquery_eval.mrr:.4f}"
        )

        print(
            f"    Subquery Recall@{metric_k}: "
            f"{subquery_eval.recall_at_k:.4f}"
        )

        print(
    f"    Semantic MRR: "
    f"{subquery_eval.semantic.mrr:.4f}"
)

        print(
            f"    Semantic Recall@{metric_k}: "
            f"{subquery_eval.semantic.recall_at_k:.4f}"
        )

        print(
            f"    Semantic nDCG@{metric_k}: "
            f"{subquery_eval.semantic.ndcg:.4f}"
        )

        print(
            f"    Evidence Support Rate: "
            f"{subquery_eval.semantic.evidence_support_rate:.4f}"
        )

        print("    Jev support judgments:")

        for judgment in subquery_eval.evidence_judgments:
            print(
                f"      Rank {judgment.rank}: "
                f"{judgment.support_probability:.4f} "
                f"-> {judgment.supports_answer}"
            )

        print(
            f"    Retrieved chunks: "
            f"{len(subquery_eval.evidence)}"
        )

        print("    Top evidence:")

        for rank, item in enumerate(
            subquery_eval.evidence[:3],
            start=1,
        ):
            preview = (
                item.document
                .replace("\n", " ")
                .strip()
            )

            if len(preview) > 220:
                preview = preview[:220] + "..."

            print(
                f"      {rank}. {item.source}"
            )

            print(
                f"         {preview}"
            )

    print(
        f"\nSubquery Coverage: "
        f"{subquery_coverage:.4f}"
    )

    

    retrieval_metrics = RetrievalMetrics(
        mrr=calculate_average_mrr(
            test.keywords,
            retrieved_docs,
        ),
        ndcg=calculate_average_ndcg(
            test.keywords,
            retrieved_docs,
            k=metric_k,
        ),
        keyword_coverage=calculate_keyword_coverage(
            test.keywords,
            retrieved_docs,
        ),
        recall_at_k=calculate_recall_at_k(
            test.keywords,
            retrieved_docs,
            k=metric_k,
        ),
        precision_at_k=calculate_precision_at_k(
            test.keywords,
            retrieved_docs,
            k=metric_k,
        ),
    )

    evidence_objects = build_evidence(retrieved_docs)

    sources = list(
        dict.fromkeys(
            item.source
            for item in evidence_objects
            if item.source
        )
    )

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    return EvaluationResult(
        question=test.question,
        category=test.category,
        reference_answer=test.reference_answer,
        generated_answer="",
        retrieval=retrieval_metrics,
        semantic_retrieval={
            "mrr": semantic_mrr,
            "ndcg": semantic_ndcg,
            "recall_at_k": semantic_recall_at_k,
            "evidence_support_rate": evidence_support_rate,
        },
        answer=None,
        evidence=evidence_objects,
        sources=sources,
        subqueries=subquery_evaluations,
        subquery_mrr=subquery_mrr,
        subquery_recall_at_k=subquery_recall_at_k,
        subquery_coverage=subquery_coverage,
        latency_ms=latency_ms,
        passed=True,
    )


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Run RAG retrieval evaluation."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of tests to evaluate.",
    )

    parser.add_argument(
    "--start",
    type=int,
    default=None,
    help="Starting test number (1-based, inclusive).",
)

    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Ending test number (1-based, inclusive).",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all evaluation tests.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run tests even if cached.",
    )

    args = parser.parse_args()

    tests = load_tests()

    if args.start is not None or args.end is not None:
        start = (args.start or 1) - 1
        end = args.end or len(tests)

        selected_tests = [
            (index, test)
            for index, test in enumerate(tests)
            if start <= index < end
        ]

    elif args.all:
        selected_tests = list(enumerate(tests))

    elif args.limit is not None:
        selected_tests = list(enumerate(tests[:args.limit]))

    else:
        selected_tests = [(0, tests[0])]

    cached_results = load_cached_results()

    total = len(selected_tests)

    print(f"\nLoaded {len(tests)} evaluation tests.")
    print(f"Running {total} test(s).\n")

    for position, (index, test) in enumerate(
    selected_tests,
    start=1,
):

        if index in cached_results and not args.force:

            print(
                f"[{position}/{total}] "
                f"SKIPPED "
                f"(test {index + 1})"
            )

            continue

        print(
            f"\n{'=' * 70}"
        )
        print(
            f"Test {position}/{total} "
            f"(evaluation #{index + 1})"
        )
        print(
            f"Question: {test.question}"
        )
        print(
            f"{'=' * 70}"
        )

        try:

            result = evaluate_retrieval(test)

            cached_results[index] = result.model_dump()

            save_cached_results(cached_results)

            print("\nSubquery metrics:")

            print(
                f"Average Subquery MRR: "
                f"{result.subquery_mrr:.4f}"
            )

            print(
                f"Average Subquery Recall@10: "
                f"{result.subquery_recall_at_k:.4f}"
            )

            print(
                f"Subquery Coverage: "
                f"{result.subquery_coverage:.4f}"
            )

            print("\nSemantic retrieval metrics:")

            print(
            f"Semantic MRR: "
            f"{result.semantic_retrieval.mrr:.4f}"
            )

            print(
            f"Semantic nDCG@10: "
            f"{result.semantic_retrieval.ndcg:.4f}"
            )

            print(
            f"Semantic Recall@10: "
            f"{result.semantic_retrieval.recall_at_k:.4f}"
            )

            print(
            f"Evidence Support Rate: "
            f"{result.semantic_retrieval.evidence_support_rate:.4f}"
            )

            print(
                "\nRetrieval metrics:"
            )

            print(
                f"MRR: "
                f"{result.retrieval.mrr:.4f}"
            )

            print(
                f"nDCG: "
                f"{result.retrieval.ndcg:.4f}"
            )

            print(
                f"Keyword Coverage: "
                f"{result.retrieval.keyword_coverage:.2f}%"
            )

            print(
                f"Recall@10: "
                f"{result.retrieval.recall_at_k:.4f}"
            )

            print(
                f"Precision@10: "
                f"{result.retrieval.precision_at_k:.4f}"
            )

            print(
                f"Latency: "
                f"{result.latency_ms:.0f} ms"
            )

        except Exception as e:

            print(
                f"\nERROR: {e}"
            )

def build_subquery_evaluations(
    evidence: list[dict],
    keywords: list[str],
    metric_k: int = 10,
) -> list[SubqueryEvaluation]:
    """
    Evaluate each generated subquery using lexical and semantic evidence metrics.
    """

    subquery_evaluations = []

    for evidence_group in evidence:
        subquery = evidence_group.get(
            "subquery",
            "",
        )

        results = evidence_group.get(
            "results",
            [],
        )

        evidence_objects = build_evidence(
            results
        )

        # Keywords explicitly represented in this subquery.
        target_keywords = [
            keyword
            for keyword in keywords
            if keyword.lower() in subquery.lower()
        ]

        # Keywords actually found in this subquery's results.
        matched_keywords = []

        for keyword in target_keywords:
            if any(
                _document_contains_keyword(
                    keyword,
                    result,
                )
                for result in results
            ):
                matched_keywords.append(keyword)

        # Existing deterministic subquery metrics.
        subquery_mrr = calculate_subquery_mrr(
            subquery,
            keywords,
            results,
        )

        subquery_recall = (
            calculate_subquery_recall_at_k(
                subquery,
                keywords,
                results,
                k=metric_k,
            )
        )

        # -------------------------------------------------
        # Semantic evidence evaluation using Jev.
        # -------------------------------------------------

        evidence_judgments = judge_subquery_evidence(
            subquery,
            results,
        )

        semantic_mrr = calculate_semantic_mrr(
            evidence_judgments,
        )

        semantic_recall = calculate_semantic_recall_at_k(
            evidence_judgments,
            k=metric_k,
        )

        semantic_ndcg = calculate_semantic_ndcg(
            evidence_judgments,
            k=metric_k,
        )

        evidence_support_rate = (
            calculate_evidence_support_rate(
                evidence_judgments,
                k=metric_k,
            )
        )

        subquery_evaluations.append(
            SubqueryEvaluation(
                subquery=subquery,
                target_keywords=target_keywords,
                mrr=subquery_mrr,
                recall_at_k=subquery_recall,
                evidence=evidence_objects,
                matched_keywords=matched_keywords,
                semantic={
                    "mrr": semantic_mrr,
                    "recall_at_k": semantic_recall,
                    "ndcg": semantic_ndcg,
                    "evidence_support_rate": (
                        evidence_support_rate
                    ),
                },
                evidence_judgments=evidence_judgments,
            )
        )

    return subquery_evaluations

if __name__ == "__main__":
    main()