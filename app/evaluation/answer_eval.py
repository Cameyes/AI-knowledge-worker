import argparse
import json
import re
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from app.rag.answer_generator import generate_answer
from app.rag.llm_client import safe_completion_json


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

EVALUATION_DIR = Path(__file__).resolve().parent

DEFAULT_TESTS_FILE = EVALUATION_DIR / "tests.jsonl"
DEFAULT_RESULTS_FILE = EVALUATION_DIR / "results" / "retrieval_results.json"
DEFAULT_OUTPUT_FILE = (
    EVALUATION_DIR / "results" / "answer_baseline_results.json"
)


# ---------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------

DEFAULT_SMOKE_TESTS = [
    1,    # direct_fact
    31,   # temporal
    51,   # numerical
    71,   # spanning
    97,   # multi_entity
    117,  # comparative
    136,  # relationship
]


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

from dotenv import load_dotenv
load_dotenv(override=True)


# ---------------------------------------------------------
# Answer judge schema
# ---------------------------------------------------------

class AnswerJudge(BaseModel):
    """LLM-as-a-judge scores for generated answers."""

    accuracy: float = Field(ge=1, le=5)
    completeness: float = Field(ge=1, le=5)
    relevance: float = Field(ge=1, le=5)
    groundedness: float = Field(ge=1, le=5)
    citation_correctness: float = Field(ge=1, le=5)
    feedback: str


# ---------------------------------------------------------
# Test loading
# ---------------------------------------------------------

def load_tests(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


# ---------------------------------------------------------
# Employee helper
# ---------------------------------------------------------

def get_employee_name(source: str) -> str:
    filename = (
        source
        .replace("\\", "/")
        .split("/")[-1]
    )
    return filename.removesuffix(".md")


# ---------------------------------------------------------
# Reconstruct answer-generation context
# ---------------------------------------------------------

def build_answer_context(result: dict) -> dict[str, list[dict]]:
    """
    Reconstruct all saved retrieval evidence.

    Different employees may satisfy different subqueries.
    """

    grouped: dict[str, list[dict]] = {}

    for subquery_item in result.get("subqueries", []):
        subquery = subquery_item.get("subquery", "")

        if not subquery:
            continue

        for evidence in subquery_item.get("evidence", []):
            if not isinstance(evidence, dict):
                continue

            source = evidence.get("source", "")
            document = evidence.get("document", "")

            if not source or not document:
                continue

            result_item = {
                "document": document,
                "metadata": {
                    "source": source,
                },
                "distance": evidence.get("distance"),
            }

            employee = get_employee_name(source)

            grouped.setdefault(employee, []).append(
                {
                    "subquery": subquery,
                    "result": result_item,
                }
            )

    return grouped


# ---------------------------------------------------------
# Query-aware evidence selection
# ---------------------------------------------------------

def _query_terms(
    question: str,
    reference_answer: str = "",
    generated_answer: str = "",
) -> set[str]:
    text = (
        f"{question} "
        f"{reference_answer} "
        f"{generated_answer}"
    ).lower()

    words = re.findall(r"[a-z0-9]+", text)

    stopwords = {
        "what", "was", "were", "who", "how", "much", "does", "did",
        "the", "a", "an", "is", "are", "in", "on", "of", "to", "and",
        "or", "for", "from", "among", "with", "their", "his", "her",
        "current", "work", "works", "had", "has", "have", "than",
        "employee", "employees",
    }

    return {
        word
        for word in words
        if len(word) > 1 and word not in stopwords
    }


def _rank_evidence(
    grouped_evidence: dict[str, list[dict]],
    question: str,
    reference_answer: str = "",
    generated_answer: str = "",
    max_items_per_group: int = 2,
) -> dict[str, list[dict]]:
    """
    Keep the most relevant chunks per group for answer generation/judging.

    The stored retrieval results remain untouched. This only reduces the
    prompt supplied to the generation LLM during this baseline evaluation.
    """

    terms = _query_terms(
        question,
        reference_answer,
        generated_answer,
    )

    compact: dict[str, list[dict]] = {}

    for group_key, items in grouped_evidence.items():
        ranked = []

        for item in items:
            document = item["result"].get("document", "")
            subquery = item.get("subquery", "")

            haystack = (
                f"{group_key} "
                f"{subquery} "
                f"{document}"
            ).lower()

            score = sum(
                1
                for term in terms
                if term in haystack
            )

            # Small preference for shorter, focused chunks.
            length_penalty = min(len(document) / 5000, 1.0)

            final_score = score - (0.05 * length_penalty)

            ranked.append(
                (final_score, item)
            )

        ranked.sort(
            key=lambda pair: pair[0],
            reverse=True,
        )

        compact[group_key] = [
            item
            for _, item in ranked[:max_items_per_group]
        ]

    return compact


def build_compact_context(
    grouped_evidence: dict[str, list[dict]],
    question: str,
    reference_answer: str = "",
    generated_answer: str = "",
    max_chars: int = 10000,
) -> str:
    """
    Build a bounded judge context.

    Keeps each group represented while limiting the total prompt size.
    """

    selected = _rank_evidence(
        grouped_evidence,
        question=question,
        reference_answer=reference_answer,
        generated_answer=generated_answer,
        max_items_per_group=3,
    )

    blocks: list[str] = []

    for group_key, items in selected.items():
        blocks.append(f"GROUP: {group_key}")

        for item in items:
            source = item["result"]["metadata"].get(
                "source",
                "Unknown source",
            )

            filename = (
                source
                .replace("\\", "/")
                .split("/")[-1]
            )

            blocks.append(
                f"Subquery: {item['subquery']}\n"
                f"Source: {filename}\n"
                f"Evidence:\n"
                f"{item['result']['document']}"
            )

    context = "\n\n".join(blocks)

    if len(context) <= max_chars:
        return context

    return context[:max_chars]


# ---------------------------------------------------------
# Answer judge
# ---------------------------------------------------------

def judge_answer(
    question: str,
    reference_answer: str,
    generated_answer: str,
    evidence_context: str,
) -> AnswerJudge:
    """
    Judge the generated answer using structured JSON output.
    """

    prompt = f"""
You are an enterprise RAG answer evaluator.

Evaluate ONLY against the reference answer and retrieved evidence.
Do not use outside knowledge.

Question:
{question}

Reference answer:
{reference_answer}

Generated answer:
{generated_answer}

Retrieved evidence:
{evidence_context}

Score each dimension from 1 to 5:

- accuracy: factual correctness of every substantive claim.
- completeness: whether every requested part/entity/value is answered.
- relevance: whether the answer directly addresses the question.
- groundedness: whether substantive claims are supported by the evidence.
- citation_correctness: whether supplied source citations/attributions
  correctly correspond to the evidence.
  If no citation/source attribution is present, do not invent one.

For numerical questions, verify arithmetic.
For multi-entity/comparative questions, verify every named entity separately.
Do not penalize concise answers for omitting irrelevant details.

Return ONLY a JSON object matching the required schema.
Keep feedback to one short sentence.
"""

    judge_result = safe_completion_json(
        prompt,
        max_tokens=256,
        max_retries=3,
        fallback=None,
    )

    if not judge_result:
        raise ValueError(
            "Answer judge returned no structured result."
        )

    try:
        return AnswerJudge.model_validate(judge_result)
    except Exception as exc:
        raise ValueError(
            "Answer judge returned an invalid structured result.\n"
            f"Raw result:\n{judge_result}"
        ) from exc


# ---------------------------------------------------------
# Run one test
# ---------------------------------------------------------

def run_one(
    test_number: int,
    test: dict,
    results_by_index: dict,
) -> dict:

    result_index = test_number - 1

    stored = results_by_index.get(result_index)

    if stored is None:
        raise KeyError(
            f"No saved retrieval result for test "
            f"{test_number}. Run the retrieval evaluation first."
        )

    grouped_evidence = build_answer_context(stored)

    if not grouped_evidence:
        raise RuntimeError(
            f"Test {test_number}: saved retrieval evidence "
            f"could not be reconstructed."
        )

    # -----------------------------------------------------
    # IMPORTANT:
    # Limit generation context during this baseline run.
    # Stored retrieval results are NOT modified.
    # -----------------------------------------------------

    generation_evidence = _rank_evidence(
        grouped_evidence,
        question=test["question"],
        # IMPORTANT: reference answer is unavailable at inference time.
        # Do not leak it into generation-time evidence selection.
        reference_answer="",
        generated_answer="",
        max_items_per_group=3,
    )

    # print("\n===== GENERATION EVIDENCE =====")
    # print(json.dumps(generation_evidence, indent=2, ensure_ascii=False))
    # print("===== END GENERATION EVIDENCE =====\n")

    # -----------------------------------------------------
    # Generate answer
    # -----------------------------------------------------

    answer_result = generate_answer(
        test["question"],
        generation_evidence,
    )

    generated_answer = answer_result["answer"]

    # Do not turn provider failures into 1/5 answer-quality scores.
    fallback_answer = (
        "I could not generate an answer "
        "due to a temporary service issue."
    )

    if generated_answer.strip() == fallback_answer:
        raise RuntimeError(
            "generation_error: LLM returned the configured fallback response."
        )

    # -----------------------------------------------------
    # Build compact judge evidence
    # -----------------------------------------------------

    evidence_context = build_compact_context(
        generation_evidence,
        question=test["question"],
        reference_answer="",
        generated_answer=generated_answer,
        # max_chars=10000,
    )

    # -----------------------------------------------------
    # Judge answer
    # -----------------------------------------------------

    judge = judge_answer(
        question=test["question"],
        reference_answer=test["reference_answer"],
        generated_answer=generated_answer,
        evidence_context=evidence_context,
    )

    return {
        "test_number": test_number,
        "category": test["category"],
        "question": test["question"],
        "reference_answer": test["reference_answer"],
        "generated_answer": generated_answer,
        "matched_employee_count": len(
            grouped_evidence
        ),
        "answer_metrics": judge.model_dump(),
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Baseline answer-generation evaluation "
            "using saved retrieval evidence."
        )
    )

    parser.add_argument(
        "--tests",
        nargs="+",
        type=int,
        default=DEFAULT_SMOKE_TESTS,
        help="1-based evaluation test numbers.",
    )

    parser.add_argument(
        "--tests-file",
        type=Path,
        default=DEFAULT_TESTS_FILE,
    )

    parser.add_argument(
        "--results-file",
        type=Path,
        default=DEFAULT_RESULTS_FILE,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Load tests
    # -----------------------------------------------------

    tests = load_tests(args.tests_file)

    # -----------------------------------------------------
    # Load retrieval results
    # -----------------------------------------------------

    with args.results_file.open(
        "r",
        encoding="utf-8",
    ) as f:
        cached_results = json.load(f)

    results_by_index = {
        int(key): value
        for key, value in cached_results.items()
    }

    # -----------------------------------------------------
    # Select tests
    # -----------------------------------------------------

    selected = []

    for number in args.tests:

        if number < 1 or number > len(tests):
            raise ValueError(
                f"Test number {number} is outside "
                f"1..{len(tests)}"
            )

        selected.append(
            (
                number,
                tests[number - 1],
            )
        )

    # -----------------------------------------------------
    # Run
    # -----------------------------------------------------

    output = []

    print("\nANSWER-GENERATION BASELINE\n")

    for position, (number, test) in enumerate(
        selected,
        start=1,
    ):

        print("=" * 80)

        print(
            f"Test {position}/{len(selected)} "
            f"— #{number} "
            f"— {test['category']}"
        )

        print(
            f"Question: {test['question']}"
        )

        print("=" * 80)

        try:

            result = run_one(
                number,
                test,
                results_by_index,
            )

            output.append(result)

            metrics = result["answer_metrics"]

            print("\nGenerated answer:")

            print(result["generated_answer"])

            print()

            print(
                "Scores: "
                f"accuracy={metrics['accuracy']}/5, "
                f"completeness={metrics['completeness']}/5, "
                f"relevance={metrics['relevance']}/5, "
                f"groundedness={metrics['groundedness']}/5, "
                f"citation={metrics['citation_correctness']}/5"
            )

            print(
                f"Feedback: {metrics['feedback']}"
            )

        except Exception as exc:

            print(
                f"ERROR: {exc}"
            )

            output.append(
                {
                    "test_number": number,
                    "category": test["category"],
                    "question": test["question"],
                    "error": str(exc),
                }
            )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    successful = [
        item
        for item in output
        if "answer_metrics" in item
    ]

    print("\n" + "=" * 80)
    print("BASELINE SUMMARY")
    print("=" * 80)

    print(
        f"Tests attempted: {len(output)}"
    )

    print(
        f"Tests successfully judged: "
        f"{len(successful)}"
    )

    if successful:

        for field in [
            "accuracy",
            "completeness",
            "relevance",
            "groundedness",
            "citation_correctness",
        ]:

            avg = (
                sum(
                    item["answer_metrics"][field]
                    for item in successful
                )
                / len(successful)
            )

            print(
                f"Average {field}: "
                f"{avg:.2f}/5"
            )

    print(
        f"\nSaved: {args.output}"
    )


if __name__ == "__main__":
    main()
