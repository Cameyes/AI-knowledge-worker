import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from app.rag.answer_generator import generate_answer
from app.rag.evidence_consolidator import consolidate_evidence
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

# Keep evaluation-only LLM cache beside benchmark outputs rather than in the
# application runtime cache. Do not override an explicit user setting.
os.environ.setdefault(
    "RAG_LLM_CACHE_DIR",
    str(EVALUATION_DIR / "results" / "llm_cache"),
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
# Evaluation reproducibility
# ---------------------------------------------------------
# Answer evaluation uses the shared LLM client. Enable greedy sampling and an
# exact-response cache by default so repeated runs over identical inputs are
# reproducible. The cache is keyed by the complete prompt/model/mode/token
# budget, so changed prompts naturally create new entries.
os.environ.setdefault("RAG_LLM_DETERMINISTIC", "1")
os.environ.setdefault("RAG_LLM_CACHE", "1")


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
    feedback: str = ""


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
# Generation evidence budget
# ---------------------------------------------------------
# Candidates handed to the consolidator per group (scoped, then ranked, then cut
# to GENERATION_CANDIDATES_PER_GROUP). Set RAG_ANSWER_CANDIDATE_POOL=3 to
# reproduce the previous cut-before-scope behavior exactly.
GENERATION_CANDIDATES_PER_GROUP = 3
CANDIDATE_POOL_PER_GROUP = max(
    GENERATION_CANDIDATES_PER_GROUP,
    int(os.getenv("RAG_ANSWER_CANDIDATE_POOL", "6")),
)


# ---------------------------------------------------------
# Domain-agnostic evidence reconstruction
# ---------------------------------------------------------

def build_answer_context(result: dict) -> dict[str, list[dict]]:
    """
    Preserve independently retrievable evidence requirements.

    Each generated subquery becomes one evidence group. No assumptions
    are made about whether a source represents an employee, product,
    project, ticket, contract, or any other domain entity.
    """

    grouped: dict[str, list[dict]] = {}

    for subquery_item in result.get("subqueries", []):
        subquery = subquery_item.get("subquery", "").strip()

        if not subquery:
            continue

        group = grouped.setdefault(subquery, [])

        for evidence in subquery_item.get("evidence", []):
            if not isinstance(evidence, dict):
                continue

            source = evidence.get("source", "")
            document = evidence.get("document", "")

            if not source or not document:
                continue

            group.append(
                {
                    "subquery": subquery,
                    "result": {
                        "document": document,
                        "metadata": {"source": source},
                        "distance": evidence.get("distance"),
                    },
                }
            )

    return grouped


def select_evidence_by_subquery(
    grouped_evidence: dict[str, list[dict]],
    max_items_per_group: int = 3,
) -> dict[str, list[dict]]:
    """
    Preserve the top retrieved evidence for every subquery.

    Retrieval already supplies ranked evidence for each subquery, so
    baseline evaluation must not replace that ranking with lexical
    heuristics or cross-group selection.

    NOTE: this is a candidate POOL, not the final generation budget. The
    consolidator applies its entity-scope firewall to the whole pool first and
    only then keeps its top ``max_candidates_per_group``. Cutting to the final
    budget here (before the firewall) lets other entities' chunks consume the
    slots that the target entity's own chunks needed.
    """

    return {
        subquery: items[:max_items_per_group]
        for subquery, items in grouped_evidence.items()
        if items
    }


def build_compact_context(
    grouped_evidence: dict[str, list[dict]],
    max_chars: int | None = None,
) -> str:
    """
    Format exactly the evidence supplied to generation.

    The judge must see the same evidence set as the generator. No
    reference-answer or generated-answer-dependent re-ranking occurs.
    """

    blocks: list[str] = []

    for group_key, group_data in grouped_evidence.items():
        # Consolidated evidence representation.
        if isinstance(group_data, dict) and "facts" in group_data:
            status = group_data.get("status", "insufficient")
            resolution_basis = group_data.get("resolution_basis") or "none"
            resolved_fact = group_data.get("resolved_fact")

            blocks.append(
                f"GROUP: {group_key}\n"
                f"STATUS: {status.upper()}\n"
                f"RESOLUTION_BASIS: {resolution_basis}"
            )

            if resolved_fact:
                blocks.append(
                    "DETERMINISTIC RESOLUTION (derived from the retrieved evidence):\n"
                    f"Subject: {resolved_fact.get('subject', '')}\n"
                    f"Attribute: {resolved_fact.get('attribute', '')}\n"
                    f"Value: {resolved_fact.get('value', '')}\n"
                    f"Statement: {resolved_fact.get('statement', '')}\n"
                    f"Temporal scope: {resolved_fact.get('temporal_scope', '')}\n"
                    f"Effective start: {resolved_fact.get('effective_start', '')}\n"
                    f"Effective end: {resolved_fact.get('effective_end', '')}\n"
                    f"Evidence indices: {resolved_fact.get('evidence_indices', [])}"
                )

            for fact in group_data.get("facts", []):
                blocks.append(
                    f"FACT: {fact['statement']}\n"
                    f"Evidence indices: {fact['evidence_indices']}"
                )

            for index, item in enumerate(group_data.get("evidence", [])):
                result = item.get("result", {})
                source = result.get("metadata", {}).get(
                    "source", "Unknown source"
                )
                filename = source.replace("\\", "/").split("/")[-1]
                document = str(result.get("document", ""))

                blocks.append(
                    f"Supporting evidence {index}: {filename}\n"
                    f"{document[:1800]}"
                )

            continue

        # Backward-compatible formatting for non-consolidated evidence.
        blocks.append(f"GROUP: {group_key}")

        for item in group_data:
            source = item["result"]["metadata"].get(
                "source", "Unknown source"
            )
            filename = source.replace("\\", "/").split("/")[-1]

            blocks.append(
                f"Subquery: {item['subquery']}\n"
                f"Source: {filename}\n"
                f"Evidence:\n"
                f"{item['result']['document']}"
            )

    context = "\n\n".join(blocks)

    if max_chars is None or len(context) <= max_chars:
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

Evaluate the generated answer primarily against the consolidated evidence
and the deterministic resolution shown below. The reference answer is a
benchmark annotation, not evidence and not an absolute authority when it
conflicts with the evidence or with a valid deterministic resolution.
Do not use outside knowledge.

Evaluation priority:
1. The consolidated evidence and resolved fact, when STATUS is RESOLVED.
2. The temporal and provenance semantics represented by that resolution.
3. The reference answer as a secondary benchmark signal only.

For current-state questions, a dated state-changing fact (for example, a
promotion, transfer, replacement, activation, or other change of state)
that has an open-ended applicability and has not been superseded should be
treated as the current state, even if an older summary or benchmark
reference still contains the prior state. Do not mark an answer incorrect
merely because the reference answer contains that older value. Conversely,
do not treat a historical fact as current when a later fact ends or
supersedes it.

If STATUS is CONTRADICTORY or INSUFFICIENT and there is no resolved fact,
do not force a single value solely to match the reference answer. An answer
that appropriately states the ambiguity or insufficiency may be more
accurate than one that blindly matches the benchmark reference.

Question:
{question}

Reference answer (benchmark annotation; not evidence):
{reference_answer}

Generated answer:
{generated_answer}

Consolidated evidence and resolution:
{evidence_context}

Score each dimension from 1 to 5:

- accuracy: factual correctness of every substantive claim, judged against
  the evidence and its valid resolution semantics. Do not penalize a
  temporally correct answer solely because the reference answer is stale or
  inconsistent.
- completeness: whether every requested part/entity/value is answered,
  using the resolved evidence where available.
- relevance: whether the answer directly addresses the question.
- groundedness: whether EVERY substantive claim is supported by the
  consolidated/retrieved evidence. A claim matching the reference answer but
  absent from evidence is NOT grounded.
- citation_correctness: whether supplied source citations/attributions
  correctly correspond to the evidence. If no citation/source attribution is
  present, do not invent one.

For numerical questions, verify arithmetic against retrieved evidence.
For multi-entity/comparative questions, verify EVERY independent
subquery/evidence group separately before accepting the final answer.
If a required group's evidence is missing and unresolved, do not award full
accuracy or groundedness for a claim that depends on that missing fact.
Do not penalize concise answers for omitting irrelevant details.

Return ONLY a JSON object matching the required schema.
Keep feedback to one short sentence.
"""

    judge_result = safe_completion_json(
        prompt,
        max_tokens=512,
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

    generation_evidence = select_evidence_by_subquery(
        grouped_evidence,
        max_items_per_group=CANDIDATE_POOL_PER_GROUP,
    )

    # Convert retrieved chunks into atomic, requirement-scoped facts
    # before generation. No new evidence is created here.
    generation_evidence = consolidate_evidence(
        generation_evidence,
        max_candidates_per_group=GENERATION_CANDIDATES_PER_GROUP,
        max_chars_per_candidate=3500,
    )


    

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
        max_chars=None,
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
    "matched_group_count": len(
        grouped_evidence
    ),
    "generation_audit": answer_result.get("audit", {}),
    "answer_metrics": judge.model_dump(),
}


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Answer-generation evaluation using saved retrieval evidence "
            "with reproducible LLM settings and cache."
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
