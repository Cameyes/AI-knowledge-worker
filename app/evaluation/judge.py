import os

import httpx
from dotenv import load_dotenv

from app.evaluation.schemas import EvidenceJudgment


# Loads environment variables from the project's .env file.
load_dotenv(override=True)


# Pinned Jev model for reproducible evaluation runs.
JEV_MODEL = "typesafe/jev-1.13"

# OpenRouter Decisions API endpoint.
OPENROUTER_DECISIONS_URL = (
    "https://openrouter.ai/api/alpha/decisions"
)

# Probability threshold used to convert Jev's probability into a boolean judgment.
DEFAULT_THRESHOLD = 0.5

# Maximum time allowed for one Jev request.
REQUEST_TIMEOUT = 60.0


def judge_subquery_evidence(
    subquery: str,
    retrieved_results: list[dict],
    threshold: float = DEFAULT_THRESHOLD,
) -> list[EvidenceJudgment]:
    """
    Evaluate whether each retrieved chunk directly supports a subquery.
    """

    if not retrieved_results:
        return []

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured."
        )

    # Build explicitly identified records for Jev.
    records = []

    for rank, result in enumerate(
        retrieved_results,
        start=1,
    ):
        metadata = result.get(
            "metadata",
            {},
        ) or {}

        records.append(
            {
                "id": f"p{rank}",
                "record": {
                    "rank": rank,
                    "source": metadata.get(
                        "source",
                        "",
                    ),
                    "text": result.get(
                        "document",
                        "",
                    ),
                },
            }
        )

    state = {
        "description": (
            "A retrieval query and ranked passages. "
            "Each record contains exactly one retrieved passage. "
            "Judge only the record explicitly named in each question."
        ),
        "query": subquery,
        "records": records,
    }

    # Create one answer-support decision for each retrieved record.
    questions = {}

    for rank in range(
        1,
        len(retrieved_results) + 1,
    ):
        questions[f"p{rank}_support"] = {
            "type": "noul",
            "instructions": (
                (
    f'For the record with id "p{rank}", determine whether '
    "the information in `record.text` alone is sufficient to "
    "answer `query` correctly. "
    "The record must contain the requested fact or the exact "
    "evidence needed to derive it. "
    "Do not use information from other records. "
    "Do not infer the requested fact merely because the record "
    "belongs to the same person, topic, or source. "
    "If the requested fact is absent from `record.text`, answer NO."
)
            ),
        }

    payload = {
        "model": JEV_MODEL,
        "state": state,
        "questions": questions,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(
            timeout=REQUEST_TIMEOUT
        ) as client:
            response = client.post(
                OPENROUTER_DECISIONS_URL,
                headers=headers,
                json=payload,
            )

        response.raise_for_status()

    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            "OpenRouter Jev request failed: "
            f"{exc.response.status_code} "
            f"{exc.response.text}"
        ) from exc

    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"OpenRouter Jev request failed: {exc}"
        ) from exc

    data = response.json()

    answers = data.get(
        "answers",
        {},
    )

    # Validate that Jev returned every requested decision.
    missing_answers = [
        question_id
        for question_id in questions
        if question_id not in answers
    ]

    if missing_answers:
        raise RuntimeError(
            "Jev response is missing expected answers: "
            f"{missing_answers}"
        )

    judgments = []

    for rank, result in enumerate(
        retrieved_results,
        start=1,
    ):
        support_probability = float(
            answers[
                f"p{rank}_support"
            ]["noul"]
        )

        metadata = result.get(
            "metadata",
            {},
        ) or {}

        judgments.append(
            EvidenceJudgment(
                rank=rank,
                source=metadata.get(
                    "source",
                    "",
                ),
                support_probability=support_probability,
                supports_answer=(
                    support_probability >= threshold
                ),
            )
        )

    return judgments