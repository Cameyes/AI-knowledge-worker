import json
import time

from litellm import completion, RateLimitError
from litellm.exceptions import BadRequestError
from app.rag.verification_context import prepare_verification_context

from app.config import (
    MODEL,
    VERIFIER_MODEL,
    CLOUDFLARE_API_TOKEN,
    CLOUDFLARE_ACCOUNT_ID,
)


def _build_prompt(
    query: str,
    subquery: str,
    evidence_items: list[dict],
) -> str:

    evidence_json = json.dumps(
        evidence_items,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
You are an evidence verification system for a generic
retrieval-augmented generation (RAG) application.

Original user query:
{query}

Subquery (shared by all evidence items below):
{subquery}

Evidence items:
{evidence_json}

Step 1: Identify the target entity.
If the subquery names a specific person, place, product, or
other named entity, state that entity once as "target_entity".
If the subquery names no specific entity, set "target_entity"
to null.

Step 2: For EACH evidence item, in order, determine:
- The entity that the evidence item is about (or null if none).
- Use the source field together with the evidence text to identify
  the evidence entity.
- The source field is authoritative provenance for identifying the
  document's subject.
- Do not require the target entity's name to appear verbatim inside
  the evidence text.
- Whether that entity matches "target_entity" exactly. If
  "target_entity" is null, treat this as true and rely on the
  checks below instead.
- Whether the item is relevant: relevant requires the entity match
  AND that the evidence contains the specific fact, relationship,
  or constraint (time period, category, status, quantity, etc.)
  the subquery asks for.

Do not infer facts absent from the evidence. Do not compare
evidence items against each other. Do not favor an item for
being more similar in wording to the subquery.

Return ONLY a JSON object in this EXACT compact form — one
array per evidence item, in the same order, with no items
skipped:

{{
  "target_entity": "<name or null>",
  "evaluations": [
    ["<id>", "<evidence_entity or null>", <entity_match bool>, <relevant bool>]
  ]
}}

The evaluations array MUST contain exactly one entry per
evidence item shown above.
"""



def _verify_batch(
    query: str,
    subquery: str,
    evidence_items: list[dict],
    max_retries: int = 8,
) -> list[list]:
    """
    Run one verification call over a batch of evidence items that
    all share the same subquery, and return the raw evaluations
    (as a list of [id, evidence_entity, entity_match, relevant]).
    """

    prompt = _build_prompt(query, subquery, evidence_items)

    for attempt in range(max_retries):

        try:
            #use the below code for cloudflare model
#             response = completion(
#     model=VERIFIER_MODEL,
#     api_base=(
#         f"https://api.cloudflare.com/client/v4/"
#         f"accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1"
#     ),
#     api_key=CLOUDFLARE_API_TOKEN,
#     messages=[
#         {
#             "role": "user",
#             "content": prompt,
#         }
#     ],
#     response_format={"type": "json_object"},
#     reasoning_effort="low",
#     max_tokens=768,
# )
            #use the below code for groq model
            response = completion(
                 model=f"deepinfra/{MODEL}",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                response_format={"type": "json_object"},
               # reasoning_effort="low",
                max_tokens=768,
            )

            usage = getattr(response, "usage", None)

            if usage:
                print("\n===== VERIFIER USAGE =====")
                print("Prompt tokens:     ", getattr(usage, "prompt_tokens", "N/A"))
                print("Completion tokens: ", getattr(usage, "completion_tokens", "N/A"))
                print("Total tokens:      ", getattr(usage, "total_tokens", "N/A"))
                print("Neurons:           ", getattr(usage, "neurons", "N/A"))

            content = response.choices[0].message.content

            if not content:
                raise ValueError("LLM returned an empty response")

            data = json.loads(content) 

            print("\n===== VERIFIER RAW OUTPUT =====")
            print(json.dumps(data, indent=2))

            evaluations = data.get("evaluations", [])

            if len(evaluations) != len(evidence_items):
                # Partial output — likely truncated mid-generation.
                # Treat as invalid and retry rather than silently
                # dropping the missing items.
                raise ValueError(
                    f"Expected {len(evidence_items)} evaluations, "
                    f"got {len(evaluations)} — response may be "
                    f"truncated."
                )

            return evaluations

        except RateLimitError:

            wait_time = min(2 ** attempt, 60)

            print(
                f"\nModel rate limit reached. "
                f"Waiting {wait_time}s before retry "
                f"({attempt + 1}/{max_retries})..."
            )

            time.sleep(wait_time)

        except (ValueError, json.JSONDecodeError, BadRequestError) as e:

            wait_time = min(2 ** attempt, 30)

            print(
                f"\nInvalid LLM response: {e}. "
                f"Retrying in {wait_time}s "
                f"({attempt + 1}/{max_retries})..."
            )

            time.sleep(wait_time)

    print(
        "\nCould not verify this batch after max retries. "
        "Treating its items as unverified."
    )

    return []


def verify_evidence(
    query: str,
    evidence: list[dict],
) -> list[dict]:
    """
    Verify which retrieved evidence items directly support
    their corresponding subqueries.

    Evidence is grouped by subquery. Each subquery's retrieved
    results are semantically compacted before being sent to the
    verifier, reducing the amount of text sent to the LLM.

    The verifier is domain-agnostic.
    """

    if not evidence:
        return []

    groups: dict[str, list[tuple[int, dict]]] = {}

    for index, item in enumerate(evidence):
        groups.setdefault(
            item["subquery"],
            [],
        ).append(
            (index, item)
        )

    verified_evidence = []

    for subquery, indexed_items in groups.items():

        local_to_global = [
            global_index
            for global_index, _ in indexed_items
        ]

        results = [
            item["result"]
            for _, item in indexed_items
        ]

        compact_context = prepare_verification_context(
            query=subquery,
            results=results,
        )

        evidence_items = [
            {
                "id": str(index),
                "source": item["source"],
                "evidence": item["evidence"],
            }
            for index, item in enumerate(
                compact_context
            )
        ]

        print("\n===== COMPACTED VERIFICATION CONTEXT =====")

        for item in evidence_items:
            print(f"\nID: {item['id']}")
            print(f"Source: {item['source']}")
            print(f"Evidence:\n{item['evidence']}")

        evaluations = _verify_batch(
            query=query,
            subquery=subquery,
            evidence_items=evidence_items,
        )

        for evaluation in evaluations:

            if len(evaluation) < 4:
                continue

            (
                local_id,
                _evidence_entity,
                _entity_match,
                is_relevant,
            ) = evaluation

            if (
                is_relevant
                and str(local_id).isdigit()
            ):

                local_index = int(local_id)

                if 0 <= local_index < len(local_to_global):

                    global_index = (
                        local_to_global[local_index]
                    )

                    verified_evidence.append(
                        evidence[global_index]
                    )

    return verified_evidence