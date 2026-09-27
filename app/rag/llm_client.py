# app/rag/llm_client.py

# import json
# import time
# import os
# from dotenv import load_dotenv

# load_dotenv(override=True)
# =========================================================
# GROQ VERSION — KEEP FOR REFERENCE
# =========================================================

# from litellm import completion, RateLimitError
# from litellm.exceptions import BadRequestError
# from app.config import MODEL


# def _call_with_retry(
#     prompt: str,
#     max_tokens: int,
#     max_retries: int,
#     json_mode: bool,
# ):
#     for attempt in range(max_retries):

#         try:
#             kwargs = dict(
#                 model=MODEL,
#                 messages=[{"role": "user", "content": prompt}],
#                 reasoning_effort="low",
#                 max_tokens=max_tokens,
#             )

#             if json_mode:
#                 kwargs["response_format"] = {
#                     "type": "json_object"
#                 }

#             response = completion(**kwargs)

#             content = response.choices[0].message.content

#             if not content:
#                 raise ValueError(
#                     "LLM returned an empty response"
#                 )

#             return (
#                 json.loads(content)
#                 if json_mode
#                 else content.strip()
#             )

#         except RateLimitError:
#             wait_time = min(2 ** attempt, 60)

#             print(
#                 f"\nRate limit reached. "
#                 f"Waiting {wait_time}s "
#                 f"({attempt + 1}/{max_retries})..."
#             )

#             time.sleep(wait_time)

#         except (
#             ValueError,
#             json.JSONDecodeError,
#             BadRequestError,
#         ) as e:

#             wait_time = min(2 ** attempt, 30)

#             print(
#                 f"\nInvalid LLM response: {e}. "
#                 f"Retrying in {wait_time}s "
#                 f"({attempt + 1}/{max_retries})..."
#             )

#             time.sleep(wait_time)

#     return None


# =========================================================
# CLOUDFLARE VERSION — KEEP FOR REFERENCE
# =========================================================

# from openai import OpenAI


# def _get_cloudflare_client() -> OpenAI:
#     api_token = os.getenv("CLOUDFLARE_API_TOKEN")
#     account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")

#     if not api_token:
#         raise RuntimeError(
#             "CLOUDFLARE_API_TOKEN is not set."
#         )

#     if not account_id:
#         raise RuntimeError(
#             "CLOUDFLARE_ACCOUNT_ID is not set."
#         )

#     return OpenAI(
#         api_key=api_token,
#         base_url=(
#             "https://api.cloudflare.com/client/v4/"
#             f"accounts/{account_id}/ai/v1"
#         ),
#     )


# app/rag/llm_client.py

# app/rag/llm_client.py

import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(override=True)

DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"


# =========================================================
# CLIENT
# =========================================================

def _get_deepinfra_client() -> OpenAI:
    api_key = os.getenv("DEEPINFRA_API_KEY")

    if not api_key:
        raise RuntimeError(
            "DEEPINFRA_API_KEY is not set."
        )

    return OpenAI(
        api_key=api_key,
        base_url=DEEPINFRA_BASE_URL,
    )


# =========================================================
# MODEL
# =========================================================

def _get_model(json_mode: bool) -> str:
    """
    Normal generation -> MODEL
    Judge / structured generation -> VERIFIER_MODEL
    """

    if json_mode:
        model = (
            os.getenv("VERIFIER_MODEL")
            or os.getenv("MODEL")
        )
    else:
        model = (
            os.getenv("MODEL")
            or os.getenv("VERIFIER_MODEL")
        )

    if not model:
        raise RuntimeError(
            "MODEL or VERIFIER_MODEL is not set."
        )

    return model


# =========================================================
# CORE CALL
# =========================================================

def _call_with_retry(
    prompt: str,
    max_tokens: int,
    max_retries: int,
    json_mode: bool,
):
    client = _get_deepinfra_client()
    model = _get_model(json_mode)

    for attempt in range(max_retries):

        try:

            kwargs = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "max_tokens": max_tokens,
            }

            if json_mode:
                kwargs["response_format"] = {
                    "type": "json_object"
                }

            response = client.chat.completions.create(
                **kwargs
            )

            choice = response.choices[0]

            content = choice.message.content
            finish_reason = choice.finish_reason

            if not content:
                raise ValueError(
                    "DeepInfra returned an empty response "
                    f"(finish_reason={finish_reason})"
                )

            if finish_reason == "length":
                raise ValueError(
                    "DeepInfra response was truncated "
                    f"(max_tokens={max_tokens})"
                )

            if json_mode:
                return json.loads(content)

            return content.strip()

        except (
            ValueError,
            json.JSONDecodeError,
        ) as error:

            wait_time = min(
                2 ** attempt,
                10,
            )

            print(
                f"\nInvalid DeepInfra response: {error}. "
                f"Retrying in {wait_time}s "
                f"({attempt + 1}/{max_retries})..."
            )

            if attempt < max_retries - 1:
                time.sleep(wait_time)

        except Exception as error:

            print(
                f"\nDeepInfra LLM error: {error}"
            )

            # DeepInfra/API errors are normally worth retrying
            # briefly, but don't hammer the provider.
            if attempt < max_retries - 1:
                wait_time = min(
                    2 ** attempt,
                    10,
                )

                print(
                    f"Retrying in {wait_time}s "
                    f"({attempt + 1}/{max_retries})..."
                )

                time.sleep(wait_time)

    return None


# =========================================================
# TEXT
# =========================================================

def safe_completion_text(
    prompt: str,
    max_tokens: int = 256,
    max_retries: int = 3,
    fallback: str = (
        "I could not generate an answer "
        "due to a temporary service issue."
    ),
) -> str:

    result = _call_with_retry(
        prompt=prompt,
        max_tokens=max_tokens,
        max_retries=max_retries,
        json_mode=False,
    )

    if result is None:
        print(
            "\nGiving up after max retries "
            "(DeepInfra text call)."
        )
        return fallback

    return result


# =========================================================
# JSON
# =========================================================

def safe_completion_json(
    prompt: str,
    max_tokens: int = 256,
    max_retries: int = 3,
    fallback: dict | None = None,
) -> dict:

    result = _call_with_retry(
        prompt=prompt,
        max_tokens=max_tokens,
        max_retries=max_retries,
        json_mode=True,
    )

    if result is None:
        print(
            "\nGiving up after max retries "
            "(DeepInfra JSON call)."
        )

        return (
            fallback
            if fallback is not None
            else {}
        )

    return result