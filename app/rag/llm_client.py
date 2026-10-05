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

import hashlib
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(override=True)

DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"


# =========================================================
# EVALUATION REPRODUCIBILITY
# =========================================================

_EVAL_DETERMINISTIC_ENV = "RAG_LLM_DETERMINISTIC"
_EVAL_CACHE_ENV = "RAG_LLM_CACHE"
_EVAL_CACHE_DIR_ENV = "RAG_LLM_CACHE_DIR"
# Optional global salt. Changing it invalidates every cached response without
# deleting files. Per-call namespaces (a prompt-contract version) are also
# supported. Both are omitted from the key when empty, so existing cache entries
# keep working for callers that do not use them.
_EVAL_CACHE_NAMESPACE_ENV = "RAG_LLM_CACHE_NAMESPACE"
_DEFAULT_CACHE_DIR = Path(".rag_llm_cache")


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _evaluation_mode_enabled() -> bool:
    return _env_flag(_EVAL_DETERMINISTIC_ENV)


def _cache_enabled() -> bool:
    return _evaluation_mode_enabled() and _env_flag(_EVAL_CACHE_ENV)


def _cache_dir() -> Path:
    configured = os.getenv(_EVAL_CACHE_DIR_ENV, "").strip()
    return Path(configured) if configured else _DEFAULT_CACHE_DIR


def _cache_path(
    *,
    prompt: str,
    max_tokens: int,
    json_mode: bool,
    model: str,
    namespace: str = "",
) -> Path:
    payload = {
        "model": model,
        "json_mode": json_mode,
        "max_tokens": max_tokens,
        "prompt": prompt,
        "temperature": 0.0 if _evaluation_mode_enabled() else None,
    }
    effective_namespace = "|".join(
        part
        for part in (
            os.getenv(_EVAL_CACHE_NAMESPACE_ENV, "").strip(),
            (namespace or "").strip(),
        )
        if part
    )
    if effective_namespace:
        payload["namespace"] = effective_namespace
    digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    return _cache_dir() / f"{digest}.json"


def _load_cached_result(path: Path):
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, ValueError, json.JSONDecodeError):
        return None

    if payload.get("version") != 1 or "result" not in payload:
        return None

    return payload["result"]


def _store_cached_result(path: Path, result) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    payload = {
        "version": 1,
        "result": result,
    }

    try:
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(
                payload,
                file,
                ensure_ascii=False,
                sort_keys=True,
            )
        os.replace(temp_path, path)
    except OSError:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


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
    cache_namespace: str = "",
):
    model = _get_model(json_mode)

    cache_path = None
    if _cache_enabled():
        cache_path = _cache_path(
            prompt=prompt,
            max_tokens=max_tokens,
            json_mode=json_mode,
            model=model,
            namespace=cache_namespace,
        )

        cached = _load_cached_result(cache_path)
        if cached is not None:
            print(f"[LLM CACHE HIT] {cache_path.name}")
            return cached

    client = _get_deepinfra_client()

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

            if _evaluation_mode_enabled():
                # DeepInfra documents temperature for GLM-5.3-Flash.
                # Use greedy sampling during evaluation to reduce sampling variance.
                kwargs["temperature"] = 0.0

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
                result = json.loads(content)
            else:
                result = content.strip()

            if cache_path is not None:
                _store_cached_result(cache_path, result)

            return result

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
    cache_namespace: str = "",
) -> str:

    result = _call_with_retry(
        prompt=prompt,
        max_tokens=max_tokens,
        max_retries=max_retries,
        json_mode=False,
        cache_namespace=cache_namespace,
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
    cache_namespace: str = "",
) -> dict:

    result = _call_with_retry(
        prompt=prompt,
        max_tokens=max_tokens,
        max_retries=max_retries,
        json_mode=True,
        cache_namespace=cache_namespace,
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