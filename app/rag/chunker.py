import json
import time
import os
from litellm import completion, RateLimitError
from tqdm import tqdm
from app.config import MODEL
from app.models.schemas import Document
from app.models.schemas import Result
from litellm.exceptions import BadRequestError


AVERAGE_CHUNK_SIZE = 1000
BLOCKS_PER_BATCH = 20

API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")

API_BASE = (
    f"https://api.cloudflare.com/client/v4/"
    f"accounts/{ACCOUNT_ID}/ai/v1"
)


def split_into_blocks(text: str) -> list[str]:
    """
    Split a document into meaningful structural blocks.

    Headings start new blocks.
    Numbered lists and bullet lists stay together with
    their surrounding section.
    """

    lines = text.splitlines()

    blocks = []
    current = []

    for line in lines:

        line = line.strip()

        if not line:
            if current:
                blocks.append("\n".join(current).strip())
                current = []

            continue

        # Markdown heading starts a new structural block.
        if line.startswith("#"):

            if current:
                blocks.append("\n".join(current).strip())
                current = []

            current.append(line)
            continue

        current.append(line)

    if current:
        blocks.append("\n".join(current).strip())

    return [
        block
        for block in blocks
        if block.strip()
    ]


def make_prompt(blocks: list[str]) -> str:
    """
    Ask GPT-OSS to divide structural blocks into semantic chunks.
    """

    how_many = max(
        1,
        round(
            sum(len(block) for block in blocks)
            / AVERAGE_CHUNK_SIZE
        ),
    )

    numbered_blocks = "\n\n".join(
        f"BLOCK {i}:\n{block}"
        for i, block in enumerate(blocks)
    )

    return f"""
You are a semantic document chunking system for a RAG application.

Divide the provided document blocks into approximately
{how_many} coherent semantic chunks.

Target average chunk size:
approximately {AVERAGE_CHUNK_SIZE} characters.

IMPORTANT RULES:

1. Preserve the original information.
2. Do not rewrite the document.
3. Do not summarize.
4. Do not invent information.
5. Every block must belong to exactly one chunk.
6. A chunk should contain related information.
7. Prefer larger coherent sections over many tiny chunks.
8. Keep headings with their related content.
9. Keep related numbered lists together whenever possible.
10. Keep related bullet lists together whenever possible.
11. Do not create a chunk containing only a heading.
12. Do not create a chunk containing only a bullet point.
13. Semantic coherence is more important than exact size.

Return ONLY valid JSON.

Format:

{{
    "boundaries": [0, 2, 4]
}}

A boundary indicates the START of a new chunk.

For example:

[0, 2, 4]

means:

Chunk 1 = blocks 0-1
Chunk 2 = blocks 2-3
Chunk 3 = blocks 4 onward

Document blocks:

{numbered_blocks}
"""


# def find_boundaries(blocks: list[str]) -> list[int]:

#     prompt = make_prompt(blocks)

#     max_retries = 10

#     for attempt in range(max_retries):

#         try:
#             response = completion(
#                 model=MODEL,
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": prompt,
#                     }
#                 ],
#                 response_format={"type": "json_object"},
#                 reasoning_effort="low",
#                 max_tokens=256,
#             )

#             content = response.choices[0].message.content

#             if not content:
#                 raise ValueError(
#                     "LLM returned an empty response"
#                 )

#             # print("\n--- RAW MODEL RESPONSE ---")
#             # print(repr(content))
#             # print("--- END RESPONSE ---\n")

#             data = json.loads(content)

#             boundaries = data.get(
#                 "boundaries",
#                 [0]
#             )

#             valid_boundaries = sorted(
#                 set(
#                     int(index)
#                     for index in boundaries
#                     if 0 <= int(index) < len(blocks)
#                 )
#             )

#             if 0 not in valid_boundaries:
#                 valid_boundaries.insert(0, 0)

#             return valid_boundaries

#         except RateLimitError:

#             wait_time = min(
#                 2 ** attempt,
#                 60
#             )

#             print(
#                 f"\nModel rate limit reached. "
#                 f"Waiting {wait_time}s "
#                 f"before retry "
#                 f"({attempt + 1}/{max_retries})..."
#             )

#             time.sleep(wait_time)

#         except (ValueError, json.JSONDecodeError) as e:

#             wait_time = min(
#                 2 ** attempt,
#                 30
#             )

#             print(
#                 f"\nInvalid LLM response: {e}. "
#                 f"Retrying in {wait_time}s "
#                 f"({attempt + 1}/{max_retries})..."
#             )

#             time.sleep(wait_time)

#     print(
#         "\nCould not determine semantic boundaries. "
#         "Using a single chunk for this batch."
#     )

#     return [0]

def find_boundaries(blocks: list[str]) -> list[int]:

    prompt = make_prompt(blocks)

    max_retries = 10

    for attempt in range(max_retries):

        try:
            # response = completion(
            #     model=MODEL,
            #     messages=[
            #         {
            #             "role": "user",
            #             "content": prompt,
            #         }
            #     ],
            #     response_format={"type": "json_object"},
            #     reasoning_effort="low", #not applicable for groq/compound-mini
            #     max_tokens=1024,  # <-- was 256, too small once reasoning tokens are counted
            # )


            response = completion(
                model="openai/@cf/openai/gpt-oss-120b",
                api_base=API_BASE,
                api_key=API_TOKEN,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                reasoning_effort="low",
                max_tokens=1024,
            )

            content = response.choices[0].message.content

            if not content:
                raise ValueError(
                    "LLM returned an empty response"
                )

            data = json.loads(content)

            boundaries = data.get(
                "boundaries",
                [0]
            )

            valid_boundaries = sorted(
                set(
                    int(index)
                    for index in boundaries
                    if 0 <= int(index) < len(blocks)
                )
            )

            if 0 not in valid_boundaries:
                valid_boundaries.insert(0, 0)

            return valid_boundaries

        except RateLimitError:

            wait_time = min(2 ** attempt, 60)

            print(
                f"\nModel rate limit reached. "
                f"Waiting {wait_time}s "
                f"before retry "
                f"({attempt + 1}/{max_retries})..."
            )

            time.sleep(wait_time)

        except (ValueError, json.JSONDecodeError, BadRequestError) as e:
            # ValueError/JSONDecodeError: empty or malformed content
            # BadRequestError: Groq's own "json_validate_failed" (empty
            # failed_generation) — this is what was crashing your run

            wait_time = min(2 ** attempt, 30)

            print(
                f"\nInvalid LLM response: {e}. "
                f"Retrying in {wait_time}s "
                f"({attempt + 1}/{max_retries})..."
            )

            time.sleep(wait_time)

    print(
        "\nCould not determine semantic boundaries. "
        "Using a single chunk for this batch."
    )

    return [0]

def build_chunks(
    blocks: list[str],
    boundaries: list[int],
) -> list[str]:
    """
    Reconstruct chunks using the original document blocks.
    """

    chunks = []

    for i, start in enumerate(boundaries):

        if i + 1 < len(boundaries):
            end = boundaries[i + 1]
        else:
            end = len(blocks)

        chunk = "\n\n".join(
            blocks[start:end]
        ).strip()

        if chunk:
            chunks.append(chunk)

    return chunks


def process_document(
    document: Document,
) -> list[Result]:

    blocks = split_into_blocks(document.text)

    if not blocks:
        return []

    all_boundaries = []

    for start in range(
        0,
        len(blocks),
        BLOCKS_PER_BATCH,
    ):

        batch = blocks[
            start:start + BLOCKS_PER_BATCH
        ]

        # print(
        # f"\nProcessing blocks "
        # f"{start} to {start + len(batch) - 1}"
        # )

        # print(
        #     f"Batch characters: "
        #     f"{sum(len(block) for block in batch)}"
        # )

        # if start == 20:
        #     for i, block in enumerate(batch, start=20):
        #         print(f"\n===== BLOCK {i} =====")
        #         print(block)


        boundaries = find_boundaries(batch)

        global_boundaries = [
            start + boundary
            for boundary in boundaries
        ]

        all_boundaries.extend(
            global_boundaries
        )

        # Pace requests to reduce Groq TPM pressure
        time.sleep(1)

    all_boundaries = sorted(
        set(all_boundaries)
    )

    if 0 not in all_boundaries:
        all_boundaries.insert(0, 0)

    chunks = build_chunks(
        blocks,
        all_boundaries,
    )

    metadata = {
        "source": document.source,
        "source_type": document.source_type,
        **document.metadata,
    }

    return [
        Result(
            page_content=chunk,
            metadata=metadata,
        )
        for chunk in chunks
    ]


def create_chunks(
    documents: list[Document],
) -> list[Result]:

    chunks = []

    for document in tqdm(
        documents,
        desc="Semantic chunking documents",
    ):

        document_chunks = process_document(
            document
        )

        chunks.extend(document_chunks)

    return chunks