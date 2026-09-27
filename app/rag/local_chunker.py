import re
from app.models.schemas import Document
from app.models.schemas import Result
from tqdm import tqdm


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def split_into_paragraphs(text: str) -> list[str]:
    """
    Split text into paragraphs using blank lines as boundaries.
    Works for Markdown, emails, chat messages, and plain text.
    """
    paragraphs = re.split(r"\n\s*\n", text)

    return [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]


def split_long_text(text: str, max_size: int) -> list[str]:
    """
    Split text that is too large into sentence-sized pieces.
    Falls back to words if a single sentence is too large.
    """

    sentences = re.split(r"(?<=[.!?])\s+", text)

    pieces = []
    current = ""

    for sentence in sentences:
        if len(sentence) > max_size:
            if current:
                pieces.append(current)
                current = ""

            words = sentence.split()
            word_chunk = ""

            for word in words:
                candidate = f"{word_chunk} {word}".strip()

                if len(candidate) <= max_size:
                    word_chunk = candidate
                else:
                    if word_chunk:
                        pieces.append(word_chunk)

                    word_chunk = word

            if word_chunk:
                pieces.append(word_chunk)

        else:
            candidate = f"{current} {sentence}".strip()

            if len(candidate) <= max_size:
                current = candidate
            else:
                if current:
                    pieces.append(current)

                current = sentence

    if current:
        pieces.append(current)

    return pieces


def create_chunk_texts(text: str) -> list[str]:
    """
    Create chunks while preserving natural text boundaries.
    """

    paragraphs = split_into_paragraphs(text)

    # First turn very large paragraphs into smaller pieces.
    pieces = []

    for paragraph in paragraphs:
        if len(paragraph) <= CHUNK_SIZE:
            pieces.append(paragraph)
        else:
            pieces.extend(
                split_long_text(paragraph, CHUNK_SIZE)
            )

    chunks = []
    current = ""

    for piece in pieces:

        candidate = f"{current}\n\n{piece}".strip()

        if len(candidate) <= CHUNK_SIZE:
            current = candidate
        else:
            if current:
                chunks.append(current)

            current = piece

    if current:
        chunks.append(current)

    # Add overlap between chunks.
    final_chunks = []

    for i, chunk in enumerate(chunks):

        if i == 0:
            final_chunks.append(chunk)
            continue

        previous = chunks[i - 1]

        overlap = previous[-CHUNK_OVERLAP:]

        combined = f"{overlap}\n\n{chunk}"

        final_chunks.append(combined)

    return final_chunks


def process_document(document: Document) -> list[Result]:

    chunk_texts = create_chunk_texts(document.text)

    results = []

    for chunk_text in chunk_texts:

        metadata = {
            "source": document.source,
            "source_type": document.source_type,
            **document.metadata,
        }

        results.append(
            Result(
                page_content=chunk_text,
                metadata=metadata,
            )
        )

    return results


def create_local_chunks(documents: list[Document]) -> list[Result]:

    chunks = []

    for document in tqdm(
        documents,
        desc="Chunking documents locally"
    ):
        document_chunks = process_document(document)
        chunks.extend(document_chunks)

    return chunks