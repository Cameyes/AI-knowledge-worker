# import re

# from app.rag.reranker import reranker

# def _is_context_fragment(
#     text: str,
#     max_chars: int = 80,
# ) -> bool:
#     """
#     Identify short structural/context fragments without assuming
#     any particular document format.

#     Examples:
#         "David Kim"
#         "Compensation History"
#         "Project Overview"
#         "Status: Approved"

#     These fragments are preserved and attached to nearby content
#     rather than discarded.
#     """
#     text = " ".join(text.split())

#     if not text:
#         return True

#     if len(text) > max_chars:
#         return False

#     # A complete sentence is usually substantive enough to stand alone.
#     if re.search(r"[.!?]$", text):
#         return False

#     return True

# def _split_units(
#     document: str,
#     max_unit_chars: int = 700,
# ) -> list[str]:
#     """
#     Split document text into context-aware natural blocks.

#     Short structural/context fragments are attached to the
#     following substantive block so they are not selected as
#     standalone evidence.

#     Oversized blocks are split into smaller sentence-based units.

#     No domain-specific or format-specific assumptions are made.
#     """

#     document = document.strip()

#     if not document:
#         return []

#     blocks = re.split(
#         r"\n\s*\n+",
#         document,
#     )

#     # ---------------------------------------------------------
#     # 1. Attach short context fragments to nearby content.
#     # ---------------------------------------------------------

#     grouped_blocks = []
#     pending_fragments = []

#     for block in blocks:
#         block = block.strip()

#         if not block:
#             continue

#         if _is_context_fragment(block):
#             pending_fragments.append(block)
#             continue

#         if pending_fragments:
#             block = "\n\n".join(
#                 pending_fragments + [block]
#             )
#             pending_fragments = []

#         grouped_blocks.append(block)

#     # Preserve trailing fragments instead of discarding them.
#     if pending_fragments:
#         if grouped_blocks:
#             grouped_blocks[-1] = "\n\n".join(
#                 [grouped_blocks[-1]]
#                 + pending_fragments
#             )
#         else:
#             grouped_blocks.extend(
#                 pending_fragments
#             )

#     # ---------------------------------------------------------
#     # 2. Split oversized grouped blocks.
#     # ---------------------------------------------------------

#     units = []

#     for block in grouped_blocks:

#         if len(block) <= max_unit_chars:
#             units.append(block)
#             continue

#         pieces = re.split(
#             r"(?<=[.!?])\s+",
#             block,
#         )

#         current = ""

#         for piece in pieces:
#             piece = piece.strip()

#             if not piece:
#                 continue

#             candidate = (
#                 f"{current} {piece}".strip()
#                 if current
#                 else piece
#             )

#             if len(candidate) <= max_unit_chars:
#                 current = candidate
#                 continue

#             if current:
#                 units.append(current)

#             if len(piece) > max_unit_chars:
#                 for start in range(
#                     0,
#                     len(piece),
#                     max_unit_chars,
#                 ):
#                     units.append(
#                         piece[
#                             start:start + max_unit_chars
#                         ]
#                     )

#                 current = ""
#             else:
#                 current = piece

#         if current:
#             units.append(current)

#     return units


# def _score_units(
#     query: str,
#     units: list[str],
# ) -> list[tuple[float, int, str]]:
#     """
#     Score complete text units using the local cross-encoder.

#     Returns:
#         (score, original_index, text)
#     """

#     if not units:
#         return []

#     pairs = [
#         (query, unit)
#         for unit in units
#     ]

#     scores = reranker.predict(
#         pairs,
#         batch_size=32,
#         show_progress_bar=False,
#     )

#     return [
#         (
#             float(score),
#             index,
#             unit,
#         )
#         for index, (score, unit) in enumerate(
#             zip(scores, units)
#         )
#     ]


# def _compact_scored_units(
#     scored_units: list[tuple[float, int, str]],
#     max_units: int,
#     max_chars: int,
# ) -> str:
#     """
#     Select the most relevant scored units and reconstruct them
#     in original document order.
#     """

#     if not scored_units:
#         return ""

#     ranked = sorted(
#         scored_units,
#         key=lambda item: item[0],
#         reverse=True,
#     )

#     selected = ranked[:max_units]

#     selected.sort(
#         key=lambda item: item[1]
#     )

#     output = []
#     current_chars = 0

#     for _, _, unit in selected:

#         separator = 2 if output else 0

#         remaining = (
#             max_chars
#             - current_chars
#             - separator
#         )

#         if remaining <= 0:
#             break

#         if len(unit) <= remaining:

#             output.append(unit)

#             current_chars += (
#                 separator
#                 + len(unit)
#             )

#         else:

#             output.append(
#                 unit[:remaining]
#             )
#             break

#     return "\n\n".join(output)


# def compact_evidence(
#     query: str,
#     document: str,
#     max_units: int = 1,
#     max_unit_chars: int = 700,
#     max_chars: int = 800,
# ) -> str:
#     """
#     Extract the most semantically relevant original text
#     from a single retrieved document.

#     No LLM is used.
#     No rewriting is performed.
#     No domain-specific rules are used.
#     """

#     if not document.strip():
#         return ""

#     units = _split_units(
#         document=document,
#         max_unit_chars=max_unit_chars,
#     )

#     if not units:
#         return ""

#     scored_units = _score_units(
#         query=query,
#         units=units,
#     )

#     if not scored_units:
#         return document[:max_chars]

#     compacted = _compact_scored_units(
#         scored_units=scored_units,
#         max_units=max_units,
#         max_chars=max_chars,
#     )

#     return compacted or document[:max_chars]


# def prepare_verification_context(
#     query: str,
#     results: list[dict],
#     max_units: int = 1,
#     max_unit_chars: int = 700,
#     max_chars: int = 800,
# ) -> list[dict]:
#     """
#     Prepare compact semantic evidence for multiple retrieved
#     results using ONE batched cross-encoder inference call.

#     Original retrieval results are not modified.

#     Each output item contains:

#         result_index
#         source
#         evidence
#     """

#     if not results:
#         return []

#     # ---------------------------------------------------------
#     # 1. Split every retrieved document into natural units.
#     # ---------------------------------------------------------

#     result_units = {}

#     all_pairs = []

#     pair_metadata = []

#     for result_index, result in enumerate(results):

#         document = result.get(
#             "document",
#             "",
#         )

#         metadata = result.get(
#             "metadata",
#             {},
#         )

#         source = metadata.get(
#             "source",
#             "Unknown source",
#         )

#         units = _split_units(
#             document=document,
#             max_unit_chars=max_unit_chars,
#         )

#         result_units[result_index] = units

#         for unit_index, unit in enumerate(units):

#             all_pairs.append(
#                 (
#                     query,
#                     f"Source: {source}\nEvidence:\n{unit}",
#                 )
#             )

#             pair_metadata.append(
#                 (
#                     result_index,
#                     unit_index,
#                     unit,
#                 )
#             )
#     # ---------------------------------------------------------
#     # 2. Score ALL units in one cross-encoder call.
#     # ---------------------------------------------------------

#     if all_pairs:

#         scores = reranker.predict(
#             all_pairs,
#             batch_size=32,
#             show_progress_bar=False,
#         )

#     else:
#         scores = []

#     # ---------------------------------------------------------
#     # 3. Put scores back into their original result groups.
#     # ---------------------------------------------------------

#     scored_by_result = {
#         index: []
#         for index in range(len(results))
#     }

#     for score, (
#         result_index,
#         unit_index,
#         unit,
#     ) in zip(scores, pair_metadata):

#         scored_by_result[result_index].append(
#             (
#                 float(score),
#                 unit_index,
#                 unit,
#             )
#         )

#     # ---------------------------------------------------------
#     # 4. Compact each result independently.
#     # ---------------------------------------------------------

#     verification_context = []

#     for result_index, result in enumerate(results):

#         metadata = result.get(
#             "metadata",
#             {},
#         )

#         source = metadata.get(
#             "source",
#             "Unknown source",
#         )

#         document = result.get(
#             "document",
#             "",
#         )

#         scored_units = scored_by_result[
#             result_index
#         ]

#         if scored_units:

#             compacted = _compact_scored_units(
#                 scored_units=scored_units,
#                 max_units=max_units,
#                 max_chars=max_chars,
#             )

#         else:

#             compacted = document[:max_chars]

#         verification_context.append(
#             {
#                 "result_index": result_index,
#                 "source": source,
#                 "evidence": compacted,
#             }
#         )

#     return verification_context

import re

from app.rag.reranker import reranker


DEFAULT_MAX_UNITS = 3
DEFAULT_MAX_UNIT_CHARS = 700
DEFAULT_MAX_CHARS = 1000


def _is_context_fragment(
    text: str,
    max_chars: int = 80,
) -> bool:
    """
    Identify short structural/context fragments without assuming
    any particular document format.

    Examples:
        "David Kim"
        "Compensation History"
        "Project Overview"
        "Status: Approved"

    These fragments are not returned as standalone units. Instead
    they're carried forward as scoring context for the substantive
    blocks that follow them (see _split_units).
    """
    text = " ".join(text.split())

    if not text:
        return True

    if len(text) > max_chars:
        return False

    # A complete sentence is usually substantive enough to stand alone.
    if re.search(r"[.!?]$", text):
        return False

    return True


def _split_units(
    document: str,
    max_unit_chars: int = DEFAULT_MAX_UNIT_CHARS,
) -> list[dict]:
    """
    Split document text into natural blocks, each returned as:

        {"context": str | None, "text": str}

    `text` is the original block content, plus any fragment(s)
    that immediately precede it (merged verbatim, as a heading or
    short fact directly preceding a block always was) — so no
    factual content is ever silently dropped from the output.
    `context` is that same fragment (or the most recent one seen),
    carried forward for SCORING purposes on every SUBSEQUENT
    sibling block too, without being re-merged into their text.

    This matters for any document where one heading precedes
    several blank-line-separated sibling entries (numbered clauses,
    dated entries, bulleted sub-items, etc.) — a pattern common
    across contracts, policies, reports, and structured records
    generally. Previously, only the first sibling after a heading
    retained it as context; later siblings were scored with no
    awareness of what section they belonged to.

    No domain-specific or format-specific assumptions are made.
    """

    document = document.strip()

    if not document:
        return []

    blocks = re.split(
        r"\n\s*\n+",
        document,
    )

    # ---------------------------------------------------------
    # 1. Group blocks with persistent heading/context tracking.
    # ---------------------------------------------------------

    grouped_blocks = []

    pending_fragments = []
    current_context = None

    for block in blocks:
        block = block.strip()

        if not block:
            continue

        if _is_context_fragment(block):
            pending_fragments.append(block)
            continue

        if pending_fragments:
            # Fragments immediately preceding this block are
            # merged directly into its returned text — this keeps
            # adjacent label/fact + description pairs (e.g. a
            # short fact line followed by its explanatory sentence)
            # verbatim in the output, exactly as before.
            fragment_text = "\n".join(pending_fragments)
            text = "\n\n".join(pending_fragments + [block])
            current_context = fragment_text
            pending_fragments = []
        else:
            text = block

        grouped_blocks.append(
            {
                # `context` persists across every subsequent
                # sibling block (not just this one) until a new
                # fragment sequence replaces it — used for SCORING
                # only, never merged into `text` again, so later
                # siblings' output stays exactly their own content.
                "context": current_context,
                "text": text,
            }
        )

    # Trailing fragments with nothing substantive after them are
    # still preserved as their own low-priority unit, rather than
    # discarded — they may occasionally be the answer themselves
    # (e.g. a standalone "Status: Approved" line).
    if pending_fragments:
        trailing_text = "\n".join(pending_fragments)
        grouped_blocks.append(
            {
                "context": current_context,
                "text": trailing_text,
            }
        )

    # ---------------------------------------------------------
    # 2. Split oversized grouped blocks, carrying context along.
    # ---------------------------------------------------------

    units = []

    for grouped in grouped_blocks:

        context = grouped["context"]
        block = grouped["text"]

        if len(block) <= max_unit_chars:
            units.append({"context": context, "text": block})
            continue

        pieces = re.split(
            r"(?<=[.!?])\s+",
            block,
        )

        current = ""

        def _flush(piece_text: str) -> None:
            if piece_text:
                units.append(
                    {"context": context, "text": piece_text}
                )

        for piece in pieces:
            piece = piece.strip()

            if not piece:
                continue

            candidate = (
                f"{current} {piece}".strip()
                if current
                else piece
            )

            if len(candidate) <= max_unit_chars:
                current = candidate
                continue

            _flush(current)

            if len(piece) > max_unit_chars:
                for start in range(
                    0,
                    len(piece),
                    max_unit_chars,
                ):
                    _flush(piece[start:start + max_unit_chars])

                current = ""
            else:
                current = piece

        _flush(current)

    return units


def _scoring_text(unit: dict) -> str:
    """
    Build the text actually shown to the cross-encoder for scoring
    a unit: heading/context prefixed onto the unit's own text, if
    context is available. This is scoring-only — never used for
    reconstruction/output.
    """

    context = unit.get("context")
    text = unit["text"]

    if context:
        return f"{context}\n{text}"

    return text


def _score_units(
    query: str,
    units: list[dict],
) -> list[tuple[float, int, dict]]:
    """
    Score units using the local cross-encoder, with each unit's
    heading/context included in the scoring input (not the unit
    text itself, which stays unmodified for reconstruction).

    Returns:
        (score, original_index, unit)
    """

    if not units:
        return []

    pairs = [
        (query, _scoring_text(unit))
        for unit in units
    ]

    scores = reranker.predict(
        pairs,
        batch_size=32,
        show_progress_bar=False,
    )

    return [
        (float(score), index, unit)
        for index, (score, unit) in enumerate(zip(scores, units))
    ]


def _compact_scored_units(
    scored_units: list[tuple[float, int, dict]],
    max_units: int,
    max_chars: int,
) -> str:
    """
    Select the top-k scored units (by cross-encoder score) and
    reconstruct them in original document order, using each
    unit's raw `text` only — never its scoring `context`.
    """

    if not scored_units:
        return ""

    ranked = sorted(
        scored_units,
        key=lambda item: item[0],
        reverse=True,
    )

    selected = ranked[:max_units]

    # Restore original document order among the selected units.
    selected.sort(key=lambda item: item[1])

    output = []
    current_chars = 0

    for _, _, unit in selected:

        text = unit["text"]

        separator = 2 if output else 0
        remaining = max_chars - current_chars - separator

        if remaining <= 0:
            break

        if len(text) <= remaining:
            output.append(text)
            current_chars += separator + len(text)
        else:
            output.append(text[:remaining])
            break

    return "\n\n".join(output)


def compact_evidence(
    query: str,
    document: str,
    max_units: int = DEFAULT_MAX_UNITS,
    max_unit_chars: int = DEFAULT_MAX_UNIT_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """
    Extract the most semantically relevant original text unit(s)
    from a single retrieved document.

    No LLM is used. No rewriting is performed. No domain-specific
    rules are used. Retaining up to `max_units` units (instead of
    exactly 1) hedges against cross-encoder ranking error — the
    correct fact-bearing unit does not always score strictly
    highest, especially for terse/tabular content, so a small
    candidate set is passed to the downstream LLM verifier instead
    of a single irreversible guess.
    """

    if not document.strip():
        return ""

    units = _split_units(
        document=document,
        max_unit_chars=max_unit_chars,
    )

    if not units:
        return ""

    scored_units = _score_units(
        query=query,
        units=units,
    )

    if not scored_units:
        return document[:max_chars]

    compacted = _compact_scored_units(
        scored_units=scored_units,
        max_units=max_units,
        max_chars=max_chars,
    )

    return compacted or document[:max_chars]


def prepare_verification_context(
    query: str,
    results: list[dict],
    max_units: int = DEFAULT_MAX_UNITS,
    max_unit_chars: int = DEFAULT_MAX_UNIT_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[dict]:
    """
    Prepare compact semantic evidence for multiple retrieved
    results using ONE batched cross-encoder inference call.

    Original retrieval results are not modified.

    Note: unlike an earlier version of this function, the source
    filename is NOT included in the cross-encoder scoring input.
    Document-level provenance biases the scorer toward selecting
    ANY block from the correct document rather than the specific
    block containing the requested fact — the opposite of what
    unit selection needs. Section-level heading context (see
    _split_units) is used instead, since it operates at the right
    granularity: distinguishing between siblings within the same
    document, not between documents.

    Each output item contains:

        result_index
        source
        evidence
    """

    if not results:
        return []

    # ---------------------------------------------------------
    # 1. Split every retrieved document into natural units.
    # ---------------------------------------------------------

    result_units: dict[int, list[dict]] = {}

    all_pairs = []
    pair_metadata = []

    for result_index, result in enumerate(results):

        document = result.get("document", "")

        units = _split_units(
            document=document,
            max_unit_chars=max_unit_chars,
        )

        result_units[result_index] = units

        for unit_index, unit in enumerate(units):

            all_pairs.append((query, _scoring_text(unit)))

            pair_metadata.append(
                (result_index, unit_index, unit)
            )

    # ---------------------------------------------------------
    # 2. Score ALL units in one cross-encoder call.
    # ---------------------------------------------------------

    if all_pairs:
        scores = reranker.predict(
            all_pairs,
            batch_size=32,
            show_progress_bar=False,
        )
    else:
        scores = []

    # ---------------------------------------------------------
    # 3. Put scores back into their original result groups.
    # ---------------------------------------------------------

    scored_by_result: dict[int, list[tuple[float, int, dict]]] = {
        index: [] for index in range(len(results))
    }

    for score, (result_index, unit_index, unit) in zip(
        scores, pair_metadata
    ):
        scored_by_result[result_index].append(
            (float(score), unit_index, unit)
        )

    # ---------------------------------------------------------
    # 4. Compact each result independently, retaining top-k units.
    # ---------------------------------------------------------

    verification_context = []

    for result_index, result in enumerate(results):

        metadata = result.get("metadata", {})
        source = metadata.get("source", "Unknown source")
        document = result.get("document", "")

        scored_units = scored_by_result[result_index]

        if scored_units:
            compacted = _compact_scored_units(
                scored_units=scored_units,
                max_units=max_units,
                max_chars=max_chars,
            )
        else:
            compacted = document[:max_chars]

        verification_context.append(
            {
                "result_index": result_index,
                "source": source,
                "evidence": compacted,
            }
        )

    return verification_context