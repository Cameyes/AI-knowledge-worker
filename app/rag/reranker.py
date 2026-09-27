from sentence_transformers import CrossEncoder


# MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# MODEL_NAME = "BAAI/bge-reranker-v2-m3"
# MODEL_NAME = "cross-encoder/ettin-reranker-17m-v1"
MODEL_NAME = "cross-encoder/ettin-reranker-32m-v1"


# Pinned explicitly rather than left to the model's implicit
# default, so truncation behavior is documented and predictable.
MAX_LENGTH = 512
RRF_K = 60
RETRIEVAL_RRF_WEIGHT = 0.6
RERANK_RRF_WEIGHT = 0.4

# A candidate just below the normal top-k cutoff can be rescued
# when it belongs to a source that is already strongly represented.
#
# This is deliberately small so we do not blindly introduce
# low-scoring candidates into the final evidence set.
SOURCE_RESCUE_MARGIN = 0.10


# At most one additional chunk from an already represented source
# can be rescued during this selection step.
#
# This prevents one source from dominating the final evidence set
# while still allowing a second chunk to survive when necessary.
MAX_RESCUES_PER_SOURCE = 1


reranker = CrossEncoder(
    MODEL_NAME,
    max_length=MAX_LENGTH,
)


def _warn_if_likely_truncated(
    query: str,
    document: str,
) -> None:
    """
    Best-effort truncation warning.

    Uses the cross-encoder's own tokenizer so the check matches
    what predict() will actually receive.

    This never changes scoring behavior. It only makes possible
    truncation visible to the developer.
    """

    try:
        tokenizer = reranker.tokenizer

        token_count = len(
            tokenizer.encode(
                query,
                document,
                truncation=False,
            )
        )

    except Exception:
        # Diagnostic failure must never break reranking.
        return

    if token_count > MAX_LENGTH:
        print(
            f"\n[reranker] WARNING: query+document pair is "
            f"{token_count} tokens, exceeding max_length={MAX_LENGTH}. "
            f"This document will be scored on a TRUNCATED view — "
            f"content near the end of the chunk is invisible to "
            f"reranking. Consider re-chunking this source into "
            f"smaller pieces."
        )


def build_rerank_context(
    result: dict,
) -> str:
    """
    Build the text passed to the cross-encoder.

    The source is retained as provenance/context, while the
    document itself remains unchanged.
    """

    document = result["document"]
    metadata = result.get("metadata", {})

    source = metadata.get("source", "")

    if source:
        return (
            f"Source: {source}\n\n"
            f"Document:\n{document}"
        )

    return document


def _get_source(
    result: dict,
) -> str | None:
    """
    Extract a generic source identifier from a retrieval result.

    No assumptions are made about the source format.

    Examples could be:
        file path
        URL
        document ID
        Slack message/thread ID
        Gmail thread ID
        SharePoint document ID

    The value is used only for grouping related chunks.
    """

    metadata = result.get("metadata", {})

    source = metadata.get("source")

    if source is None:
        return None

    source = str(source).strip()

    if not source:
        return None

    return source


def _select_with_source_coverage(
    results: list[dict],
    top_k: int,
    rescue_margin: float = SOURCE_RESCUE_MARGIN,
    max_rescues_per_source: int = MAX_RESCUES_PER_SOURCE,
) -> list[dict]:
    """
    Select the final reranked evidence set.

    Normal cross-encoder ranking remains the primary signal.

    The problem this solves is a hard global top-k cutoff:

        chunk A from source X -> rank 1
        chunk B from source X -> rank 18

    If only the first 10 results are kept, chunk B disappears even
    though source X may contain multiple pieces of evidence required
    to answer the query.

    This function therefore allows a limited "source-aware rescue":

    1. Start with the normal top-k results.
    2. Look at candidates below the cutoff.
    3. If a candidate belongs to a source already represented in
       the selected results, and its score is close enough to the
       original cutoff, it may replace the weakest selected result.
    4. At most one additional chunk per source is rescued.
    5. The final number of results remains exactly top_k.

    This is completely domain-agnostic.
    """

    if top_k <= 0:
        return []

    if not results:
        return []

    if len(results) <= top_k:
        return results[:top_k]

    # The normal cross-encoder result set.
    selected = list(results[:top_k])

    # Candidates that the normal hard cutoff would discard.
    candidates = results[top_k:]

    # The original score threshold before any rescue occurs.
    #
    # Keeping this fixed prevents a chain of replacements from
    # progressively lowering the quality threshold.
    original_cutoff_score = min(
        result["rerank_score"]
        for result in selected
    )

    # Sources already represented by the normal top-k.
    represented_sources = {
        source
        for source in (
            _get_source(result)
            for result in selected
        )
        if source is not None
    }

    rescued_per_source: dict[str, int] = {}

    for candidate in candidates:
        source = _get_source(candidate)

        # Candidates without provenance cannot participate in
        # source-aware grouping.
        if source is None:
            continue

        # We only rescue an additional chunk from a source that is
        # already considered strongly relevant by the reranker.
        if source not in represented_sources:
            continue

        # Do not allow one source to dominate the rescue step.
        current_rescues = rescued_per_source.get(
            source,
            0,
        )

        if current_rescues >= max_rescues_per_source:
            continue

        candidate_score = candidate["rerank_score"]

        # Candidate must be reasonably close to the original
        # top-k boundary.
        if candidate_score < (
            original_cutoff_score - rescue_margin
        ):
            continue

        # Find the weakest currently selected candidate.
        weakest_index = min(
            range(len(selected)),
            key=lambda index: selected[index]["rerank_score"],
        )

        weakest = selected[weakest_index]
        weakest_source = _get_source(weakest)

        # Replacing a chunk with another chunk from the exact same
        # source would not improve source coverage.
        if weakest_source == source:
            continue

        # Replace the weakest selected candidate.
        selected[weakest_index] = candidate

        rescued_per_source[source] = (
            current_rescues + 1
        )

        print(
            "\n[reranker] Source-aware rescue:"
        )
        print(
            f"  Source: {source}"
        )
        print(
            f"  Candidate score: "
            f"{candidate_score:.4f}"
        )
        print(
            f"  Original cutoff: "
            f"{original_cutoff_score:.4f}"
        )
        print(
            f"  Replaced score: "
            f"{weakest['rerank_score']:.4f}"
        )

        # Keep the selected evidence ordered by reranker score.
        selected.sort(
            key=lambda result: result["rerank_score"],
            reverse=True,
        )

    return selected[:top_k]


def rerank(
    query: str,
    results: list[dict],
    top_k: int = 5,
    warn_on_truncation: bool = True,
):
    """
    Rerank retrieved results using a cross-encoder.

    The cross-encoder scores every retrieved candidate first.

    After scoring, a limited source-aware selection step prevents
    the hard top-k cutoff from unnecessarily discarding a second
    high-quality chunk from an already strongly relevant source.

    The final output still contains at most top_k results.
    """

    if not results:
        return []

    if top_k <= 0:
        return []

    if warn_on_truncation:
        for result in results:
            _warn_if_likely_truncated(
                query,
                result["document"],
            )

    pairs = [
        (
            query,
            build_rerank_context(result),
        )
        for result in results
    ]

    scores = reranker.predict(
        pairs,
        show_progress_bar=False,
    )

    reranked_results = []

    for result, score in zip(
        results,
        scores,
    ):
        reranked_results.append(
            {
                **result,
                "rerank_score": float(score),
            }
        )

         # ---------------------------------------------------------
    # Rank fusion
    # ---------------------------------------------------------
    #
    # We have two retrieval paths:
    #
    # 1. Global dense retrieval
    #    -> global_retrieval_rank
    #
    # 2. Source-local semantic retrieval
    #    -> source_anchor_rank + local_retrieval_rank
    #
    # Source-local retrieval is hierarchical:
    #
    #     source_path_rank =
    #         source_anchor_rank
    #         + local_retrieval_rank
    #         - 1
    #
    # We then use the best available retrieval rank.
    #
    # This avoids incorrectly treating a source-local result
    # as a poor global candidate simply because it was discovered
    # through source expansion.
    #

    for result in reranked_results:

        retrieval_ranks = []

        # -----------------------------------------------------
        # Global retrieval path
        # -----------------------------------------------------

        global_rank = result.get(
            "global_retrieval_rank"
        )

        if global_rank is not None:
            retrieval_ranks.append(
                global_rank
            )

        # -----------------------------------------------------
        # Source-local hierarchical retrieval path
        # -----------------------------------------------------

        source_anchor_rank = result.get(
            "source_anchor_rank"
        )

        local_rank = result.get(
            "local_retrieval_rank"
        )

        if (
            source_anchor_rank is not None
            and local_rank is not None
        ):
            source_path_rank = (
                source_anchor_rank
                + local_rank
                - 1
            )

            result["source_path_rank"] = (
                source_path_rank
            )

            retrieval_ranks.append(
                source_path_rank
            )

        # -----------------------------------------------------
        # Effective retrieval rank
        # -----------------------------------------------------

        if retrieval_ranks:
            effective_retrieval_rank = min(
                retrieval_ranks
            )

            result["retrieval_rank"] = (
                effective_retrieval_rank
            )

            result["retrieval_fusion_score"] = (
                RETRIEVAL_RRF_WEIGHT
                / (
                    RRF_K
                    + effective_retrieval_rank
                )
            )

        else:
            result["retrieval_rank"] = None

            result["retrieval_fusion_score"] = 0.0

    # ---------------------------------------------------------
    # Cross-encoder ranking
    # ---------------------------------------------------------

    reranked_by_cross_encoder = sorted(
        reranked_results,
        key=lambda result: result["rerank_score"],
        reverse=True,
    )

    for rerank_rank, result in enumerate(
        reranked_by_cross_encoder,
        start=1,
    ):
        result["rerank_rank"] = rerank_rank

        result["rerank_fusion_score"] = (
            RERANK_RRF_WEIGHT
            / (
                RRF_K
                + rerank_rank
            )
        )

    # ---------------------------------------------------------
    # Final fused score
    # ---------------------------------------------------------

    for result in reranked_results:
        result["fusion_score"] = (
            result["retrieval_fusion_score"]
            + result["rerank_fusion_score"]
        )

    # Highest fused score first.
    reranked_results.sort(
        key=lambda result: result["fusion_score"],
        reverse=True,
    )

    # Apply the existing generic source-aware selection.
    return _select_with_source_coverage(
        reranked_results,
        top_k=top_k,
    )