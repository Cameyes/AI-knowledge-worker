from app.rag.query_decomposer import decompose_query
from app.rag.retriever import retrieve, retrieve_by_source
from app.rag.reranker import rerank


# Maximum number of sources expanded for each subquery.
MAX_SOURCE_EXPANSIONS = 3

# Maximum number of chunks retrieved from each expanded source.
SOURCE_EXPANSION_TOP_K = 10


def _get_source_filename(result: dict) -> str | None:
    """
    Extract a stable source identifier from retrieval metadata.

    Prefer the retriever's filename field, while accepting a generic
    source field when present. The retrieval layer remains domain-agnostic.
    """

    metadata = result.get("metadata", {})

    source = metadata.get("filename") or metadata.get("source")

    if source is None:
        return None

    source = str(source).strip()

    return source or None


def _select_source_anchors(
    query: str,
    results: list[dict],
    max_source_expansions: int,
) -> list[str]:
    """
    Select source anchors from cross-encoder-ranked global results.

    Global dense retrieval supplies the candidate pool. The
    cross-encoder then determines which candidates are most relevant
    to the complete subquery before any source-local expansion occurs.

    This avoids expanding every distinct source that merely appears
    somewhere in the dense top-k results.

    The function is fully domain-agnostic: it never extracts entity
    types, uses domain vocabulary, or relies on fixed distance thresholds.
    """

    if not results or max_source_expansions <= 0:
        return []

    # Rerank the complete global candidate pool so source selection is
    # driven by query-document relevance rather than dense rank alone.
    reranked_results = rerank(
        query,
        results,
        top_k=len(results),
    )

    anchors: list[str] = []
    seen: set[str] = set()

    for result in reranked_results:
        source = _get_source_filename(result)

        if not source or source in seen:
            continue

        seen.add(source)
        anchors.append(source)

        if len(anchors) >= max_source_expansions:
            break

    return anchors


def _merge_duplicate_results(
    results: list[dict],
) -> list[dict]:
    """
    Merge duplicate chunks while preserving retrieval
    information from all retrieval paths.

    A chunk may be found through:

    1. Global dense retrieval.
    2. Source-local semantic retrieval.

    We must preserve both pieces of information instead
    of discarding one copy.
    """

    merged = {}
    order = []

    for result in results:

        document = result.get("document", "")

        if document not in merged:
            merged[document] = result.copy()
            order.append(document)
            continue

        existing = merged[document]

        # -----------------------------------------------------
        # Preserve global retrieval rank
        # -----------------------------------------------------

        global_rank = result.get(
            "global_retrieval_rank"
        )

        if global_rank is not None:

            existing_global_rank = existing.get(
                "global_retrieval_rank"
            )

            if (
                existing_global_rank is None
                or global_rank < existing_global_rank
            ):
                existing["global_retrieval_rank"] = (
                    global_rank
                )

        # -----------------------------------------------------
        # Preserve source-local retrieval rank
        # -----------------------------------------------------

        local_rank = result.get(
            "local_retrieval_rank"
        )

        if local_rank is not None:

            existing_local_rank = existing.get(
                "local_retrieval_rank"
            )

            if (
                existing_local_rank is None
                or local_rank < existing_local_rank
            ):
                existing["local_retrieval_rank"] = (
                    local_rank
                )

        # -----------------------------------------------------
        # Preserve source-anchor rank
        # -----------------------------------------------------

        source_anchor_rank = result.get(
            "source_anchor_rank"
        )

        if source_anchor_rank is not None:

            existing_anchor_rank = existing.get(
                "source_anchor_rank"
            )

            if (
                existing_anchor_rank is None
                or source_anchor_rank < existing_anchor_rank
            ):
                existing["source_anchor_rank"] = (
                    source_anchor_rank
                )

        # -----------------------------------------------------
        # Preserve the best dense distance
        # -----------------------------------------------------

        distance = result.get("distance")

        if distance is not None:

            existing_distance = existing.get(
                "distance"
            )

            if (
                existing_distance is None
                or distance < existing_distance
            ):
                existing["distance"] = distance

        # -----------------------------------------------------
        # Remember that this chunk was source-expanded
        # -----------------------------------------------------

        if result.get("source_expanded", False):
            existing["source_expanded"] = True

    return [
        merged[document]
        for document in order
    ]


def multi_evidence_retrieve(
    query: str,
    retrieval_top_k: int = 10,
    rerank_top_k: int = 5,
):
    """
    Retrieve evidence for a complex query.

    Pipeline:

        1. Decompose query.
        2. Global dense retrieval.
        3. Identify promising sources.
        4. Perform source-local semantic expansion.
        5. Preserve global + local retrieval ranks.
        6. Deduplicate without losing retrieval information.
        7. Cross-encoder reranking + RRF.
    """

    # ---------------------------------------------------------
    # 1. Decompose query
    # ---------------------------------------------------------

    subqueries = decompose_query(query)

    print("\nGenerated subqueries:")

    for i, subquery in enumerate(
        subqueries,
        start=1,
    ):
        print(f"{i}. {subquery}")

    evidence = []

    # ---------------------------------------------------------
    # 2. Process each subquery independently
    # ---------------------------------------------------------

    for subquery in subqueries:

        print("\nProcessing subquery:")
        print(subquery)

        # -----------------------------------------------------
        # 2a. Initial global dense retrieval
        # -----------------------------------------------------

        results = retrieve(
            subquery,
            top_k=retrieval_top_k,
        )

        # Preserve the original global retrieval ranking and provenance.
        for global_rank, result in enumerate(
            results,
            start=1,
        ):
            result["global_retrieval_rank"] = global_rank
            result["retrieval_path"] = "global"

        # -----------------------------------------------------
        # 2b. Identify promising source anchors
        # -----------------------------------------------------

        source_anchors = _select_source_anchors(
            subquery,
            results,
            MAX_SOURCE_EXPANSIONS,
        )

        print("\nSource expansion candidates:")

        for source_rank, source in enumerate(
            source_anchors,
            start=1,
        ):
            print(
                f"- {source} "
                f"(source anchor rank={source_rank})"
            )

        # -----------------------------------------------------
        # 2c. Semantic source-local expansion
        # -----------------------------------------------------

        expanded_results = []

        for source_rank, source in enumerate(
            source_anchors,
            start=1,
        ):

            source_results = retrieve_by_source(
                query=subquery,
                source=source,
                top_k=SOURCE_EXPANSION_TOP_K,
            )

            print(
                f"\nExpanded source: {source}"
            )

            for i, result in enumerate(
                source_results,
                start=1,
            ):
                print(
                    f"  {i}. "
                    f"distance={result['distance']:.4f} "
                    f"| local_rank={result['local_retrieval_rank']} "
                    f"| {result['metadata'].get('filename')}"
                )

                print(
                    "     ",
                    result["document"][:250]
                    .replace("\n", " ")
                )

                # Preserve the hierarchical retrieval path.
                result["source_anchor_rank"] = source_rank
                result["source_expanded"] = True
                result["retrieval_path"] = "source_expansion"

            expanded_results.extend(
                source_results
            )

        # -----------------------------------------------------
        # 2d. Combine original + expanded candidates
        # -----------------------------------------------------

        combined_results = (
            results
            + expanded_results
        )

        # -----------------------------------------------------
        # 2e. Deduplicate WITHOUT losing ranking metadata
        # -----------------------------------------------------

        combined_results = _merge_duplicate_results(
            combined_results
        )

        # -----------------------------------------------------
        # 2f. Final cross-encoder reranking + RRF
        # -----------------------------------------------------

        final_results = rerank(
            subquery,
            combined_results,
            top_k=rerank_top_k,
        )

        evidence.append(
            {
                "subquery": subquery,
                "results": final_results,
            }
        )

    return evidence