import re

from app.rag.query_decomposer import decompose_query_structured
from app.rag.retriever import retrieve, retrieve_by_source
from app.rag.reranker import rerank


# Maximum number of sources expanded for each subquery.
MAX_SOURCE_EXPANSIONS = 10

# Maximum number of chunks retrieved from each expanded source.
SOURCE_EXPANSION_TOP_K = 10


# Generic stopwords used only for lightweight query/source affinity.
# This does not assume anything about the domain or entity type.
_SOURCE_QUERY_STOPWORDS = {
    "what", "was", "were", "who", "whom", "whose", "which", "where",
    "when", "why", "how", "much", "many", "does", "do", "did",
    "the", "a", "an", "is", "are", "am", "be", "been", "being",
    "in", "on", "at", "of", "to", "for", "from", "and", "or",
    "with", "by", "among", "between", "than", "their", "his",
    "her", "its", "this", "that", "these", "those", "current",
    "had", "has", "have", "work", "works", "get", "got",
}


def _normalize_terms(text: str) -> set[str]:
    """
    Extract generic content terms from text.

    This helper is deliberately domain-agnostic. It is used only
    as a weak provenance/anchor signal; semantic retrieval remains
    the primary retrieval mechanism.
    """

    words = re.findall(r"[a-z0-9]+", text.lower())

    return {
        word
        for word in words
        if len(word) > 1
        and word not in _SOURCE_QUERY_STOPWORDS
    }


def _source_tokens(source: str) -> set[str]:
    """
    Extract generic tokens from the source basename.

    Handles common source formats such as:
        report.pdf
        Product_A.md
        customer-profile.json
        https://example/.../document-123

    No assumptions are made about what the source represents.
    """

    normalized = source.replace("\\", "/")
    basename = normalized.rsplit("/", 1)[-1]

    # Remove only the final extension.
    stem = basename.rsplit(".", 1)[0]

    return _normalize_terms(
        stem.replace("_", " ")
            .replace("-", " ")
    )


def _source_query_overlap(
    query: str,
    source: str,
) -> int:
    """
    Count exact normalized term overlap between a query and source.

    This is only a lightweight source-affinity signal. A value of
    zero never means the source is invalid; it only means there is
    no obvious textual anchor in the source identifier.
    """

    query_terms = _normalize_terms(query)
    source_terms = _source_tokens(source)

    if not query_terms or not source_terms:
        return 0

    return len(query_terms & source_terms)


def _get_source_filename(result: dict) -> str | None:
    """
    Extract the source filename from retrieval metadata.
    """

    metadata = result.get("metadata", {})

    return metadata.get("filename")


def _collect_unique_sources(
    results: list[dict],
) -> list[str]:
    """
    Return unique source identifiers in their first-seen retrieval order.
    """

    sources = []
    seen = set()

    for result in results:
        source = _get_source_filename(result)

        if not source or source in seen:
            continue

        seen.add(source)
        sources.append(source)

    return sources


def _build_cross_subquery_source_candidates(
    subquery: str,
    source_records: dict[str, dict],
    excluded_sources: set[str],
) -> list[str]:
    """
    Find sources discovered by other subqueries that have a textual
    source/query affinity with the current subquery.

    This is generic entity/concept coverage: it does not know whether
    a source represents a person, product, project, location, ticket,
    contract, or any other domain object.
    """

    candidates = []

    for source, record in source_records.items():

        if source in excluded_sources:
            continue

        overlap = _source_query_overlap(
            subquery,
            source,
        )

        if overlap <= 0:
            continue

        candidates.append(
            (
                overlap,
                record["best_global_rank"],
                source,
            )
        )

    # Prefer stronger source/query lexical anchors first, then the
    # best dense rank observed anywhere across the query.
    candidates.sort(
        key=lambda item: (
            -item[0],
            item[1],
            item[2],
        )
    )

    return [
        source
        for _, _, source in candidates
    ]


def _select_source_anchors(
    subquery: str,
    own_results: list[dict],
    source_records: dict[str, dict],
    max_sources: int,
    required_sources: list[str] | None = None,
) -> list[str]:
    """
    Select source anchors for one requirement.

    Required sources are selected first and are never displaced by
    ordinary cross-encoder/query-affinity ranking. This guarantees
    coverage for explicitly identified entities or other structured
    retrieval targets discovered during the coverage pass.

    Remaining optional sources follow the existing domain-agnostic
    affinity strategy.
    """

    required_sources = required_sources or []

    own_sources = _collect_unique_sources(own_results)

    own_anchored = [
        source
        for source in own_sources
        if _source_query_overlap(subquery, source) > 0
    ]

    own_unanchored = [
        source
        for source in own_sources
        if source not in set(own_anchored)
    ]

    selected: list[str] = []
    selected_set: set[str] = set()

    def add_required(sources: list[str]) -> None:
        for source in sources:
            if not source or source in selected_set:
                continue
            selected.append(source)
            selected_set.add(source)

    def add_optional(sources: list[str]) -> None:
        for source in sources:
            if len(selected) >= max_sources + len(required_sources):
                return
            if not source or source in selected_set:
                continue
            selected.append(source)
            selected_set.add(source)

    # 1. Guaranteed coverage for explicitly identified entities.
    add_required(required_sources)

    # 2. Preserve sources directly found for the current requirement.
    add_optional(own_anchored)

    # 3. Reuse strongly anchored sources discovered by other
    # requirements.
    cross_sources = _build_cross_subquery_source_candidates(
        subquery,
        source_records,
        excluded_sources=selected_set,
    )
    add_optional(cross_sources)

    # 4. Fill the remaining optional capacity with other own sources.
    add_optional(own_unanchored)

    return selected


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

        # -----------------------------------------------------
        # Preserve whether the source anchor was discovered
        # through another subquery.
        # -----------------------------------------------------

        if result.get("cross_subquery_source", False):
            existing["cross_subquery_source"] = True

    return [
        merged[document]
        for document in order
    ]


def _entity_coverage_sources(
    entity: str,
    coverage_results: list[dict],
    source_records: dict[str, dict],
) -> list[str]:
    """
    Identify source documents that plausibly correspond to an explicit
    entity/concept from the structured decomposition.

    The logic is generic: it compares normalized entity tokens with
    source identifiers. If no identifier match exists, the highest-ranked
    entity retrieval source is retained as a fallback anchor.
    """

    entity_tokens = _normalize_terms(entity)
    ranked_sources: list[str] = []
    seen: set[str] = set()

    def add(source: str | None) -> None:
        if not source or source in seen:
            return
        seen.add(source)
        ranked_sources.append(source)

    for result in coverage_results:
        source = _get_source_filename(result)
        if not source:
            continue

        source_tokens = _source_tokens(source)
        if entity_tokens and entity_tokens & source_tokens:
            add(source)

    # Fallback: search the shared registry if the coverage retrieval
    # result itself did not expose a filename-token match.
    if not ranked_sources and entity_tokens:
        candidates = []
        for source, record in source_records.items():
            overlap = len(entity_tokens & _source_tokens(source))
            if overlap > 0:
                candidates.append((overlap, record["best_global_rank"], source))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        for _, _, source in candidates[:1]:
            add(source)

    # Last-resort coverage anchor: the first source returned by the
    # entity-only retrieval. It is still provenance-backed retrieval,
    # never an invented source.
    if not ranked_sources and coverage_results:
        add(_get_source_filename(coverage_results[0]))

    return ranked_sources


def multi_evidence_retrieve(
    query: str,
    retrieval_top_k: int = 20,
    rerank_top_k: int | None = None,
):
    """
    Retrieve evidence for a complex query.

    Pipeline:
        1. Structured query decomposition with explicit entities.
        2. Global dense retrieval.
        3. Entity/concept coverage retrieval.
        4. Guaranteed source expansion for required sources.
        5. Source-local semantic expansion.
        6. Deduplication with provenance preservation.
        7. Cross-encoder reranking WITHOUT output truncation.

    `rerank_top_k` is retained for API compatibility. When omitted,
    all reranked candidates are returned. The default is intentionally
    uncapped so downstream consolidation cannot silently lose evidence.
    """

    requirements = decompose_query_structured(query)

    print("\nGenerated requirements:")
    for i, item in enumerate(requirements, start=1):
        print(f"{i}. {item['requirement']}")
        if item.get("entities"):
            print(f"   entities: {item['entities']}")

    global_results_by_requirement: list[dict] = []
    source_records: dict[str, dict] = {}

    # ---------------------------------------------------------
    # 1. Global retrieval + entity coverage retrieval
    # ---------------------------------------------------------
    for requirement_index, item in enumerate(requirements):
        subquery = item["requirement"]
        entities = item.get("entities", [])

        print("\nProcessing global retrieval:")
        print(subquery)

        results = retrieve(subquery, top_k=retrieval_top_k)

        for global_rank, result in enumerate(results, start=1):
            result["global_retrieval_rank"] = global_rank
            result["retrieval_path"] = "global"

            source = _get_source_filename(result)
            if not source:
                continue

            record = source_records.setdefault(
                source,
                {
                    "best_global_rank": global_rank,
                    "requirement_indexes": set(),
                },
            )
            record["best_global_rank"] = min(
                record["best_global_rank"],
                global_rank,
            )
            record["requirement_indexes"].add(requirement_index)

        coverage_results: list[dict] = []

        for entity in entities:
            entity_hits = retrieve(
                entity,
                top_k=max(5, retrieval_top_k // 2),
            )

            for coverage_rank, result in enumerate(entity_hits, start=1):
                result["coverage_query"] = entity
                result["coverage_retrieval_rank"] = coverage_rank
                result["retrieval_path"] = "entity_coverage"

                source = _get_source_filename(result)
                if source:
                    record = source_records.setdefault(
                        source,
                        {
                            "best_global_rank": retrieval_top_k + coverage_rank,
                            "requirement_indexes": set(),
                        },
                    )
                    record["requirement_indexes"].add(requirement_index)

            coverage_results.extend(entity_hits)

        required_sources: list[str] = []
        for entity in entities:
            required_sources.extend(
                source
                for source in _entity_coverage_sources(
                    entity,
                    coverage_results,
                    source_records,
                )
                if source not in required_sources
            )

        print("\nRequired entity coverage sources:")
        if required_sources:
            for source in required_sources:
                print(f"- {source}")
        else:
            print("- none")

        global_results_by_requirement.append(
            {
                "requirement": subquery,
                "entities": entities,
                "results": results,
                "coverage_results": coverage_results,
                "required_sources": required_sources,
            }
        )

    evidence = []

    # ---------------------------------------------------------
    # 2. Source-local expansion + final uncapped reranking
    # ---------------------------------------------------------
    for requirement_index, item in enumerate(global_results_by_requirement):
        subquery = item["requirement"]
        results = item["results"]
        coverage_results = item["coverage_results"]
        required_sources = item["required_sources"]

        print("\nProcessing requirement:")
        print(subquery)

        source_anchors = _select_source_anchors(
            subquery=subquery,
            own_results=results,
            source_records=source_records,
            max_sources=MAX_SOURCE_EXPANSIONS,
            required_sources=required_sources,
        )

        print("\nSource expansion candidates:")
        for source_rank, source in enumerate(source_anchors, start=1):
            print(
                f"- {source} "
                f"(source anchor rank={source_rank}, "
                f"required={'yes' if source in required_sources else 'no'})"
            )

        expanded_results = []
        own_sources = set(_collect_unique_sources(results))

        for source_rank, source in enumerate(source_anchors, start=1):
            source_results = retrieve_by_source(
                query=subquery,
                source=source,
                top_k=SOURCE_EXPANSION_TOP_K,
            )

            print(f"\nExpanded source: {source}")

            cross_subquery_source = source not in own_sources
            entity_coverage_source = source in required_sources

            for result in source_results:
                result["source_anchor_rank"] = source_rank
                result["source_expanded"] = True
                result["cross_subquery_source"] = cross_subquery_source
                result["entity_coverage_source"] = entity_coverage_source

                print(
                    f"  local_rank={result['local_retrieval_rank']} "
                    f"| {result['metadata'].get('filename')}"
                )
                print(
                    "    ",
                    result["document"][:250].replace("\n", " "),
                )

            expanded_results.extend(source_results)

        combined_results = (
            results
            + coverage_results
            + expanded_results
        )

        combined_results = _merge_duplicate_results(combined_results)

        # Preserve every candidate after scoring. The caller can still
        # apply an explicit downstream budget, but retrieval itself does
        # not silently discard evidence here.
        effective_top_k = len(combined_results)
        if rerank_top_k is not None:
            effective_top_k = max(effective_top_k, rerank_top_k)

        final_results = rerank(
            subquery,
            combined_results,
            top_k=effective_top_k,
        )

        evidence.append(
            {
                "subquery": subquery,
                "entities": item["entities"],
                "results": final_results,
                "required_sources": required_sources,
            }
        )

    return evidence
