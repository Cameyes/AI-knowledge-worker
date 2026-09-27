from app.rag.evidence_retriever import (
    _entity_coverage_sources,
    _select_source_anchors,
)


def _result(filename: str) -> dict:
    return {
        "document": "dummy",
        "metadata": {"filename": filename, "source": filename},
    }


def main() -> None:
    david = _result("David Kim.md")
    wrong = _result("Robert Chen.md")

    selected = _select_source_anchors(
        subquery="What was the 2022 bonus of David Kim?",
        own_results=[wrong],
        source_records={
            "David Kim.md": {"best_global_rank": 9, "requirement_indexes": {0}},
            "Robert Chen.md": {"best_global_rank": 1, "requirement_indexes": {0}},
        },
        max_sources=1,
        required_sources=["David Kim.md"],
    )

    assert selected[0] == "David Kim.md", selected

    coverage_sources = _entity_coverage_sources(
        "David Kim",
        [david, wrong],
        {
            "David Kim.md": {"best_global_rank": 9, "requirement_indexes": {0}},
            "Robert Chen.md": {"best_global_rank": 1, "requirement_indexes": {0}},
        },
    )

    assert coverage_sources[0] == "David Kim.md", coverage_sources

    print("Retrieval coverage structural test: PASS")


if __name__ == "__main__":
    main()
