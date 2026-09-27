"""Phase 2 tests for deterministic evidence resolution."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODULE_PATH = ROOT / "app" / "rag" / "evidence_consolidator.py"

# Load the consolidator without requiring the user's full application package.
app = types.ModuleType("app")
rag = types.ModuleType("app.rag")
llm = types.ModuleType("app.rag.llm_client")
llm.safe_completion_json = lambda *args, **kwargs: kwargs.get("fallback", {})
sys.modules.setdefault("app", app)
sys.modules.setdefault("app.rag", rag)
sys.modules.setdefault("app.rag.llm_client", llm)

spec = importlib.util.spec_from_file_location("evidence_consolidator", MODULE_PATH)
ec = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = ec
spec.loader.exec_module(ec)
ec.AtomicFact.model_rebuild()


def fact(statement, attribute, value, value_type, temporal_scope="unknown"):
    return ec.AtomicFact(
        statement=statement,
        attribute=attribute,
        value=value,
        value_type=value_type,
        temporal_scope=temporal_scope,
        section_type="summary" if value_type == "declared_attribute" else "history",
        source_candidate_id="test-candidate",
    )


def assert_result(facts, expected_status, expected_basis=None, expected_value=None):
    status, basis, resolved = ec._resolve_requirement_facts(facts)
    assert status == expected_status, (status, expected_status)
    assert basis == expected_basis, (basis, expected_basis)
    if expected_value is None:
        assert resolved is None
    else:
        assert resolved is not None
        assert resolved.value == expected_value


def main():
    # Same value from multiple representations: no resolution required.
    assert_result(
        [
            fact("Status: Active", "status", "Active", "declared_attribute", "current"),
            fact("The item remains Active.", "status", "Active", "narrative_reference", "current"),
        ],
        "supported",
        None,
        "Active",
    )

    # Core Phase 2 rule: declared value wins over conflicting narrative value.
    assert_result(
        [
            fact("Job Title: Data Engineer", "job_title", "Data Engineer", "declared_attribute", "current"),
            fact("January 2021 - Present: Senior Data Engineer", "job_title", "Senior Data Engineer", "narrative_reference", "current"),
        ],
        "resolved",
        "declared_attribute_precedence",
        "Data Engineer",
    )

    # Benchmark-style Maxine case: explicit Present still resolves by the
    # deliberate V1 declared-field precedence policy.
    assert_result(
        [
            fact("Job Title: Data Engineer", "job_title", "Data Engineer", "declared_attribute", "current"),
            fact("January 2021 - Present: Senior Data Engineer", "job_title", "Senior Data Engineer", "narrative_reference", "current"),
        ],
        "resolved",
        "declared_attribute_precedence",
        "Data Engineer",
    )

    # Two declared values are a genuine unresolved conflict.
    assert_result(
        [
            fact("Status: Active", "status", "Active", "declared_attribute"),
            fact("Status: Suspended", "status", "Suspended", "declared_attribute"),
        ],
        "contradictory",
    )

    # Narrative-only disagreement is also unresolved.
    assert_result(
        [
            fact("The item was moved to Active.", "status", "Active", "narrative_reference", "historical"),
            fact("The item was moved to Suspended.", "status", "Suspended", "narrative_reference", "historical"),
        ],
        "contradictory",
    )

    # Unknown provenance with a real attribute must never trigger automatic
    # precedence resolution.
    assert_result(
        [
            fact("Status: Active", "status", "Active", "declared_attribute"),
            fact("Status is Suspended.", "status", "Suspended", "unknown"),
        ],
        "contradictory",
    )

    # Unknown extractive spans with no attribute are preserved evidence, but
    # they do not block resolution based on structured facts.
    structured = fact(
        "Status: Active", "status", "Active", "declared_attribute", "current"
    )
    raw_span = ec.AtomicFact(
        statement="status current active",
        attribute="",
        value="status current active",
        value_type="unknown",
        temporal_scope="unknown",
        source_candidate_id="candidate:0",
    )
    assert_result([structured, raw_span], "supported", None, "Active")

    # No resolvable facts means insufficient evidence.
    assert_result([], "insufficient")

    # Facts about unrelated attributes must not be auto-resolved as one value.
    assert_result(
        [
            fact("Status: Active", "status", "Active", "declared_attribute"),
            fact("Version: 3.2", "version", "3.2", "declared_attribute"),
        ],
        "contradictory",
    )

    # Integration: consolidate_evidence now derives status/resolution fields
    # from the extracted facts instead of trusting an LLM-provided status.
    class FakeModel:
        def __call__(self, prompt, *args, **kwargs):
            return {
                "group": "What is the current status of Item A?",
                "facts": [
                    {
                        "statement": "Status: Active",
                        "attribute": "status",
                        "value": "Active",
                        "value_type": "declared_attribute",
                        "temporal_scope": "current",
                        "section_type": "summary",
                        "evidence_indices": [0],
                        "qualifiers": {},
                    },
                    {
                        "statement": "Item A was moved to Suspended during maintenance.",
                        "attribute": "status",
                        "value": "Suspended",
                        "value_type": "narrative_reference",
                        "temporal_scope": "historical",
                        "section_type": "history",
                        "evidence_indices": [1],
                        "qualifiers": {},
                    },
                ],
            }

    original_llm = ec.safe_completion_json
    original_extractive = ec._extractive_spans
    ec.safe_completion_json = FakeModel()
    ec._extractive_spans = lambda group, items: []
    try:
        grouped = {
            "What is the current status of Item A?": [
                {
                    "subquery": "What is the current status of Item A?",
                    "result": {
                        "document": "Status: Active",
                        "metadata": {"source": "a.md", "candidate_id": "cand-a"},
                    },
                },
                {
                    "subquery": "What is the current status of Item A?",
                    "result": {
                        "document": "Item A was moved to Suspended during maintenance.",
                        "metadata": {"source": "a.md", "candidate_id": "cand-b"},
                    },
                },
            ]
        }
        result = ec.consolidate_evidence(grouped)
        resolved = result["What is the current status of Item A?"]
        assert resolved["status"] == "resolved"
        assert resolved["resolution_basis"] == "declared_attribute_precedence"
        assert resolved["resolved_fact"]["value"] == "Active"
        assert len(resolved["facts"]) == 2
        assert len(resolved["raw_candidates"]) == 2
    finally:
        ec.safe_completion_json = original_llm
        ec._extractive_spans = original_extractive

    print("Phase 2 deterministic resolver tests: PASS")


if __name__ == "__main__":
    main()
