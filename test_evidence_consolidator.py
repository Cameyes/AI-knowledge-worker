"""Phase 1 tests for provenance-aware fact extraction."""
from app.rag import evidence_consolidator as ec


class FakeModel:
    calls = []

    def __call__(self, prompt, *args, **kwargs):
        self.calls.append(prompt)
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
                    "statement": "Item A was moved to Suspended during the maintenance event.",
                    "attribute": "status",
                    "value": "Suspended",
                    "value_type": "narrative_reference",
                    "temporal_scope": "historical",
                    "section_type": "history",
                    "evidence_indices": [1],
                    "qualifiers": {"event": "maintenance"},
                },
            ],
        }


def main():
    fake = FakeModel()
    original = ec.safe_completion_json
    ec.safe_completion_json = fake
    try:
        grouped = {
            "What is the current status of Item A?": [
                {
                    "subquery": "What is the current status of Item A?",
                    "result": {
                        "document": "## Summary\nStatus: Active",
                        "metadata": {
                            "source": "a.md",
                            "candidate_id": "cand-a",
                        },
                    },
                },
                {
                    "subquery": "What is the current status of Item A?",
                    "result": {
                        "document": "## History\nItem A was moved to Suspended during the maintenance event.",
                        "metadata": {
                            "source": "a.md",
                            "candidate_id": "cand-b",
                        },
                    },
                },
            ]
        }

        result = ec.consolidate_evidence(grouped)
        group = result["What is the current status of Item A?"]

        assert fake.calls
        prompt = fake.calls[0]
        assert "Do not return a status field" in prompt
        assert '"declared_attribute"' in prompt
        assert '"narrative_reference"' in prompt
        assert '"temporal_scope"' in prompt

        assert group["status"] == "supported"  # compatibility only; resolver comes in Phase 2
        assert len(group["facts"]) == 2

        declared = group["facts"][0]
        narrative = group["facts"][1]

        assert declared["attribute"] == "status"
        assert declared["value"] == "Active"
        assert declared["value_type"] == "declared_attribute"
        assert declared["temporal_scope"] == "current"
        assert declared["section_type"] == "summary"
        assert declared["source_candidate_id"] == "cand-a"

        assert narrative["value_type"] == "narrative_reference"
        assert narrative["temporal_scope"] == "historical"
        assert narrative["section_type"] == "history"
        assert narrative["source_candidate_id"] == "cand-b"
        assert narrative["qualifiers"]["event"] == "maintenance"

    finally:
        ec.safe_completion_json = original

    print("Phase 1 provenance extraction test: PASS")


if __name__ == "__main__":
    main()
