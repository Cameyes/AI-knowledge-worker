"""Structural test for atomic evidence consolidation v4."""
from app.rag import evidence_consolidator as ec


class FakeModel:
    calls = []
    counts = {}

    def __call__(self, prompt, *args, **kwargs):
        self.calls.append(prompt)
        key = "A" if "Item A" in prompt else "B"
        self.counts[key] = self.counts.get(key, 0) + 1
        if key == "A":
            return {
                "group": "What is the 2022 value of Item A?",
                "status": "supported",
                "facts": [{"statement": "Item A: $8,000 in 2022.", "evidence_indices": [1, 99]}],
            }
        if self.counts[key] == 1:
            return {
                "group": "What is the 2022 value of Item B?",
                "status": "insufficient",
                "facts": [],
            }
        return {
            "group": "What is the 2022 value of Item B?",
            "status": "supported",
            "facts": [{"statement": "Item B: $9,000 in 2022.", "evidence_indices": [0]}],
        }


def main():
    fake = FakeModel()
    original = ec.safe_completion_json
    ec.safe_completion_json = fake
    try:
        grouped = {
            "What is the 2022 value of Item A?": [
                {"subquery": "What is the 2022 value of Item A?", "result": {"document": "irrelevant", "metadata": {"source": "a.md"}}},
                {"subquery": "What is the 2022 value of Item A?", "result": {"document": "Item A: 2022 = $8,000", "metadata": {"source": "a.md"}}},
            ],
            "What is the 2022 value of Item B?": [
                {"subquery": "What is the 2022 value of Item B?", "result": {"document": "Item B: 2022 = $9,000", "metadata": {"source": "b.md"}}},
            ],
        }
        result = ec.consolidate_evidence(grouped)
        assert len(fake.calls) == 3
        assert result["What is the 2022 value of Item A?"]["facts"][0]["evidence_indices"] == [1]
        assert result["What is the 2022 value of Item A?"]["evidence"][0]["result"]["document"] == "Item A: 2022 = $8,000"
        assert result["What is the 2022 value of Item B?"]["facts"][0]["evidence_indices"] == [0]
    finally:
        ec.safe_completion_json = original
    print("Atomic consolidation structural test v4: PASS")


if __name__ == "__main__":
    main()
