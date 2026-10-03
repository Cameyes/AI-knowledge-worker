"""Regression test: insufficient consolidation must not expose raw candidates for guessing."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODULE_PATH = ROOT / "answer_generator.py"

app = types.ModuleType("app")
rag = types.ModuleType("app.rag")
llm = types.ModuleType("app.rag.llm_client")
llm.safe_completion_text = lambda *args, **kwargs: kwargs.get("fallback", "")
sys.modules.setdefault("app", app)
sys.modules.setdefault("app.rag", rag)
sys.modules.setdefault("app.rag.llm_client", llm)

spec = importlib.util.spec_from_file_location("answer_generator_insufficient", MODULE_PATH)
ag = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = ag
spec.loader.exec_module(ag)


def main():
    captured = {}

    def fake_completion(prompt, *args, **kwargs):
        captured["prompt"] = prompt
        return "I could not determine the answer from the available evidence."

    original = ag.safe_completion_text
    ag.safe_completion_text = fake_completion

    try:
        evidence = {
            "What is Jordan Blake's current job title?": {
                "group": "What is Jordan Blake's current job title?",
                "status": "insufficient",
                "resolution_basis": None,
                "resolved_fact": None,
                "facts": [],
                "evidence": [
                    {
                        "result": {
                            "document": "Jordan Blake unrelated source text",
                            "metadata": {"source": "Jordan Blake.md"},
                        }
                    },
                    {
                        "result": {
                            "document": "Jordan K. Bishop unrelated source text",
                            "metadata": {"source": "Jordan K. Bishop.md"},
                        }
                    },
                ],
                "raw_candidates": [],
            }
        }

        result = ag.generate_answer(
            "What is Jordan Blake's current job title?",
            evidence,
        )

        assert result["sources"] == [
            "Jordan Blake.md",
            "Jordan K. Bishop.md",
        ]
        prompt = captured["prompt"]
        assert "STATUS: insufficient" in prompt
        assert "Jordan Blake unrelated source text" not in prompt
        assert "Jordan K. Bishop unrelated source text" not in prompt
        assert "available evidence is" in prompt.lower() and "insufficient" in prompt.lower()
    finally:
        ag.safe_completion_text = original

    print("Insufficient-answer safety test: PASS")


if __name__ == "__main__":
    main()
