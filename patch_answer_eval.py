from pathlib import Path

TARGET = Path("app/evaluation/answer_eval.py")

IMPORT_ANCHOR = "from app.rag.answer_generator import generate_answer\n"
IMPORT_LINE = "from app.rag.evidence_consolidator import consolidate_evidence\n"

OLD_IMPORT = IMPORT_ANCHOR + IMPORT_LINE

OLD_BLOCK = '''    generation_evidence = select_evidence_by_subquery(\n        grouped_evidence,\n        max_items_per_group=3,\n    )\n'''

NEW_BLOCK = '''    generation_evidence = select_evidence_by_subquery(\n        grouped_evidence,\n        max_items_per_group=3,\n    )\n\n    # Convert retrieved chunks into atomic, requirement-scoped facts\n    # before generation. No new evidence is created here.\n    generation_evidence = consolidate_evidence(\n        generation_evidence,\n        max_candidates_per_group=3,\n        max_chars_per_candidate=3500,\n    )\n'''

OLD_CONTEXT = '''    for group_key, items in grouped_evidence.items():\n        blocks.append(f"GROUP: {group_key}")\n\n        for item in items:\n            source = item["result"]["metadata"].get(\n                "source", "Unknown source"\n            )\n            filename = source.replace("\\\\", "/").split("/")[-1]\n\n            blocks.append(\n                f"Subquery: {item['subquery']}\\n"\n                f"Source: {filename}\\n"\n                f"Evidence:\\n"\n                f"{item['result']['document']}"\n            )\n'''

NEW_CONTEXT = '''    for group_key, group_data in grouped_evidence.items():\n        # Consolidated evidence representation.\n        if isinstance(group_data, dict) and "facts" in group_data:\n            status = group_data.get("status", "insufficient")\n            blocks.append(\n                f"GROUP: {group_key}\\nSTATUS: {status.upper()}"\n            )\n\n            for fact in group_data.get("facts", []):\n                blocks.append(\n                    f"FACT: {fact['statement']}\\n"\n                    f"Evidence indices: {fact['evidence_indices']}"\n                )\n\n            for index, item in enumerate(group_data.get("evidence", [])):\n                result = item.get("result", {})\n                source = result.get("metadata", {}).get(\n                    "source", "Unknown source"\n                )\n                filename = source.replace("\\\\", "/").split("/")[-1]\n                document = str(result.get("document", ""))\n\n                blocks.append(\n                    f"Supporting evidence {index}: {filename}\\n"\n                    f"{document[:1800]}"\n                )\n\n            continue\n\n        # Backward-compatible formatting for non-consolidated evidence.\n        blocks.append(f"GROUP: {group_key}")\n\n        for item in group_data:\n            source = item["result"]["metadata"].get(\n                "source", "Unknown source"\n            )\n            filename = source.replace("\\\\", "/").split("/")[-1]\n\n            blocks.append(\n                f"Subquery: {item['subquery']}\\n"\n                f"Source: {filename}\\n"\n                f"Evidence:\\n"\n                f"{item['result']['document']}"\n            )\n'''

text = TARGET.read_text(encoding="utf-8")

if "from app.rag.evidence_consolidator import consolidate_evidence" not in text:
    if IMPORT_ANCHOR not in text:
        raise RuntimeError(
            "Could not find the answer_generator import in answer_eval.py"
        )
    text = text.replace(
        IMPORT_ANCHOR,
        IMPORT_ANCHOR + IMPORT_LINE,
        1,
    )

if OLD_BLOCK not in text:
    raise RuntimeError(
        "Could not find the expected generation_evidence block. "
        "Your local answer_eval.py differs from the known generic baseline."
    )

text = text.replace(OLD_BLOCK, NEW_BLOCK, 1)

if OLD_CONTEXT not in text:
    raise RuntimeError(
        "Could not find the expected build_compact_context block. "
        "Patch the evaluator manually using the v2 file in this bundle."
    )

text = text.replace(OLD_CONTEXT, NEW_CONTEXT, 1)

TARGET.write_text(text, encoding="utf-8")
print("Patched app/evaluation/answer_eval.py for atomic evidence consolidation v2")
