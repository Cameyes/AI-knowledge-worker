from pathlib import Path
import subprocess
import sys

EXTRACT = Path("app/extract.py")


def run_test(test_name: str) -> int:
    cmd = [sys.executable, "-m", "unittest", test_name, "-v"]
    print("\n$", " ".join(cmd))
    return subprocess.run(cmd).returncode


def mutate_once(label: str, old: str, new: str, test_name: str) -> None:
    source = EXTRACT.read_text(encoding="utf-8")

    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly 1 mutation target, found {count}"
        )

    # Apply mutation
    EXTRACT.write_text(
        source.replace(old, new, 1),
        encoding="utf-8",
    )

    try:
        rc = run_test(test_name)

        # We EXPECT the regression test to fail under the mutation.
        if rc == 0:
            raise AssertionError(
                f"{label}: MUTATION SURVIVED — "
                "the regression test passed even after the security "
                "control was deliberately removed."
            )

        print(f"{label}: PASS — mutation was killed.")

    finally:
        # ALWAYS restore original source
        EXTRACT.write_text(source, encoding="utf-8")


def main() -> None:
    if not EXTRACT.exists():
        raise SystemExit(
            f"Could not find {EXTRACT}. "
            "Run this script from the backend directory."
        )

    print("=== Phase 3 mutation test ===")
    print(f"Target: {EXTRACT.resolve()}")

    # ============================================================
    # MUTATION 1
    # Re-introduce the old dangerous firewall early-return.
    #
    # Current safe code:
    #
    #     if firewall_verdict == "mismatch":
    #         return "mismatch"
    #
    # Mutated code:
    #
    #     if firewall_verdict in ("match", "mismatch"):
    #         return firewall_verdict
    #
    # This resurrects the old document-identity bypass.
    # ============================================================

    old_early_return = """            if firewall_verdict == "mismatch":
                return "mismatch"
"""

    new_early_return = """            if firewall_verdict in ("match", "mismatch"):
                return firewall_verdict
"""

    mutate_once(
        "MUTATION 1 — firewall match -> early return",
        old_early_return,
        new_early_return,
        "app.tests.test_extract.DefaultEntityAttributor.test_document_identity_match_cannot_override_jane_quote",
    )

    # ============================================================
    # MUTATION 2
    # Remove the markdown-table row ownership protection.
    #
    # This simulates the case where the competing entity (Jane)
    # is omitted from the model's proposed entities, so entity-name
    # boundaries cannot protect us.
    # ============================================================

    old_table_guard = """        if table_row(mention_line) or table_row(quote_line) or table_row(quote_end_line):
            if mention_line != quote_line or quote_line != quote_end_line:
                return False

"""

    new_table_guard = """        if False:
            if mention_line != quote_line or quote_line != quote_end_line:
                return False

"""

    old_same_line_guard = """        if label is None and not heading_only and mention_line != quote_line:
            return False
"""

    new_same_line_guard = """        if False:
            return False
"""

    source = EXTRACT.read_text(encoding="utf-8")

    if source.count(old_table_guard) != 1:
        raise RuntimeError("Mutation 2: table guard target not found exactly once.")

    if source.count(old_same_line_guard) != 1:
        raise RuntimeError("Mutation 2: same-line guard target not found exactly once.")

    mutated = source.replace(old_table_guard, new_table_guard, 1)
    mutated = mutated.replace(old_same_line_guard, new_same_line_guard, 1)

    EXTRACT.write_text(mutated, encoding="utf-8")

    try:
        rc = run_test(
            "app.tests.test_extract.CrossEntityAttribution."
            "test_27_unlabeled_table_with_omitted_competing_entity_fails_closed"
        )

        if rc == 0:
            raise AssertionError(
                "MUTATION 2 SURVIVED — all deterministic cross-line/table "
                "ownership protection was deliberately removed, but the "
                "regression test still passed."
            )

        print(
            "MUTATION 2 — remove omitted-entity ownership boundaries: "
            "PASS — mutation was killed."
        )

    finally:
        EXTRACT.write_text(source, encoding="utf-8")

    # ============================================================
    # FINAL INTEGRITY CHECK
    # The mutations have been restored automatically.
    # ============================================================

    print("\n=== Final post-mutation integrity run ===")

    rc = run_test("app.tests.test_extract")

    if rc != 0:
        raise SystemExit(
            "Final extract suite failed after mutation restoration."
        )

    print("\n=== Full suite ===")

    rc = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "app/tests",
            "-t",
            ".",
            "-v",
        ]
    ).returncode

    if rc != 0:
        raise SystemExit(
            "Full suite failed after mutation restoration."
        )

    print("\nALL MUTATION CHECKS PASSED.")
    print(
        "Both deliberate security regressions were detected."
    )
    print(
        "Original app/extract.py was restored before final verification."
    )


if __name__ == "__main__":
    main()