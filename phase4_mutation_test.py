"""Small mutation checks for Phase 4 host invariants.

These mutations target the implementation mechanisms rather than Docker. The
sandbox adversarial tests separately exercise Sandbox v3 directly.
"""
from __future__ import annotations

import copy
import unittest

from app.code_contract import ContractValidationError, validate_output
from app.static_precheck import precheck_code


RECORDS = [
    {"id": "r1", "value": 10},
    {"id": "r2", "value": None},
]


def valid_output():
    return {
        "result": 10,
        "inputs_used": ["r1"],
        "excluded": [
            {"record_ids": ["r2"], "reason_code": "MISSING_FIELD", "reason": "value missing"}
        ],
        "assumptions": [],
        "fields_used": {"r1": ["value"]},
    }


class MutationTests(unittest.TestCase):
    def test_mutation_drop_import_allowlist_is_killed(self):
        bad = "import os\ndef solve(records, extra):\n    return {}\n"
        self.assertFalse(precheck_code(bad).ok)

    def test_mutation_drop_dunder_ban_is_killed(self):
        bad = "def solve(records, extra):\n    return {\"result\": (1).__class__.__name__}\n"
        self.assertFalse(precheck_code(bad).ok)
        self.assertEqual(precheck_code(bad).code, "dunder_forbidden")

    def test_mutation_skip_null_declared_field_is_killed(self):
        out = valid_output()
        out["fields_used"] = {"r2": ["value"]}
        with self.assertRaises(ContractValidationError):
            validate_output(out, RECORDS)

    def test_mutation_skip_duplicate_input_detection_is_killed(self):
        dup_records = copy.deepcopy(RECORDS)
        dup_records.append({"id": "r1", "value": 20})
        with self.assertRaises(ContractValidationError):
            validate_output(valid_output(), dup_records)


if __name__ == "__main__":
    unittest.main()
