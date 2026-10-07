import unittest

from app.code_contract import ContractValidationError, canonical_json, validate_output


RECORDS = [
    {"id": "r1", "value": 10, "salary": 200},
    {"id": "r2", "value": 20, "salary": None},
    {"id": "r3", "value": 30, "salary": 250},
]


class ContractTests(unittest.TestCase):
    def valid(self):
        return {
            "result": {"sum": 60},
            "inputs_used": ["r1", "r3"],
            "excluded": [
                {
                    "record_ids": ["r2"],
                    "reason_code": "MISSING_FIELD",
                    "reason": "salary is missing",
                }
            ],
            "assumptions": [],
            "fields_used": {
                "r1": ["value"],
                "r3": ["value"],
            },
        }

    def assert_invalid(self, output, expected=None, records=RECORDS):
        with self.assertRaises(ContractValidationError) as cm:
            validate_output(output, records)
        if expected:
            self.assertIn(expected, str(cm.exception))

    def test_valid_contract(self):
        result = validate_output(self.valid(), RECORDS)
        self.assertTrue(result.canonical_result())
        self.assertEqual(result.inputs_used, ("r1", "r3"))
        self.assertEqual(result.excluded_ids(), ("r2",))

    def test_exact_output_keys_required(self):
        output = self.valid()
        del output["fields_used"]
        self.assert_invalid(output, "missing keys")

    def test_unknown_used_id(self):
        output = self.valid()
        output["inputs_used"] = ["r1", "rx"]
        self.assert_invalid(output, "unknown")

    def test_unknown_excluded_id(self):
        output = self.valid()
        output["excluded"][0]["record_ids"] = ["rx"]
        self.assert_invalid(output, "unknown")

    def test_overlap_used_and_excluded(self):
        output = self.valid()
        output["excluded"][0]["record_ids"] = ["r1"]
        self.assert_invalid(output, "both")

    def test_unaccounted_record(self):
        output = self.valid()
        output["excluded"] = []
        self.assert_invalid(output, "unaccounted")

    def test_duplicate_input_ids_are_rejected_before_partition(self):
        records = [{"id": "r1", "value": 1}, {"id": "r1", "value": 2}]
        self.assert_invalid(self.valid(), "duplicated", records)

    def test_null_declared_field_is_rejected(self):
        output = self.valid()
        output["inputs_used"] = ["r2", "r3"]
        output["excluded"] = [
            {
                "record_ids": ["r1"],
                "reason_code": "OUT_OF_SCOPE",
                "reason": "not selected",
            }
        ]
        output["fields_used"] = {"r2": ["salary"], "r3": ["value"]}
        self.assert_invalid(output, "null field")

    def test_missing_declared_field_is_rejected(self):
        output = self.valid()
        output["fields_used"]["r1"] = ["not_there"]
        self.assert_invalid(output, "missing field")

    def test_fields_used_must_belong_to_inputs_used(self):
        output = self.valid()
        output["fields_used"]["r2"] = []
        self.assert_invalid(output, "not in inputs_used")

    def test_empty_exclusion_is_rejected(self):
        output = self.valid()
        output["excluded"][0]["record_ids"] = []
        self.assert_invalid(output, "cannot be empty")

    def test_bad_reason_code_is_rejected(self):
        output = self.valid()
        output["excluded"][0]["reason_code"] = "MADE_UP"
        self.assert_invalid(output, "reason_code")

    def test_single_id_legacy_shape_is_rejected(self):
        output = self.valid()
        output["excluded"][0].pop("record_ids")
        output["excluded"][0]["record_id"] = "r2"
        self.assert_invalid(output, "record_ids")

    def test_strict_json_rejects_nan(self):
        output = self.valid()
        output["result"] = float("nan")
        self.assert_invalid(output, "not strict JSON")

    def test_assumptions_are_strings_and_bounded(self):
        output = self.valid()
        output["assumptions"] = [123]
        self.assert_invalid(output, "assumptions")

    def test_canonical_json_is_stable(self):
        a = {"b": 2, "a": [3, 1]}
        b = {"a": [3, 1], "b": 2}
        self.assertEqual(canonical_json(a), canonical_json(b))

    def test_grouped_exclusions_are_canonicalized(self):
        output = self.valid()
        output["excluded"] = [
            {
                "record_ids": ["r2"],
                "reason_code": "MISSING_FIELD",
                "reason": "salary is missing",
            }
        ]
        result = validate_output(output, RECORDS)
        self.assertEqual(result.excluded[0]["record_ids"], ["r2"])


if __name__ == "__main__":
    unittest.main()
