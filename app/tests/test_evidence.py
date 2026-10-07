import unittest

from app.evidence import (
    EvidenceConflictError,
    EvidenceRecord,
    EvidenceStore,
    EvidenceValidationError,
    make_record_id,
    verify_record,
)


SOURCE = "Alice works in Mumbai. Her salary is 1200000 and she joined in 2022."


class EvidenceTests(unittest.TestCase):
    def record(self, *, verified=False, span=None, quotes=None, fields=None):
        return EvidenceRecord(
            id=make_record_id("doc-1", span or (0, len(SOURCE)), "employee", "salary"),
            doc_id="doc-1",
            span=span or (0, len(SOURCE)),
            entity="Alice",
            schema="employee",
            fields=fields or {"salary": 1200000, "joining_year": 2022},
            quotes=quotes or {
                "salary": "salary is 1200000",
                "joining_year": "joined in 2022",
            },
            period="2026",
            extractor="test-model@v1",
            verified=verified,
        )

    def test_valid_record_is_verified_by_host(self):
        verified = verify_record(self.record(), SOURCE)
        self.assertTrue(verified.verified)

    def test_store_only_contains_verified_records(self):
        store = EvidenceStore()
        with self.assertRaises(EvidenceValidationError):
            store.add_verified(self.record())
        stored = store.add(self.record(), SOURCE)
        self.assertTrue(stored.verified)
        self.assertEqual(len(store), 1)
        self.assertIs(store.get(stored.id), stored)

    def test_quote_must_be_inside_declared_span(self):
        # Salary quote is outside the declared span.
        bad = self.record(span=(0, SOURCE.index("Her salary")))
        with self.assertRaises(EvidenceValidationError):
            verify_record(bad, SOURCE)

    def test_non_null_field_requires_quote(self):
        bad = self.record(quotes={"salary": "salary is 1200000"})
        with self.assertRaises(EvidenceValidationError):
            bad.validate_structure()

    def test_null_field_does_not_need_quote(self):
        good = self.record(fields={"salary": 1200000, "bonus": None}, quotes={"salary": "salary is 1200000"})
        verified = verify_record(good, SOURCE)
        self.assertTrue(verified.verified)

    def test_quote_is_verbatim(self):
        bad = self.record(quotes={"salary": "Salary is 1200000"})
        with self.assertRaises(EvidenceValidationError):
            verify_record(bad, SOURCE)

    def test_duplicate_same_record_is_idempotent(self):
        store = EvidenceStore()
        record = store.add(self.record(), SOURCE)
        again = store.add(record, SOURCE)
        self.assertEqual(record, again)
        self.assertEqual(len(store), 1)

    def test_duplicate_id_with_different_content_is_rejected(self):
        store = EvidenceStore()
        store.add(self.record(), SOURCE)
        different = self.record(fields={"salary": 999999, "joining_year": 2022}, quotes={
            "salary": "salary is 1200000", "joining_year": "joined in 2022"
        })
        with self.assertRaises(EvidenceConflictError):
            store.add(different, SOURCE)

    def test_lookup_by_document(self):
        store = EvidenceStore()
        record = store.add(self.record(), SOURCE)
        self.assertEqual(store.for_doc("doc-1"), (record,))
        self.assertEqual(store.for_doc("missing"), ())


if __name__ == "__main__":
    unittest.main()
