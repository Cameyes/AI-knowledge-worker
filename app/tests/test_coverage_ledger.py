import unittest

from app.coverage_ledger import CoverageError, CoverageEntry, CoverageLedger


class CoverageLedgerTests(unittest.TestCase):
    def ledger(self):
        return CoverageLedger.for_documents(
            ["d1", "d2", "d3", "d4"],
            candidate_basis="exhaustive",
            basis_detail="all 4 documents",
        )

    def test_new_candidates_start_pending(self):
        ledger = self.ledger()
        self.assertEqual(ledger.counts(), {
            "pending": 4, "extracted": 0, "no_match": 0, "failed": 0, "skipped": 0
        })
        self.assertFalse(ledger.complete())
        self.assertEqual(ledger.fraction_examined(), 0.0)

    def test_extracted_and_no_match_count_as_examined(self):
        ledger = self.ledger()
        ledger.mark_extracted("d1")
        ledger.mark_no_match("d2", "no relevant employee record")
        self.assertEqual(ledger.fraction_examined(), 0.5)
        self.assertFalse(ledger.complete())

    def test_failed_and_pending_make_ledger_incomplete(self):
        ledger = self.ledger()
        ledger.mark_extracted("d1")
        ledger.mark_no_match("d2", "not present")
        ledger.mark_failed("d3", "extractor timeout")
        ledger.mark_skipped("d4", "excluded by explicit user filter")
        self.assertFalse(ledger.complete())
        self.assertEqual(ledger.fraction_examined(), 0.5)

    def test_complete_has_no_pending_or_failed(self):
        ledger = self.ledger()
        ledger.mark_extracted("d1")
        ledger.mark_no_match("d2", "not present")
        ledger.mark_skipped("d3", "outside requested period")
        ledger.mark_extracted("d4")
        self.assertTrue(ledger.complete())
        self.assertEqual(ledger.fraction_examined(), 0.75)

    def test_missing_document_cannot_be_marked(self):
        ledger = self.ledger()
        with self.assertRaises(CoverageError):
            ledger.mark_extracted("unknown")

    def test_reason_is_required_for_negative_states(self):
        ledger = self.ledger()
        with self.assertRaises(CoverageError):
            ledger.mark_failed("d1", "")
        with self.assertRaises(CoverageError):
            CoverageEntry("d1", "pending", "should not be here").validate()

    def test_retry_can_return_failed_entry_to_pending(self):
        ledger = self.ledger()
        ledger.mark_failed("d1", "temporary extractor failure")
        ledger.mark_pending("d1")
        self.assertFalse(ledger.complete())
        self.assertEqual(ledger.get("d1").state, "pending")

    def test_empty_candidate_set_is_complete_and_fully_examined(self):
        ledger = CoverageLedger.for_documents([], candidate_basis="exhaustive", basis_detail="empty KB")
        self.assertTrue(ledger.complete())
        self.assertEqual(ledger.fraction_examined(), 1.0)

    def test_duplicate_documents_rejected(self):
        with self.assertRaises(CoverageError):
            CoverageLedger.for_documents(["d1", "d1"], candidate_basis="exhaustive", basis_detail="bad")

    def test_search_selected_preserves_basis(self):
        ledger = CoverageLedger.for_documents(
            ["d1"], candidate_basis="search_selected", basis_detail="query='salary' top_k=10"
        )
        self.assertEqual(ledger.candidate_basis, "search_selected")
        self.assertEqual(ledger.basis_detail, "query='salary' top_k=10")


if __name__ == "__main__":
    unittest.main()
