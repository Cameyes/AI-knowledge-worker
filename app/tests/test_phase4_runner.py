import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from app.phase4_runner import Phase4Executor


RECORDS = [
    {"id": "r1", "value": 10},
    {"id": "r2", "value": 20},
    {"id": "r3", "value": 30},
]

GOOD_CODE = '''
def solve(records, extra):
    used = [r for r in records if r["value"] >= 20]
    excluded = [r for r in records if r["value"] < 20]
    return {
        "result": {"sum": sum(r["value"] for r in used)},
        "inputs_used": [r["id"] for r in used],
        "excluded": [{
            "record_ids": [r["id"] for r in excluded],
            "reason_code": "FILTERED_BY_CRITERIA",
            "reason": "below threshold",
        }] if excluded else [],
        "assumptions": [],
        "fields_used": {r["id"]: ["value"] for r in used},
    }
'''

BAD_PARTITION_CODE = '''
def solve(records, extra):
    used = [r for r in records if r["value"] >= 20]
    excluded = [r for r in records if r["value"] < 20]
    return {
        "result": {"average": sum(r["value"] for r in records) / len(records)},
        "inputs_used": [r["id"] for r in used],
        "excluded": [{
            "record_ids": [r["id"] for r in excluded],
            "reason_code": "FILTERED_BY_CRITERIA",
            "reason": "below threshold",
        }] if excluded else [],
        "assumptions": [],
        "fields_used": {r["id"]: ["value"] for r in used},
    }
'''

NONDETERMINISTIC_CODE = '''
def solve(records, extra):
    order = list({"a", "b", "c", "d", "e", "f", "g", "h"})
    return {
        "result": {"order": order},
        "inputs_used": [r["id"] for r in records],
        "excluded": [],
        "assumptions": [],
        "fields_used": {r["id"]: [] for r in records},
    }
'''


@dataclass
class Request:
    code: str
    records: list
    extra: dict


class FakeBackend:
    def __init__(self, fn):
        self.fn = fn
        self.requests = []

    def run(self, request):
        self.requests.append(request)
        return SimpleNamespace(status="ok", result_json=self.fn(request.code, request.records, request.extra), stderr="", detail="")


def factory(code, records, extra, limits):
    return Request(code=code, records=list(records), extra=dict(extra))


class RunnerTests(unittest.TestCase):
    def make_executor(self, fn):
        backend = FakeBackend(fn)
        executor = Phase4Executor(backend, limits=object(), request_factory=factory)
        return executor, backend

    def test_three_runs_and_success(self):
        def fn(code, records, extra):
            used = [r for r in records if r["value"] >= 20]
            excluded = [r for r in records if r["value"] < 20]
            return {
                "result": {"sum": sum(r["value"] for r in used)},
                "inputs_used": [r["id"] for r in used],
                "excluded": [{
                    "record_ids": [r["id"] for r in excluded],
                    "reason_code": "FILTERED_BY_CRITERIA",
                    "reason": "below threshold",
                }] if excluded else [],
                "assumptions": [],
                "fields_used": {r["id"]: ["value"] for r in used},
            }

        executor, backend = self.make_executor(fn)
        result = executor.execute(GOOD_CODE, RECORDS)
        self.assertTrue(result.ok, result)
        self.assertEqual(result.sandbox_runs, 3)
        self.assertEqual(len(backend.requests), 3)
        self.assertEqual([r["id"] for r in backend.requests[2].records], ["r2", "r3"])

    def test_partition_failure_caught(self):
        def fn(code, records, extra):
            if len(records) == 3:
                return {
                    "result": {"average": sum(r["value"] for r in records) / len(records)},
                    "inputs_used": ["r2", "r3"],
                    "excluded": [{
                        "record_ids": ["r1"],
                        "reason_code": "FILTERED_BY_CRITERIA",
                        "reason": "below threshold",
                    }],
                    "assumptions": [],
                    "fields_used": {"r2": ["value"], "r3": ["value"]},
                }
            return {
                "result": {"average": sum(r["value"] for r in records) / len(records)},
                "inputs_used": [r["id"] for r in records],
                "excluded": [],
                "assumptions": [],
                "fields_used": {r["id"]: ["value"] for r in records},
            }

        executor, _ = self.make_executor(fn)
        result = executor.execute(BAD_PARTITION_CODE, RECORDS)
        self.assertFalse(result.ok)
        self.assertEqual(result.failure.stage, "metamorphic")
        self.assertEqual(result.failure.code, "exclusion_changed_result")
        self.assertIn("r1", result.failure.hint)

    def test_static_precheck_failure_uses_zero_sandbox_runs(self):
        executor, backend = self.make_executor(lambda *_: {})
        result = executor.execute("import os\ndef solve(records, extra):\n    return {}\n", RECORDS)
        self.assertFalse(result.ok)
        self.assertEqual(result.failure.stage, "precheck")
        self.assertEqual(result.sandbox_runs, 0)
        self.assertEqual(len(backend.requests), 0)

    def test_contract_failure_before_replays(self):
        def fn(code, records, extra):
            return {
                "result": 1,
                "inputs_used": ["r1"],
                "excluded": [],
                "assumptions": [],
                "fields_used": {"r1": ["value"]},
            }

        executor, backend = self.make_executor(fn)
        result = executor.execute(GOOD_CODE, RECORDS)
        self.assertFalse(result.ok)
        self.assertEqual(result.failure.stage, "contract")
        self.assertEqual(result.sandbox_runs, 1)
        self.assertEqual(len(backend.requests), 1)


if __name__ == "__main__":
    unittest.main()
