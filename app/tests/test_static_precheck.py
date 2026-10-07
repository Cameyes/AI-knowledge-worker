import unittest

from app.static_precheck import precheck_code


VALID = '''
"""generic computation"""
from decimal import Decimal
import math

LIMIT = 100


def helper(x):
    return x + 1


def solve(records, extra):
    values = [r["value"] for r in records if r["value"] is not None]
    total = math.fsum(values)
    return {
        "result": {"total": total, "limit": LIMIT},
        "inputs_used": [r["id"] for r in records],
        "excluded": [],
        "assumptions": [],
        "fields_used": {r["id"]: ["value"] for r in records},
    }
'''


class StaticPrecheckTests(unittest.TestCase):
    def assert_rejects(self, code, expected):
        result = precheck_code(code)
        self.assertFalse(result.ok, result)
        self.assertEqual(result.code, expected)

    def test_valid_generic_code(self):
        self.assertTrue(precheck_code(VALID).ok)

    def test_missing_solve(self):
        self.assert_rejects("x = 1\n", "solve_missing")

    def test_wrong_signature(self):
        self.assert_rejects("def solve(records):\n    return {}\n", "solve_signature")
        self.assert_rejects("def solve(records, extra, more):\n    return {}\n", "solve_signature")
        self.assert_rejects("def solve(records, extra=None):\n    return {}\n", "solve_signature")
        self.assert_rejects("def solve(*args):\n    return {}\n", "solve_signature")

    def test_duplicate_solve(self):
        code = "def solve(records, extra):\n    return {}\ndef solve(records, extra):\n    return {}\n"
        self.assert_rejects(code, "solve_duplicate")

    def test_forbidden_imports(self):
        for module in ("os", "subprocess", "socket", "requests", "pathlib", "time", "random", "secrets", "uuid"):
            code = f"import {module}\ndef solve(records, extra):\n    return {{}}\n"
            self.assert_rejects(code, "import_forbidden")

    def test_wildcard_import(self):
        self.assert_rejects(
            "from math import *\ndef solve(records, extra):\n    return {}\n",
            "import_wildcard_forbidden",
        )

    def test_forbidden_calls(self):
        for call in (
            "eval('1')",
            "exec('x=1')",
            "compile('x=1', '<x>', 'exec')",
            "__import__('os')",
            "open('x')",
            "getattr({}, 'x')",
            "setattr({}, 'x', 1)",
            "delattr({}, 'x')",
            "globals()",
            "locals()",
            "vars()",
            "input()",
            "breakpoint()",
        ):
            code = f"def solve(records, extra):\n    x = {call}\n    return {{}}\n"
            expected = "call_forbidden"
            self.assert_rejects(code, expected)

    def test_dunder_access(self):
        for expr in ("x.__class__", "x.__globals__", "x.__subclasses__", "x.__mro__"):
            code = f"def solve(records, extra):\n    x = {expr}\n    return {{}}\n"
            self.assert_rejects(code, "dunder_forbidden")

    def test_time_access(self):
        for expr in ("datetime.datetime.now()", "datetime.date.today()", "datetime.datetime.utcnow()"):
            code = "import datetime\ndef solve(records, extra):\n    x = " + expr + "\n    return {}\n"
            self.assert_rejects(code, "nondeterministic_attribute")

    def test_top_level_executable_statement_is_rejected(self):
        code = "print('bad')\ndef solve(records, extra):\n    return {}\n"
        self.assert_rejects(code, "top_level_statement_forbidden")

    def test_decorators_are_rejected(self):
        code = "@staticmethod\ndef solve(records, extra):\n    return {}\n"
        self.assert_rejects(code, "decorator_forbidden")

    def test_async_is_rejected(self):
        self.assert_rejects("async def solve(records, extra):\n    return {}\n", "async_forbidden")

    def test_null_byte_is_rejected_without_execution(self):
        result = precheck_code("def solve(records, extra):\x00\n    return {}\n")
        self.assertFalse(result.ok)
        self.assertIn(result.code, {"syntax_error", "parse_resource_limit"})

    def test_source_size_is_bounded(self):
        result = precheck_code("#" * 200_001)
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "source_too_large")

    def test_open_word_as_data_is_not_rejected(self):
        code = '''
def solve(records, extra):
    return {
        "result": [r.get("open", None) for r in records],
        "inputs_used": [],
        "excluded": [],
        "assumptions": [],
        "fields_used": {},
    }
'''
        self.assertTrue(precheck_code(code).ok)


if __name__ == "__main__":
    unittest.main()
