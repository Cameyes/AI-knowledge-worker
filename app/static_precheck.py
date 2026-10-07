"""Bounded, deterministic AST policy gate for LLM-generated computation code.

This module is intentionally NOT the security boundary. It only performs cheap,
host-side rejection of malformed or obviously unsafe code before the code reaches
Sandbox v3. It never compiles or executes user code.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Literal


PrecheckCode = Literal[
    "empty_source",
    "source_too_large",
    "syntax_error",
    "parse_resource_limit",
    "ast_too_large",
    "ast_too_deep",
    "top_level_statement_forbidden",
    "decorator_forbidden",
    "async_forbidden",
    "solve_missing",
    "solve_not_function",
    "solve_duplicate",
    "solve_signature",
    "import_forbidden",
    "import_wildcard_forbidden",
    "dunder_forbidden",
    "call_forbidden",
    "nondeterministic_attribute",
]


@dataclass(frozen=True)
class PrecheckResult:
    ok: bool
    code: str
    message: str
    line: int | None = None
    column: int | None = None

    @classmethod
    def accept(cls) -> "PrecheckResult":
        return cls(True, "ok", "code passed static pre-check")

    @classmethod
    def reject(
        cls,
        code: str,
        message: str,
        node: ast.AST | None = None,
    ) -> "PrecheckResult":
        return cls(
            False,
            code,
            message,
            getattr(node, "lineno", None),
            getattr(node, "col_offset", None),
        )


MAX_SOURCE_CHARS = 200_000
MAX_AST_NODES = 100_000
MAX_AST_DEPTH = 200

# v1 intentionally exposes only deterministic standard-library functionality.
ALLOWED_IMPORT_ROOTS = frozenset(
    {
        "math",
        "statistics",
        "decimal",
        "fractions",
        "datetime",
        "calendar",
        "re",
        "json",
        "collections",
        "itertools",
        "functools",
        "operator",
        "typing",
        "copy",
    }
)

FORBIDDEN_CALL_NAMES = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "input",
        "breakpoint",
    }
)

NONDETERMINISTIC_ATTRIBUTES = frozenset({"now", "today", "utcnow"})

# Deliberately allowed at module scope only when assigned from literals.
_ALLOWED_TOP_LEVEL = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Assign, ast.AnnAssign)


def _root_module(module: str) -> str:
    return module.split(".", 1)[0]


def _is_literal_tree(node: ast.AST) -> bool:
    """Return True only for assignments whose RHS is structurally constant."""
    allowed = (
        ast.Constant,
        ast.Tuple,
        ast.List,
        ast.Set,
        ast.Dict,
        ast.UnaryOp,
        ast.BinOp,
    )
    if not isinstance(node, allowed):
        return False
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            # No names: top-level constants cannot execute arbitrary lookups.
            return False
        if isinstance(child, (ast.Call, ast.Attribute, ast.Subscript, ast.Lambda, ast.comprehension)):
            return False
    return True


def _check_ast_size_and_depth(tree: ast.AST) -> PrecheckResult | None:
    stack: list[tuple[ast.AST, int]] = [(tree, 1)]
    count = 0
    while stack:
        node, depth = stack.pop()
        count += 1
        if count > MAX_AST_NODES:
            return PrecheckResult.reject("ast_too_large", f"AST node count exceeds {MAX_AST_NODES}", node)
        if depth > MAX_AST_DEPTH:
            return PrecheckResult.reject("ast_too_deep", f"AST nesting exceeds {MAX_AST_DEPTH}", node)
        for child in ast.iter_child_nodes(node):
            stack.append((child, depth + 1))
    return None


def _check_top_level(tree: ast.Module) -> PrecheckResult | None:
    for index, node in enumerate(tree.body):
        if index == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue

        if isinstance(node, ast.AsyncFunctionDef):
            return PrecheckResult.reject("async_forbidden", "async functions are not allowed", node)

        if not isinstance(node, _ALLOWED_TOP_LEVEL):
            return PrecheckResult.reject(
                "top_level_statement_forbidden",
                f"top-level {type(node).__name__} is not allowed; keep executable logic inside solve()",
                node,
            )

        if isinstance(node, ast.Assign):
            if not _is_literal_tree(node.value):
                return PrecheckResult.reject(
                    "top_level_statement_forbidden",
                    "top-level assignments must contain literals only",
                    node,
                )
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            if not _is_literal_tree(node.value):
                return PrecheckResult.reject(
                    "top_level_statement_forbidden",
                    "top-level annotated assignments must contain literals only",
                    node,
                )

        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            return PrecheckResult.reject("decorator_forbidden", "function decorators are not allowed", node)

    return None


def _check_solve(tree: ast.Module) -> PrecheckResult | None:
    solves = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "solve"]
    assignments = [node for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))]

    # A module-level assignment named solve is also a contract violation.
    for node in assignments:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets.extend(node.targets)
        else:
            targets.append(node.target)
        if any(isinstance(target, ast.Name) and target.id == "solve" for target in targets):
            return PrecheckResult.reject("solve_not_function", "solve must be a module-level function", node)

    if not solves:
        return PrecheckResult.reject("solve_missing", "module must define solve(records, extra)")
    if len(solves) != 1:
        return PrecheckResult.reject("solve_duplicate", "module must define exactly one solve() function")

    solve = solves[0]
    args = solve.args
    if args.posonlyargs or args.vararg or args.kwonlyargs or args.kwarg or args.defaults or args.kw_defaults:
        return PrecheckResult.reject(
            "solve_signature",
            "solve must have exactly two positional parameters with no defaults, *args, or **kwargs",
            solve,
        )
    if len(args.args) != 2 or [a.arg for a in args.args] != ["records", "extra"]:
        return PrecheckResult.reject("solve_signature", "solve must have signature solve(records, extra)", solve)
    if solve.decorator_list:
        return PrecheckResult.reject("decorator_forbidden", "solve decorators are not allowed", solve)
    return None


def _check_import(node: ast.Import | ast.ImportFrom) -> PrecheckResult | None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            root = _root_module(alias.name)
            if root not in ALLOWED_IMPORT_ROOTS:
                return PrecheckResult.reject(
                    "import_forbidden",
                    f"import '{alias.name}' is not allowed",
                    node,
                )
    else:
        if node.level != 0:
            return PrecheckResult.reject("import_forbidden", "relative imports are not allowed", node)
        module = node.module or ""
        if _root_module(module) not in ALLOWED_IMPORT_ROOTS:
            return PrecheckResult.reject(
                "import_forbidden",
                f"import from '{module}' is not allowed",
                node,
            )
        if any(alias.name == "*" for alias in node.names):
            return PrecheckResult.reject("import_wildcard_forbidden", "wildcard imports are not allowed", node)
    return None


def _check_forbidden_names(tree: ast.AST) -> PrecheckResult | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute)):
            name = node.id if isinstance(node, ast.Name) else node.attr
            if name.startswith("__") or name.endswith("__"):
                return PrecheckResult.reject(
                    "dunder_forbidden",
                    f"dunder access/name '{name}' is not allowed",
                    node,
                )

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALL_NAMES:
                return PrecheckResult.reject(
                    "call_forbidden",
                    f"call to '{node.func.id}' is not allowed",
                    node,
                )
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in FORBIDDEN_CALL_NAMES:
                    return PrecheckResult.reject(
                        "call_forbidden",
                        f"call to attribute '{node.func.attr}' is not allowed",
                        node,
                    )

        if isinstance(node, ast.Attribute) and node.attr in NONDETERMINISTIC_ATTRIBUTES:
            return PrecheckResult.reject(
                "nondeterministic_attribute",
                f"'.{node.attr}()' is not allowed because it depends on wall-clock state",
                node,
            )

        if isinstance(node, ast.AsyncFunctionDef) or isinstance(node, (ast.Await, ast.AsyncFor, ast.AsyncWith)):
            return PrecheckResult.reject("async_forbidden", "async execution is not allowed", node)

    return None


def precheck_code(code: str) -> PrecheckResult:
    """Validate generated source without compiling or executing it."""
    if not isinstance(code, str) or not code.strip():
        return PrecheckResult.reject("empty_source", "generated code must be non-empty")
    if len(code) > MAX_SOURCE_CHARS:
        return PrecheckResult.reject(
            "source_too_large",
            f"generated code exceeds the {MAX_SOURCE_CHARS}-character pre-check limit",
        )

    try:
        tree = ast.parse(code, mode="exec")
    except (SyntaxError, ValueError) as exc:
        message = "syntax error" if isinstance(exc, SyntaxError) else "source contains invalid characters"
        return PrecheckResult.reject("syntax_error", f"{message}: {exc}")
    except (RecursionError, MemoryError, TypeError) as exc:
        return PrecheckResult.reject(
            "parse_resource_limit",
            f"AST parsing refused the source safely: {type(exc).__name__}",
        )

    result = _check_ast_size_and_depth(tree)
    if result:
        return result

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            result = _check_import(node)
            if result:
                return result

    result = _check_top_level(tree)
    if result:
        return result

    result = _check_solve(tree)
    if result:
        return result

    result = _check_forbidden_names(tree)
    if result:
        return result

    return PrecheckResult.accept()


__all__ = [
    "ALLOWED_IMPORT_ROOTS",
    "MAX_AST_DEPTH",
    "MAX_AST_NODES",
    "MAX_SOURCE_CHARS",
    "PrecheckResult",
    "precheck_code",
]
