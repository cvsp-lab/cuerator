"""AST-based symbolic-form verifier.

Rejects formulations that won't fit in a paper as a single closed-form
equation. Enforces (a) small total function count, (b) bounded body length,
(c) no control flow, (d) restricted numeric literals.

The constants below are the policy. Keep them strict so the search produces
formulas with no hidden hyperparameters.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

MAX_FUNCTIONS = 7                 # 2 main + up to 5 primitive helpers
MAX_MAIN_STMTS = 18                # excluding leading docstring (params_to_thresholds*)
MAX_HELPER_STMTS = 8               # per primitive helper — keeps each primitive expressible
                                    # as a one-line LaTeX equation (with up to a few intermediate
                                    # bindings for top-2 indexing tricks, learned scale/shift,
                                    # etc.)
MAX_HELPER_HSUM = 16               # total statement count across all helpers — caps the
                                    # combined primitive complexity (paper-friendly)

# Float literals that are allowed inside function bodies. Any other float
# constant must come from `params` (i.e., be a learned coefficient).
ALLOWED_FLOATS = frozenset({-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0})

# Integer constants are allowed as long as |n| <= INT_BOUND. This covers
# dim=, k=, index, view(-1, 1, 1) etc. without enabling magic weights.
INT_BOUND = 16

# Numerical-stability epsilon: any float with abs <= EPS_THRESHOLD is allowed.
EPS_THRESHOLD = 1e-3

REQUIRED_FUNCTIONS = ("params_to_thresholds", "params_to_thresholds_batch")

# Module-level assignment names whose RHS is exempt from constant checks
# (these are the public interface declarations).
EXEMPT_MODULE_NAMES = frozenset(
    {"NAME", "DESCRIPTION", "NUM_PARAMS", "PARAM_RANGES", "PARAM_NAMES"}
)

FORBIDDEN_NODES = (
    ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler,
    ast.With, ast.AsyncFor, ast.AsyncWith, ast.AsyncFunctionDef,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
    ast.Lambda, ast.Yield, ast.YieldFrom, ast.Await,
    ast.Global, ast.Nonlocal, ast.Raise, ast.Assert,
)

# Top-level import packages forbidden — these enable smuggling complexity from
# other formulation files at runtime (e.g., `_F036 = _load_formulation("f036.py")`),
# which defeats the whole symbolic-form constraint.
FORBIDDEN_IMPORT_ROOTS = frozenset({"importlib", "imp", "runpy"})

# Names forbidden anywhere they are referenced — dynamic-execution primitives.
FORBIDDEN_NAMES = frozenset(
    {"__import__", "exec", "eval", "compile", "globals", "locals", "vars", "open"}
)


def _check_constant(node: ast.Constant) -> Optional[str]:
    v = node.value
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if isinstance(v, int):
        if abs(v) <= INT_BOUND:
            return None
        return f"int constant {v} at line {node.lineno} exceeds bound ±{INT_BOUND}"
    fv = float(v)
    if fv in ALLOWED_FLOATS:
        return None
    if abs(fv) <= EPS_THRESHOLD:
        return None
    allowed_str = sorted(ALLOWED_FLOATS)
    return (
        f"float constant {v} at line {node.lineno} not in allowed set {allowed_str} "
        f"(eps tolerance: |x|<={EPS_THRESHOLD}). Use a learned param instead."
    )


def _strip_docstring(body):
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _check_no_inner_params(tree: ast.AST) -> Optional[str]:
    """Reject formulations where `params[k]` (a learned coefficient) is passed
    as an argument to any function call.

    The intent is to keep parameters as **outer weights only** — i.e., `params[k]`
    may appear in `params[k] * feature` or `params[0] + ...` expressions (BinOp),
    but never inside a primitive helper (e.g., `_softmax_carrier(sim, params[4])`)
    or a builtin (e.g., `torch.relu(a_sim - params[4])`). Inner params introduce
    non-linear coupling between params and features that REINFORCE has trouble
    optimising jointly with the outer weights — empirically, formulations with
    inner params consistently underperform outer-only formulations on this task.

    Tensor reshapes/views on a `params[...]` subscript are still allowed
    (`params[:, 1].view(-1, 1, 1)`), since they only restructure the param's
    shape for broadcasting — the param continues to act as an outer weight.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Allow tensor-reshape methods called *on* a params subscript
        # (`params[:, k].view(...)`, `params[:, k].unsqueeze(...)` etc.).
        # These do not pass params as an argument; they only reshape it.
        if isinstance(node.func, ast.Attribute):
            recv = node.func.value
            if (
                isinstance(recv, ast.Subscript)
                and isinstance(recv.value, ast.Name)
                and recv.value.id == "params"
            ):
                # Still need to check the args of this method call don't
                # smuggle a separate params subscript.
                for arg in node.args + [k.value for k in node.keywords]:
                    for sub in ast.walk(arg):
                        if (
                            isinstance(sub, ast.Subscript)
                            and isinstance(sub.value, ast.Name)
                            and sub.value.id == "params"
                        ):
                            line = getattr(node, "lineno", "?")
                            return (
                                f"line {line}: a `params[...]` subscript appears as "
                                f"an argument to a method call. Inner-param usage is "
                                f"forbidden — `params[k]` may only appear as a direct "
                                f"outer weight or bias (e.g., `params[k] * feature`)."
                            )
                continue
        # General case: forbid params subscript anywhere in the args of any call.
        for arg in node.args + [k.value for k in node.keywords]:
            for sub in ast.walk(arg):
                if (
                    isinstance(sub, ast.Subscript)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == "params"
                ):
                    line = getattr(node, "lineno", "?")
                    return (
                        f"line {line}: function call has `params[...]` (or an "
                        f"expression containing it) as an argument. Inner-param "
                        f"usage is forbidden — `params[k]` must only appear as a "
                        f"direct outer weight or bias in the threshold expression "
                        f"(e.g., `params[k] * feature`, `params[0] + ...`), not as "
                        f"an argument to a primitive helper or builtin."
                    )
    return None


def _check_function(func: ast.FunctionDef, max_stmts: int) -> Optional[str]:
    body = _strip_docstring(func.body)
    if len(body) > max_stmts:
        kind = "main" if func.name in REQUIRED_FUNCTIONS else "helper"
        return (
            f"function '{func.name}' ({kind}) has {len(body)} statements "
            f"(max {max_stmts}). Move logic into a helper or simplify."
        )
    for node in ast.walk(func):
        if isinstance(node, FORBIDDEN_NODES):
            return (
                f"function '{func.name}' uses forbidden construct "
                f"{type(node).__name__} at line {getattr(node, 'lineno', '?')}"
            )
        if isinstance(node, ast.Constant):
            err = _check_constant(node)
            if err:
                return f"in function '{func.name}': {err}"
    return None


def verify_formulation(
    path: Path,
    *,
    max_functions: int = MAX_FUNCTIONS,
    max_main_stmts: int = MAX_MAIN_STMTS,
    max_helper_stmts: int = MAX_HELPER_STMTS,
    max_helper_hsum: int = MAX_HELPER_HSUM,
) -> Optional[str]:
    """Return None on success, or a human-readable error string on failure.

    The S1/S2 size limits default to the module constants but may be overridden
    (the orchestrator passes them from config.yaml's `verifier:` section so the
    prompt and this verifier stay in sync).
    """
    source = Path(path).read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"SyntaxError at line {e.lineno}: {e.msg}"

    func_defs_all = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if len(func_defs_all) > max_functions:
        names = [f.name for f in func_defs_all]
        return (
            f"too many function definitions: {len(func_defs_all)} > {max_functions} "
            f"(found: {names}). Allowed: 2 mains + up to {max_functions - 2} primitive helpers."
        )

    top_func_names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    missing = [n for n in REQUIRED_FUNCTIONS if n not in top_func_names]
    if missing:
        return f"missing required top-level functions: {missing}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_IMPORT_ROOTS:
                    return (
                        f"forbidden import '{alias.name}' at line {node.lineno}: "
                        f"dynamic loading of other files is not allowed."
                    )
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod in FORBIDDEN_IMPORT_ROOTS:
                return (
                    f"forbidden import 'from {node.module}' at line {node.lineno}: "
                    f"dynamic loading of other files is not allowed."
                )
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return (
                f"forbidden name '{node.id}' at line {node.lineno}: "
                f"dynamic-execution / file-IO primitives are not allowed."
            )

    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        target_names = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
        if any(t in EXEMPT_MODULE_NAMES for t in target_names):
            continue
        for node in ast.walk(stmt):
            if isinstance(node, ast.Constant):
                err = _check_constant(node)
                if err:
                    return f"module-level assignment to {target_names}: {err}"

    helper_stmt_total = 0
    for func in func_defs_all:
        is_main = func.name in REQUIRED_FUNCTIONS
        max_stmts = max_main_stmts if is_main else max_helper_stmts
        err = _check_function(func, max_stmts)
        if err:
            return err
        if not is_main:
            helper_stmt_total += len(_strip_docstring(func.body))

    if helper_stmt_total > max_helper_hsum:
        return (
            f"helpers' total statement count is {helper_stmt_total} "
            f"(max {max_helper_hsum}). Each primitive should be a "
            f"single closed-form expression — split or simplify."
        )

    err = _check_no_inner_params(tree)
    if err:
        return err

    return None


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Validate a formulation file.")
    parser.add_argument("path")
    args = parser.parse_args()
    err = verify_formulation(Path(args.path))
    if err is None:
        print("ok")
    else:
        print(f"REJECTED: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
