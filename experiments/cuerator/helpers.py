"""Helper CLI for cuerator/orchestrate.sh.

Usage:
    python -m experiments.cuerator.helpers <command> [args]
"""

import argparse
import json
import re
import sys

import yaml


ROLE_HEADERS = {
    "analysis", "apply patch", "apply_patch", "assistant", "codex",
    "commentary", "exec", "tool", "user",
}

SUMMARY_RE = re.compile(r"<!--\s*summary:\s*(.*?)\s*-->")


def _extract_summary(path):
    """Extract the last `<!-- summary: ... -->` marker from a markdown file."""
    if not path:
        return ""
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        return ""
    matches = SUMMARY_RE.findall(text)
    if not matches:
        return ""
    return matches[-1].strip()


def cmd_config_get(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    val = cfg
    for k in args.key.split("."):
        val = val.get(k, args.default) if isinstance(val, dict) else args.default
        if val is args.default:
            break
    print(val if val is not None else args.default)


def cmd_state_get(args):
    try:
        with open(args.state) as f:
            s = json.load(f)
        print(s.get(args.key, args.default))
    except FileNotFoundError:
        print(args.default)


def cmd_check_syntax(args):
    try:
        with open(args.file) as f:
            content = f.read()
        compile(content, args.file, "exec")
        print("ok")
    except FileNotFoundError:
        print(f"MissingFile:{args.file}")
    except SyntaxError as e:
        print(f"SyntaxError:{e.lineno}:{e.msg}")


def cmd_verify_formulation(args):
    """Verify a formulation file: Python compile + AST contract verifier.

    Stage 2's verify component. Combines:
      1. Python compile (catches syntax errors before any AST work)
      2. AST contract verification (function count, statement budgets,
         no control flow, allowed constants, etc.) — see verifier.py.

    Prints 'ok' on success, or 'REJECTED:<reason>' on any failure. The
    orchestrator treats anything other than 'ok' as a verify failure and
    retries the design agent (up to max_retries) with the rejection reason
    fed back via last_failure.txt.
    """
    from pathlib import Path

    from experiments.cuerator.verifier import verify_formulation

    try:
        with open(args.file) as f:
            content = f.read()
    except FileNotFoundError:
        print(f"REJECTED:MissingFile:{args.file}")
        return
    try:
        compile(content, args.file, "exec")
    except SyntaxError as e:
        print(f"REJECTED:SyntaxError:line {e.lineno}:{e.msg}")
        return
    # S1/S2 size limits come from config.yaml's `verifier:` section so the Design
    # prompt (rendered from the same keys) and this verifier stay in sync. Absent
    # keys fall back to the verifier's module defaults.
    limits = {}
    if getattr(args, "config", None):
        with open(args.config) as f:
            vcfg = (yaml.safe_load(f) or {}).get("verifier", {}) or {}
        if "max_helpers" in vcfg:
            limits["max_functions"] = vcfg["max_helpers"] + 2
        if "max_main_stmts" in vcfg:
            limits["max_main_stmts"] = vcfg["max_main_stmts"]
        if "max_helper_stmts" in vcfg:
            limits["max_helper_stmts"] = vcfg["max_helper_stmts"]
        if "max_helper_total" in vcfg:
            limits["max_helper_hsum"] = vcfg["max_helper_total"]
    err = verify_formulation(Path(args.file), **limits)
    if err is not None:
        print(f"REJECTED:{err}")
        return
    print("ok")


PROTOCOL_MAX_ABLATIONS = 6
POLICY_PROTOCOL_MAX_ANALYSES = 4
PROTOCOL_NUM_PARAMS = 10
PROTOCOL_NAME_RE = re.compile(r"^[A-Za-z0-9_]{1,32}$")


def cmd_verify_protocol(args):
    """Verify eval_protocol JSON against the schema. Print 'ok' or 'REJECTED:...'."""
    try:
        with open(args.file, encoding="utf-8") as f:
            doc = json.load(f)
    except FileNotFoundError:
        print(f"REJECTED:MissingFile:{args.file}")
        return
    except json.JSONDecodeError as e:
        print(f"REJECTED:JSONDecodeError:line {e.lineno}:{e.msg}")
        return

    if not isinstance(doc, dict):
        print("REJECTED:protocol root must be a JSON object")
        return
    if "ablations" not in doc:
        print("REJECTED:missing 'ablations' key")
        return
    abls = doc["ablations"]
    if not isinstance(abls, list):
        print("REJECTED:'ablations' must be a list")
        return
    if len(abls) > PROTOCOL_MAX_ABLATIONS:
        print(f"REJECTED:too many ablations ({len(abls)}; cap {PROTOCOL_MAX_ABLATIONS})")
        return
    seen_names = set()
    for i, a in enumerate(abls):
        if not isinstance(a, dict):
            print(f"REJECTED:ablations[{i}] is not an object")
            return
        for k in ("name", "params_zero", "intent"):
            if k not in a:
                print(f"REJECTED:ablations[{i}] missing key '{k}'")
                return
        name = a["name"]
        if not isinstance(name, str) or not PROTOCOL_NAME_RE.match(name):
            print(f"REJECTED:ablations[{i}].name must match /^[A-Za-z0-9_]{{1,32}}$/")
            return
        if name in seen_names:
            print(f"REJECTED:ablations[{i}].name duplicate '{name}'")
            return
        seen_names.add(name)
        pz = a["params_zero"]
        if not isinstance(pz, list) or len(pz) == 0:
            print(f"REJECTED:ablations[{i}].params_zero must be non-empty list")
            return
        seen_idx = set()
        for v in pz:
            if not isinstance(v, int) or isinstance(v, bool):
                print(f"REJECTED:ablations[{i}].params_zero must contain integers")
                return
            if v < 0 or v >= PROTOCOL_NUM_PARAMS:
                print(f"REJECTED:ablations[{i}].params_zero index {v} out of [0,{PROTOCOL_NUM_PARAMS - 1}]")
                return
            if v in seen_idx:
                print(f"REJECTED:ablations[{i}].params_zero has duplicate index {v}")
                return
            seen_idx.add(v)
        if len(pz) >= PROTOCOL_NUM_PARAMS:
            print(f"REJECTED:ablations[{i}].params_zero zeros all {PROTOCOL_NUM_PARAMS} params (no free dims)")
            return
        intent = a["intent"]
        if not isinstance(intent, str) or not intent.strip():
            print(f"REJECTED:ablations[{i}].intent must be a non-empty string")
            return
    cpa = doc.get("candidates_per_ablation", 8192)
    if not isinstance(cpa, int) or cpa <= 0:
        print("REJECTED:'candidates_per_ablation' must be a positive integer")
        return
    print("ok")


def cmd_verify_policy_protocol(args):
    """Verify policy_protocol JSON. Print 'ok' or 'REJECTED:...'."""
    try:
        with open(args.file, encoding="utf-8") as f:
            doc = json.load(f)
    except FileNotFoundError:
        print(f"REJECTED:MissingFile:{args.file}")
        return
    except json.JSONDecodeError as e:
        print(f"REJECTED:JSONDecodeError:line {e.lineno}:{e.msg}")
        return

    if not isinstance(doc, dict):
        print("REJECTED:protocol root must be a JSON object")
        return
    if "analyses" not in doc:
        print("REJECTED:missing 'analyses' key")
        return
    analyses = doc["analyses"]
    if not isinstance(analyses, list):
        print("REJECTED:'analyses' must be a list")
        return
    if len(analyses) > POLICY_PROTOCOL_MAX_ANALYSES:
        print(f"REJECTED:too many analyses ({len(analyses)}; cap {POLICY_PROTOCOL_MAX_ANALYSES})")
        return
    seen_names = set()
    for i, a in enumerate(analyses):
        if not isinstance(a, dict):
            print(f"REJECTED:analyses[{i}] is not an object")
            return
        for k in ("name", "intent"):
            if k not in a:
                print(f"REJECTED:analyses[{i}] missing key '{k}'")
                return
        name = a["name"]
        if not isinstance(name, str) or not PROTOCOL_NAME_RE.match(name):
            print(f"REJECTED:analyses[{i}].name must match /^[A-Za-z0-9_]{{1,32}}$/")
            return
        if name in seen_names:
            print(f"REJECTED:analyses[{i}].name duplicate '{name}'")
            return
        seen_names.add(name)
        intent = a["intent"]
        if not isinstance(intent, str) or not intent.strip():
            print(f"REJECTED:analyses[{i}].intent must be a non-empty string")
            return
    print("ok")


def _read_history_rows(history_path):
    rows = []
    try:
        with open(history_path) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        pass
    return rows


def cmd_record_result(args):
    """Append iteration result to history.jsonl and update state.json.

    history.jsonl row (agent-visible, test fields excluded) includes:
      iter, name, num_params, train_oracle_frame_acc,
      val_frame_acc, best_val_epoch,
      oracle_delta_3iter, val_best_streak, val_oracle_gap,
      plan_summary, oracle_summary, policy_summary
    """
    with open(args.result) as f:
        r = json.load(f)

    train_oracle_frame_acc = r["best_train"]["metrics"]["frame_acc"]
    plan_summary = _extract_summary(args.memo)
    oracle_summary = _extract_summary(args.oracle_report) if args.oracle_report else ""
    policy_summary = _extract_summary(args.policy_report) if args.policy_report else ""

    val_frame_acc = None
    best_val_epoch = None
    if args.policy_result and args.policy_result.strip():
        try:
            with open(args.policy_result) as f:
                pol = json.load(f)
            best = pol.get("best_val_metrics", {})
            val_frame_acc = best.get("frame_acc")
            best_val_epoch = pol.get("best_val_epoch")
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    # Derived trend metrics computed from prior history rows.
    prior_rows = _read_history_rows(args.history)

    # oracle_delta_3iter: change in oracle vs iter (current - 3); None if not enough history.
    oracle_delta_3iter = None
    if len(prior_rows) >= 3:
        ref = prior_rows[-3].get("train_oracle_frame_acc")
        if isinstance(ref, (int, float)):
            oracle_delta_3iter = round(train_oracle_frame_acc - ref, 4)

    # val_best_streak: number of iterations since the last time val_frame_acc strictly
    # improved over the running best. 0 means this iter set a new best.
    val_best_streak = None
    if isinstance(val_frame_acc, (int, float)):
        # Walk forward through prior rows to find the index of the last iter that set a new best.
        forward_best = float("-inf")
        last_improve_idx = -1
        for i, row in enumerate(prior_rows):
            v = row.get("val_frame_acc")
            if isinstance(v, (int, float)) and v > forward_best:
                forward_best = v
                last_improve_idx = i
        # Then check whether the current iter improves on that.
        current_idx = len(prior_rows)  # this iter's index after it's appended
        if val_frame_acc > forward_best:
            val_best_streak = 0
        elif last_improve_idx >= 0:
            val_best_streak = current_idx - last_improve_idx
        else:
            # No prior valid val_frame_acc, and current didn't beat -inf — should not happen,
            # but fall back to counting from the start.
            val_best_streak = current_idx + 1

    # val_oracle_gap: oracle - val (positive means oracle higher).
    val_oracle_gap = None
    if isinstance(val_frame_acc, (int, float)):
        val_oracle_gap = round(train_oracle_frame_acc - val_frame_acc, 4)

    summary = {
        "iter": args.iter,
        "name": r.get("name", ""),
        "num_params": r.get("num_params"),
        "train_oracle_frame_acc": round(train_oracle_frame_acc, 4),
        "val_frame_acc": round(val_frame_acc, 4) if isinstance(val_frame_acc, (int, float)) else None,
        "best_val_epoch": best_val_epoch,
        "oracle_delta_3iter": oracle_delta_3iter,
        "val_best_streak": val_best_streak,
        "val_oracle_gap": val_oracle_gap,
        "plan_summary": plan_summary,
        "oracle_summary": oracle_summary,
        "policy_summary": policy_summary,
    }
    with open(args.history, "a") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    if args.full_history:
        full_row = {
            **summary,
            "plan_seconds": args.plan_seconds,
            "design_seconds": args.design_seconds,
            "oracle_search_seconds": args.oracle_search_seconds,
            "oracle_agent_seconds": args.oracle_agent_seconds,
            "policy_train_seconds": args.policy_train_seconds,
            "policy_seconds": args.policy_seconds,
        }
        with open(args.full_history, "a") as f:
            f.write(json.dumps(full_row, ensure_ascii=False) + "\n")

    try:
        with open(args.state) as f:
            s = json.load(f)
    except FileNotFoundError:
        s = {}
    s["iteration"] = args.iter
    if train_oracle_frame_acc > s.get("best_train_oracle_frame_acc", 0):
        s["best_train_oracle_frame_acc"] = train_oracle_frame_acc
        s["best_oracle_iteration"] = args.iter
        s["best_oracle_name"] = r.get("name", "")
        s["best_oracle_formulation_file"] = r.get("formulation_file", "")
    if val_frame_acc is not None and val_frame_acc > s.get("best_val_frame_acc", -1):
        s["best_val_frame_acc"] = val_frame_acc
        s["best_val_iteration"] = args.iter
        s["best_val_name"] = r.get("name", "")
        s["best_val_formulation_file"] = r.get("formulation_file", "")
    with open(args.state, "w") as f:
        json.dump(s, f, indent=2)

    print(json.dumps(summary, ensure_ascii=False))


def cmd_record_failure(args):
    try:
        with open(args.state) as f:
            s = json.load(f)
    except FileNotFoundError:
        s = {}
    s["iteration"] = args.iter
    with open(args.state, "w") as f:
        json.dump(s, f, indent=2)


def _finalize_block(blocks, current):
    block = "\n".join(current).strip()
    if not block:
        return
    first_line = block.splitlines()[0].strip()
    if first_line.startswith(("Added [", "Created [", "Implemented [", "Updated [")):
        return
    blocks.append(block)


def _extract_codex_blocks(text):
    blocks = []
    current = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        if stripped == "codex":
            if current:
                _finalize_block(blocks, current)
            current = []
            continue
        if current is None:
            continue
        if (
            stripped in ROLE_HEADERS
            or stripped == "--------"
            or line.startswith("OpenAI Codex ")
            or line.startswith("diff --git ")
            or stripped == "tokens used"
        ):
            _finalize_block(blocks, current)
            current = None
            continue
        current.append(line)
    if current:
        _finalize_block(blocks, current)
    return blocks


def cmd_extract_reasoning(args):
    with open(args.input) as f:
        text = f.read()
    blocks = _extract_codex_blocks(text)
    if not blocks:
        return
    with open(args.output, "a") as f:
        if args.label:
            f.write(f"=== {args.label} ===\n")
        for idx, block in enumerate(blocks):
            if idx > 0:
                f.write("\n\n")
            f.write(block)
        f.write("\n\n")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("config-get")
    p.add_argument("--config", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--default", default="")

    p = sub.add_parser("state-get")
    p.add_argument("--state", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--default", default="0")

    p = sub.add_parser("check-syntax")
    p.add_argument("--file", required=True)

    p = sub.add_parser("verify-formulation")
    p.add_argument("--file", required=True)
    p.add_argument("--config", default="",
                   help="Path to config.yaml; reads verifier.* S1/S2 limits (falls back to verifier defaults).")

    p = sub.add_parser("verify-protocol")
    p.add_argument("--file", required=True)

    p = sub.add_parser("verify-policy-protocol")
    p.add_argument("--file", required=True)

    p = sub.add_parser("record-result")
    p.add_argument("--result", required=True)
    p.add_argument("--memo", default="",
                   help="Path to strategy_memo/iter_NNN.md (used to extract plan_summary).")
    p.add_argument("--oracle-report", default="",
                   help="Path to oracle_reports/iter_NNN.md (used to extract oracle_summary).")
    p.add_argument("--policy-report", default="",
                   help="Path to policy_reports/iter_NNN.md (used to extract policy_summary).")
    p.add_argument("--policy-result", default="",
                   help="Path to policy_results/iter_NNN.json (used to extract val_frame_acc / best_val_epoch).")
    p.add_argument("--history", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--iter", type=int, required=True)
    p.add_argument("--full-history", default="",
                   help="Optional path to a human-readable history.jsonl (extra timing fields). Kept outside agent_context.")
    p.add_argument("--plan-seconds", type=int, default=0)
    p.add_argument("--design-seconds", type=int, default=0)
    p.add_argument("--oracle-search-seconds", type=int, default=0)
    p.add_argument("--oracle-agent-seconds", type=int, default=0)
    p.add_argument("--policy-train-seconds", type=int, default=0)
    p.add_argument("--policy-seconds", type=int, default=0)

    p = sub.add_parser("record-failure")
    p.add_argument("--state", required=True)
    p.add_argument("--iter", type=int, required=True)

    p = sub.add_parser("extract-reasoning")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--label", default="")

    args = parser.parse_args()
    if args.command == "config-get":
        cmd_config_get(args)
    elif args.command == "state-get":
        cmd_state_get(args)
    elif args.command == "check-syntax":
        cmd_check_syntax(args)
    elif args.command == "verify-formulation":
        cmd_verify_formulation(args)
    elif args.command == "verify-protocol":
        cmd_verify_protocol(args)
    elif args.command == "verify-policy-protocol":
        cmd_verify_policy_protocol(args)
    elif args.command == "record-result":
        cmd_record_result(args)
    elif args.command == "record-failure":
        cmd_record_failure(args)
    elif args.command == "extract-reasoning":
        cmd_extract_reasoning(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
