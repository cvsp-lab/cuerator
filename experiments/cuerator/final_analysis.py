"""Post-loop analysis: pick best iteration by val_frame_acc, look up its
already-computed test_summary.json, write final_summary.json.

Replaces the old Stage 2 final_policy_train: since policy training already runs
in-loop, every iter has its own test_summary.json. We just select the winner by
val and surface the corresponding test result here, at the end.

Usage:
    python -m experiments.cuerator.final_analysis --session <session_dir>
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, type=str)
    args = ap.parse_args()

    session_dir = Path(args.session).resolve()
    history = session_dir / "history.jsonl"  # full history (with timing fields)
    if not history.is_file():
        print(f"  [final-analysis] history.jsonl not found at {history}", file=sys.stderr)
        sys.exit(1)

    rows = []
    with history.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    val_rows = [r for r in rows if r.get("val_frame_acc") is not None]
    if not val_rows:
        print("  [final-analysis] no rows with val_frame_acc — nothing to select.", file=sys.stderr)
        sys.exit(1)

    best = max(val_rows, key=lambda r: r["val_frame_acc"])
    best_iter = best["iter"]
    print(f"  [final-analysis] best by val_frame_acc: iter {best_iter} "
          f"({best['name'][:60]}) val={best['val_frame_acc']:.4f} "
          f"oracle={best.get('train_oracle_frame_acc'):.4f}")

    iter_dir = session_dir / "policy_runs" / f"iter_{best_iter:03d}"
    test_summary_path = iter_dir / "test_summary.json"
    best_val_summary_path = iter_dir / "best_val_summary.json"

    final = {
        "best_iter": best_iter,
        "best_name": best.get("name"),
        "train_oracle_frame_acc": best.get("train_oracle_frame_acc"),
        "val_frame_acc": best.get("val_frame_acc"),
        "best_val_epoch": best.get("best_val_epoch"),
        "policy_run_dir": str(iter_dir.relative_to(session_dir)),
    }
    if test_summary_path.is_file():
        with test_summary_path.open() as f:
            final["test"] = json.load(f)
    else:
        final["test"] = None
        print(f"  [final-analysis] WARNING test_summary.json missing at {test_summary_path}",
              file=sys.stderr)
    if best_val_summary_path.is_file():
        with best_val_summary_path.open() as f:
            final["best_val"] = json.load(f)

    out_path = session_dir / "final_test_summary.json"
    with out_path.open("w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    print(f"  [final-analysis] final_test_summary saved to {out_path}")


if __name__ == "__main__":
    main()
