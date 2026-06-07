"""
Apples-to-apples comparison of two adapters evaluated on the SAME common test set.

Each input is an eval_on_common.jsonl produced by evaluate.py (both adapters run
on data/track2_both_pass/test.jsonl, the full held-out set). We report metrics
overall and split by whether the *reference* target contains dynamic():

  overall       — the whole held-out set
  dynamic-free  — targets a precise (no-dynamic) annotation  (= track1's test)
  with-dynamic  — targets a gradual annotation

Metrics: exact-match %, semantic success % (distance 0 or 1), typecheck-pass %.

Usage:
  python scripts/compare_tracks.py runs/track1_no_gradual/eval_on_common.jsonl \\
                                   runs/track2_both_pass/eval_on_common.jsonl
"""
import json
import sys
from pathlib import Path


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def metrics(rows):
    n = len(rows)
    if n == 0:
        return None
    em = sum(1 for r in rows if r.get("exact_match"))
    succ = sum(1 for r in rows if r.get("semantic_distance", 99) <= 1)
    tcp = sum(1 for r in rows if r.get("type_check_pass") is True)
    # how often the model itself emitted dynamic() (precision signal)
    emit_dyn = sum(1 for r in rows if "dynamic()" in (r.get("generated_elixir_type") or ""))
    # safe AND precise: typechecks WITHOUT resorting to the dynamic() escape hatch.
    # This is the metric that actually measures "safer types"; plain TC% is
    # inflated because any dynamic()-containing prediction typechecks trivially.
    safe_prec = sum(
        1 for r in rows
        if r.get("type_check_pass") is True
        and "dynamic()" not in (r.get("generated_elixir_type") or "")
    )
    return {
        "N": n,
        "EM%": 100 * em / n,
        "succ%": 100 * succ / n,
        "TC%": 100 * tcp / n,
        "safe&prec%": 100 * safe_prec / n,
        "emit_dyn%": 100 * emit_dyn / n,
    }


def subset(rows, kind):
    if kind == "overall":
        return rows
    has = lambda r: "dynamic()" in (r.get("reference_elixir_type") or "")
    return [r for r in rows if (has(r) if kind == "with-dynamic" else not has(r))]


def main():
    files = sys.argv[1:]
    if len(files) < 2:
        print("usage: compare_tracks.py <evalA.jsonl> <evalB.jsonl> [...]")
        sys.exit(1)

    # label = the run/track directory name
    labels = [Path(f).parent.name for f in files]
    data = {lab: load(f) for lab, f in zip(labels, files)}

    cols = ["N", "EM%", "succ%", "TC%", "safe&prec%", "emit_dyn%"]
    header = f"{'subset':14} {'adapter':22} " + " ".join(f"{c:>9}" for c in cols)
    for kind in ("overall", "dynamic-free", "with-dynamic"):
        print(header if kind == "overall" else "")
        print("-" * len(header))
        for lab in labels:
            m = metrics(subset(data[lab], kind))
            if not m:
                continue
            vals = " ".join(
                f"{m[c]:9d}" if c == "N" else f"{m[c]:9.1f}" for c in cols
            )
            print(f"{kind:14} {lab:22} {vals}")

    print("\nLegend: EM exact-match | succ semantic distance<=1 | "
          "TC typecheck-accept | safe&prec typecheck-accept WITHOUT dynamic() "
          "(genuine safety) | emit_dyn share of predictions containing dynamic()")


if __name__ == "__main__":
    main()
