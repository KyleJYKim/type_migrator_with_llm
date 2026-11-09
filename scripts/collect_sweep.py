"""
Collect a checkpoint sweep: one row per checkpoint with every selection signal
and the task metric on the validation subset, then answer two questions:

  1. Which checkpoint does each selection rule pick, and how compatible is it?
     (rules: max validation compatibility, min eval_loss logged in training,
      min full-sequence loss, min type-only loss, max type-token accuracy)
  2. How well does each cheap signal track validation compatibility across
     checkpoints? (Spearman rank correlation; losses are negated so that
     +1 always means "ranks checkpoints like compatibility does")

    python scripts/collect_sweep.py runs/sweep/qwen_completion_instr [more runs ...]

Writes <run>/sweep/summary.md and summary.csv; prints a cross-run table when
given several runs. Uses only the validation subset -- report the TEST score of
the chosen checkpoint separately, once, after choosing.
"""
import csv
import json
import math
import sys
from pathlib import Path


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra)); vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb) if va and vb else float("nan")


def train_eval_losses(run):
    p = run / "train_log_history.json"
    if not p.exists():
        st = sorted(run.glob("checkpoint-*/trainer_state.json"),
                    key=lambda q: int(q.parent.name.split("-")[1]))
        if not st:
            return {}
        hist = json.load(open(st[-1]))["log_history"]
    else:
        hist = json.load(open(p))
    return {h["step"]: h["eval_loss"] for h in hist if "eval_loss" in h}


def collect(run):
    run = Path(run)
    evl = train_eval_losses(run)
    rows = []
    for d in sorted((run / "sweep").iterdir()):
        ev = d / "val_predictions.evaluated.jsonl"
        if not ev.exists():
            continue
        recs = [json.loads(l) for l in open(ev) if l.strip()]
        n = len(recs)
        comp = sum(r.get("set_theoretic_compatibility") is True for r in recs)
        exact = sum(r.get("set_theoretic_distance") == 0 for r in recs)
        null = sum(r.get("set_theoretic_compatibility") is None for r in recs)
        sig = json.load(open(d / "signals.json")) if (d / "signals.json").exists() else {}
        step = sig.get("step", int(d.name.split("-")[1]) if d.name.startswith("checkpoint-") else -1)
        p = comp / n
        rows.append({
            "checkpoint": d.name, "step": step, "n": n,
            "compat": p, "compat_se": math.sqrt(p * (1 - p) / n),
            "exact": exact / n, "unparseable": null / n,
            "train_eval_loss": evl.get(step),
            "full_seq_loss": sig.get("full_seq_loss"), "type_loss": sig.get("type_loss"),
            "type_token_acc": sig.get("type_token_acc"),
        })
    rows.sort(key=lambda r: (r["step"] < 0, r["step"]))
    return run, rows


RULES = [  # (name, column, higher_is_better)
    ("validation compatibility", "compat", True),
    ("eval_loss logged in training", "train_eval_loss", False),
    ("full-sequence loss (sweep)", "full_seq_loss", False),
    ("type-only loss (sweep)", "type_loss", False),
    ("type-token accuracy (sweep)", "type_token_acc", True),
]


def analyse(run, rows):
    out = []
    pick = {}
    for name, col, hi in RULES:
        cand = [r for r in rows if r.get(col) is not None]
        if not cand:
            continue
        key = (lambda r: (r[col], r["exact"])) if col == "compat" else (lambda r: r[col])
        best = max(cand, key=key) if hi else min(cand, key=key)
        sp = spearman([r[col] if hi else -r[col] for r in cand], [r["compat"] for r in cand]) if col != "compat" else 1.0
        pick[name] = (best, sp)

    lines = [f"# Checkpoint sweep: {run}", "",
             f"Validation subset: n = {rows[0]['n'] if rows else 0}. "
             "Compatibility ± one standard error.", "",
             "| checkpoint | step | compat | exact | unparseable | eval_loss (train) | full loss | type loss | type acc |",
             "|---|---|---|---|---|---|---|---|---|"]
    f = lambda v, pct=False: "—" if v is None else (f"{100*v:.1f}%" if pct else f"{v:.4f}")
    for r in rows:
        lines.append(f"| {r['checkpoint']} | {r['step']} | {100*r['compat']:.1f} ± {100*r['compat_se']:.1f}% "
                     f"| {f(r['exact'],1)} | {f(r['unparseable'],1)} | {f(r['train_eval_loss'])} "
                     f"| {f(r['full_seq_loss'])} | {f(r['type_loss'])} | {f(r['type_token_acc'],1)} |")
    lines += ["", "## What each selection rule picks", "",
              "| rule | picks | its val compat | its exact | Spearman with compat |", "|---|---|---|---|---|"]
    for name, (b, sp) in pick.items():
        lines.append(f"| {name} | {b['checkpoint']} | {100*b['compat']:.1f}% | {100*b['exact']:.1f}% | "
                     f"{'—' if name.startswith('validation') else f'{sp:+.2f}'} |")
    best = pick.get("validation compatibility", (None,))[0]
    if best:
        lines += ["", f"Selected by validation compatibility: **{best['checkpoint']}**. "
                  "Score the test set with this checkpoint only (generate.py --adapter_dir <it>)."]
    md = "\n".join(lines) + "\n"
    (run / "sweep" / "summary.md").write_text(md)
    with open(run / "sweep" / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(md)
    return pick


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    picks = {}
    for run in sys.argv[1:]:
        r, rows = collect(run)
        if not rows:
            print(f"(no scored checkpoints under {r}/sweep)")
            continue
        picks[r.name] = analyse(r, rows)
    if len(picks) > 1:
        print("## Across runs: validation compatibility of the checkpoint each rule picks\n")
        names = [n for n, _, _ in RULES]
        print("| run | " + " | ".join(names) + " |")
        print("|---|" + "---|" * len(names))
        for run, p in picks.items():
            cells = [f"{100*p[n][0]['compat']:.1f}% ({p[n][0]['checkpoint']})" if n in p else "—" for n in names]
            print(f"| {run} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
