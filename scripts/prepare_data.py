"""
Splits dataset.jsonl into train/val/test, using module-root as the sub-project unit 
since github org names like 'mbta', 'acalejos', or 'dashbit' actually contain many independent repos.
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import sys

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
random.seed(SEED)

DATASET = "../type_migrator/results/dataset.jsonl"
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

def subproject(e):
    """Use module root as the split unit; falls back to project for safety."""
    mod = e.get("module", "")
    if not mod:
        return e["project"]
    root = mod.split(".")[0]
    # Combine org + module-root so different orgs with same root don't collide
    return f"{e['project']}/{root}"

def is_candidate(e):
    return (
        e.get("dialyzer", {}).get("pass") is True
        and e.get("typecheck", {}).get("pass") is True
        and e.get("elixir_type")
        and "dynamic()" not in e["elixir_type"]
        and e.get("definition")
        and len(e["definition"]) < 2000
        and len(e["elixir_type"]) < 1000
    )

with open(DATASET) as f:
    entries = [json.loads(l) for l in f if l.strip()]

candidates = [e for e in entries if is_candidate(e)]
print(f"Candidates: {len(candidates)} / {len(entries)}")

# Group by sub-project
by_subproj = defaultdict(list)
for e in candidates:
    by_subproj[subproject(e)].append(e)

print(f"Distinct sub-projects: {len(by_subproj)}")
sub_sizes = Counter({k: len(v) for k, v in by_subproj.items()})
print(f"Top 10 sub-projects by size:")
for sp, c in sub_sizes.most_common(10):
    print(f"  {c:5d}  {sp}")

# Split sub-projects (not entries) — 70/15/15
subprojects = sorted(by_subproj.keys())
random.shuffle(subprojects)
n = len(subprojects)
train_sp = set(subprojects[: int(n * 0.7)])
val_sp   = set(subprojects[int(n * 0.7) : int(n * 0.85)])
test_sp  = set(subprojects[int(n * 0.85) :])

splits = {"train": [], "val": [], "test": []}
for e in candidates:
    sp = subproject(e)
    if sp in train_sp:   splits["train"].append(e)
    elif sp in val_sp:   splits["val"].append(e)
    else:                splits["test"].append(e)

print()
for name, items in splits.items():
    n_sp = len({subproject(e) for e in items})
    print(f"  {name}: {len(items):5d} entries from {n_sp:3d} sub-projects")
    with open(OUTPUT_DIR / f"{name}.jsonl", "w") as f:
        for e in items:
            f.write(json.dumps(e) + "\n")

with open(OUTPUT_DIR / "split_info.json", "w") as f:
    json.dump({
        "seed": SEED,
        "split_unit": "subproject = project/module_root",
        "n_train_subprojects": len(train_sp),
        "n_val_subprojects":   len(val_sp),
        "n_test_subprojects":  len(test_sp),
        "train_subprojects":   sorted(train_sp),
        "val_subprojects":     sorted(val_sp),
        "test_subprojects":    sorted(test_sp),
        "target_field":        "elixir_type",
        "auxiliary_fields":    ["spec", "type"],
    }, f, indent=2)