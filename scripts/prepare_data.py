"""
Build train/val/test splits from ../type_migrator/results/dataset.jsonl for a
TWO-TRACK study of how gradual types (dynamic()) affect type prediction:

  track1_no_gradual : both tools pass AND the label contains no dynamic()
                      -> precise, "safe" labels only.
  track2_both_pass  : both tools pass (dynamic() allowed)
                      -> the larger pool, including gradual-typed labels.

Both tracks share ONE subproject -> split assignment, computed over track2's
subproject universe (the superset, since track1 is a subset). This guarantees:
  * the two tracks are directly comparable (same subprojects in the same split);
  * no subproject leaks across train/val/test in either track.

A subproject is `project/module_root`, because a GitHub org (the `project`
field) such as `mbta` or `cldr` actually contains many independent codebases.

Usage:  python scripts/prepare_data.py [seed]
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
DATASET = "data/dataset.jsonl"
DATA_DIR = Path("data")

# Length guards: drop pathological functions / annotations that would dominate
# batches without proportionate learning signal.
MAX_DEF_LEN = 2000
MAX_TYPE_LEN = 1000


def subproject(e):
    """Split unit: org + top-level module segment (e.g. mbta/AlertProcessor)."""
    mod = e.get("module", "")
    if not mod:
        return e["project"]
    return f"{e['project']}/{mod.split('.')[0]}"


def both_pass(e):
    return (
        e.get("dialyzer", {}).get("pass") is True
        and e.get("typecheck", {}).get("pass") is True
        and e.get("elixir_type")
        and e.get("definition")
    )


def length_ok(e):
    return len(e["definition"]) < MAX_DEF_LEN and len(e["elixir_type"]) < MAX_TYPE_LEN


def is_track2(e):
    """both_pass (gradual types allowed)."""
    return both_pass(e) and length_ok(e)


def is_track1(e):
    """both_pass minus gradual types (no dynamic() in the label)."""
    return is_track2(e) and "dynamic()" not in e["elixir_type"]


def main():
    with open(DATASET) as f:
        entries = [json.loads(l) for l in f if l.strip()]

    track2 = [e for e in entries if is_track2(e)]
    track1 = [e for e in entries if is_track1(e)]
    print(f"Total entries        : {len(entries)}")
    print(f"track2 (both_pass)   : {len(track2)}")
    print(f"track1 (no_gradual)  : {len(track1)}  "
          f"(= track2 minus {len(track2) - len(track1)} dynamic-containing)")

    # One subproject split over the superset (track2), applied to both tracks.
    universe = sorted({subproject(e) for e in track2})
    random.seed(SEED)
    random.shuffle(universe)
    n = len(universe)
    train_sp = set(universe[: int(n * 0.70)])
    val_sp = set(universe[int(n * 0.70): int(n * 0.85)])
    test_sp = set(universe[int(n * 0.85):])
    print(f"\nSubproject universe  : {n}  "
          f"(train {len(train_sp)} / val {len(val_sp)} / test {len(test_sp)})")

    def split_of(e):
        sp = subproject(e)
        if sp in train_sp:
            return "train"
        elif sp in val_sp:
            return "val"
        return "test"

    def write_track(name, candidates):
        out = DATA_DIR / name
        out.mkdir(parents=True, exist_ok=True)
        splits = defaultdict(list)
        for e in candidates:
            splits[split_of(e)].append(e)

        print(f"\n[{name}]  {len(candidates)} entries")
        counts = {}
        for s in ("train", "val", "test"):
            items = splits[s]
            n_sp = len({subproject(e) for e in items})
            counts[s] = {"entries": len(items), "subprojects": n_sp}
            print(f"  {s:5s}: {len(items):5d} entries from {n_sp:3d} subprojects")
            with open(out / f"{s}.jsonl", "w") as f:
                for e in items:
                    f.write(json.dumps(e) + "\n")

        track1_filter = (f"both tools pass; dynamic-free; "
                         f"def<{MAX_DEF_LEN}, type<{MAX_TYPE_LEN}")
        track2_filter = f"both tools pass; def<{MAX_DEF_LEN}, type<{MAX_TYPE_LEN}"
        with open(out / "split_info.json", "w") as f:
            json.dump({
                "track": name,
                "seed": SEED,
                "split_unit": "subproject = project/module_root",
                "filter": track1_filter if name.startswith("track1") else track2_filter,
                "counts": counts,
                "target_field": "elixir_type",
                "auxiliary_fields": ["spec", "type"],
                "train_subprojects": sorted(train_sp),
                "val_subprojects": sorted(val_sp),
                "test_subprojects": sorted(test_sp),
            }, f, indent=2)

    write_track("track1_no_gradual", track1)
    write_track("track2_both_pass", track2)


if __name__ == "__main__":
    main()
