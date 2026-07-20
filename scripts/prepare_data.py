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
import hashlib
import json
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
    #
    # HASH-BASED, not shuffle-based: a subproject's split is a pure function
    # of its own name (md5 mod 100, salted with SEED), independent of every
    # other subproject. The old shuffle-of-the-universe approach made the
    # entire partition depend on the dataset's exact composition -- a ~200
    # entry change between two dataset builds once replaced two-thirds of
    # the test subprojects, silently invalidating every cross-round metric
    # comparison. With hashing, a subproject keeps its split assignment
    # across dataset regenerations forever, so runs stay comparable.
    universe = sorted({subproject(e) for e in track2})

    def split_of_name(sp):
        h = int(hashlib.md5(f"{SEED}:{sp}".encode()).hexdigest(), 16) % 100
        if h < 70:
            return "train"
        elif h < 85:
            return "val"
        return "test"

    train_sp = {sp for sp in universe if split_of_name(sp) == "train"}
    val_sp = {sp for sp in universe if split_of_name(sp) == "val"}
    test_sp = {sp for sp in universe if split_of_name(sp) == "test"}
    n = len(universe)
    print(f"\nSubproject universe  : {n}  "
          f"(train {len(train_sp)} / val {len(val_sp)} / test {len(test_sp)})")

    def split_of(e):
        return split_of_name(subproject(e))

    # Each mode -> (dir-name suffix, source field swapped into `elixir_type`
    # for train/val labels, split_info note). `test` is IDENTICAL across all
    # modes (always the original, fully expanded elixir_type -- it is the
    # scoring ground truth, never a training target). The three modes form an
    # ablation of the label-honesty transforms:
    #   compact  : struct/keyword/map compaction AND return dynamic()-hedging
    #   hedged   : ONLY return dynamic()-hedging (isolates hedging's effect)
    #   expanded : none -- the raw human-declared spec translation
    LABEL_MODES = {
        "compact": (
            "",
            "compact_elixir_type",
            "train/val: elixir_type is compact_elixir_type -- ungrounded struct "
            "expansions collapsed to open %{..., :__struct__ => ...}, ungrounded "
            "keyword pairs/maps generalized, AND return-union arms with no visible "
            "tail-constructor evidence hedged to dynamic(). test: original expanded "
            "reference, unchanged (scoring ground truth).",
        ),
        "hedged": (
            "_hedged",
            "hedged_elixir_type",
            "train/val: elixir_type is hedged_elixir_type -- the fully expanded "
            "reference with ONLY return-union arms lacking visible tail-constructor "
            "evidence hedged to dynamic() (no struct/keyword/map compaction). "
            "Ablation isolating the return-hedging transform. test: original "
            "expanded reference, unchanged (scoring ground truth).",
        ),
        "expanded": (
            "_expanded",
            None,
            "train/val/test: elixir_type is the original, fully expanded reference "
            "throughout -- the raw human-declared spec translation, no transforms.",
        ),
    }

    def write_track(name, candidates, mode):
        suffix, source_field, note = LABEL_MODES[mode]
        out = DATA_DIR / f"{name}{suffix}"
        out.mkdir(parents=True, exist_ok=True)
        splits = defaultdict(list)
        for e in candidates:
            splits[split_of(e)].append(e)

        print(f"\n[{out.name}]  {len(candidates)} entries")
        counts = {}
        for s in ("train", "val", "test"):
            items = splits[s]
            n_sp = len({subproject(e) for e in items})
            counts[s] = {"entries": len(items), "subprojects": n_sp}
            print(f"  {s:5s}: {len(items):5d} entries from {n_sp:3d} subprojects")
            with open(out / f"{s}.jsonl", "w") as f:
                for e in items:
                    out_e = e
                    if source_field and s in ("train", "val") and e.get(source_field):
                        out_e = {**e, "elixir_type": e[source_field]}
                    f.write(json.dumps(out_e) + "\n")

        track1_filter = (f"both tools pass; dynamic-free; "
                         f"def<{MAX_DEF_LEN}, type<{MAX_TYPE_LEN}")
        track2_filter = f"both tools pass; def<{MAX_DEF_LEN}, type<{MAX_TYPE_LEN}"
        with open(out / "split_info.json", "w") as f:
            json.dump({
                "track": out.name,
                "seed": SEED,
                "split_unit": "subproject = project/module_root",
                "filter": track1_filter if name.startswith("track1") else track2_filter,
                "counts": counts,
                "target_field": "elixir_type",
                "target_field_note": note,
                "auxiliary_fields": ["spec", "type", "return_expressions", "argument_patterns"],
                "train_subprojects": sorted(train_sp),
                "val_subprojects": sorted(val_sp),
                "test_subprojects": sorted(test_sp),
            }, f, indent=2)

    for track_name, candidates in [("track1_no_gradual", track1), ("track2_both_pass", track2)]:
        for mode in ("compact", "hedged", "expanded"):
            write_track(track_name, candidates, mode)


if __name__ == "__main__":
    main()
