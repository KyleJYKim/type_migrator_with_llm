"""
Build train/val/test splits for the TYPESPEC track: the model predicts the
original Erlang-style `@spec`, which is translated into an Elixir Type
afterwards (`mix eval_typespec_predictions`) and only then scored.

Separate from prepare_data.py so that neither track's data can shift under the
other. The pool, the filter, and the split arithmetic here are deliberately
IDENTICAL to that script's track2_both_pass:

  * pool   : entries both Dialyzer and the typechecker accept;
  * filter : definition < 2000 chars AND elixir_type < 1000 chars;
  * split  : subproject (project/module_root) hashed as md5(f"{SEED}:{sp}") % 100,
             [0,70) train, [70,85) val, [85,100) test.

The elixir_type length filter is kept even though this track never trains on
elixir_type. Dropping it would admit entries the Descr track excluded and the
two test sets would no longer be the same 2127 entries -- the one thing that
makes the tracks comparable. (It costs little either way: spec lengths run to
1178 characters against a 2000-character median definition.)

The only real difference is the target: `spec` rather than `elixir_type`. Both
fields are carried through to every split, since scoring needs the reference
annotation and the evaluator needs the reference spec for its ceiling.

Usage:  python scripts/prepare_data_typespec.py [seed]
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42

# Read the SAME source file as prepare_data.py, including its preference for the
# translated dataset. This track's prompt reads `type` and does not need
# `translated_type`, but both tracks must be cut from one file or their splits
# can diverge -- and comparability of the two test sets is the whole point.
TRANSLATED_DATASET = "data/dataset.with_translated_types.jsonl"
ORIGINAL_DATASET = "data/dataset.jsonl"
DATASET = TRANSLATED_DATASET if Path(TRANSLATED_DATASET).is_file() else ORIGINAL_DATASET
DATA_DIR = Path(f"data/seed{SEED}")
OUT_DIR = DATA_DIR / "typespec_both_pass"

# The Descr track's split, used only to assert the two tracks agree entry for
# entry. Absent is fine -- the check is then skipped.
DESCR_TRACK_DIR = DATA_DIR / "track2_both_pass_expanded"

MAX_DEF_LEN = 2000
MAX_TYPE_LEN = 1000

TARGET_FIELD = "spec"


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
    )


def length_ok(e):
    return len(e["definition"]) < MAX_DEF_LEN and len(e["elixir_type"]) < MAX_TYPE_LEN


def eligible(e):
    # A spec is the target here, so an entry without one cannot be used. Every
    # both-pass entry in the current dataset has one (an entry earns its
    # translated annotation FROM its spec), so this drops nothing today; it
    # guards a future dataset that mixes in spec-less functions.
    return both_pass(e) and length_ok(e) and bool((e.get(TARGET_FIELD) or "").strip())


def split_of_name(sp):
    h = int(hashlib.md5(f"{SEED}:{sp}".encode()).hexdigest(), 16) % 100
    if h < 70:
        return "train"
    elif h < 85:
        return "val"
    return "test"


def entry_key(e):
    """Identity of an entry across tracks, for the comparability assertion."""
    return (e.get("file"), e.get("module"), e.get("function"), e.get("arity"),
            tuple(e.get("spec_lines") or ()))


def check_against_descr_track(splits):
    """Warn loudly if this track's splits are not the Descr track's splits."""
    if not DESCR_TRACK_DIR.is_dir():
        print(f"\n(no {DESCR_TRACK_DIR} -- skipping the cross-track check)")
        return

    print(f"\n=== Cross-track check against {DESCR_TRACK_DIR.name} ===")
    for s in ("train", "val", "test"):
        other = DESCR_TRACK_DIR / f"{s}.jsonl"
        if not other.is_file():
            print(f"  {s:5s}: {other} missing -- skipped")
            continue
        with open(other) as f:
            theirs = {entry_key(json.loads(l)) for l in f if l.strip()}
        ours = {entry_key(e) for e in splits[s]}
        if ours == theirs:
            print(f"  {s:5s}: identical ({len(ours)} entries)")
        else:
            print(f"  {s:5s}: *** DIFFERS *** ours {len(ours)}, theirs {len(theirs)}, "
                  f"only-ours {len(ours - theirs)}, only-theirs {len(theirs - ours)}")
            print("         the two tracks are NOT comparable entry for entry")


def main():
    with open(DATASET) as f:
        entries = [json.loads(l) for l in f if l.strip()]

    pool = [e for e in entries if eligible(e)]
    print(f"Dataset              : {DATASET}")
    print(f"Total entries          : {len(entries)}")
    print(f"both_pass + length ok  : {len(pool)}")

    universe = sorted({subproject(e) for e in pool})
    counts_sp = {s: sum(1 for sp in universe if split_of_name(sp) == s)
                 for s in ("train", "val", "test")}
    print(f"Subproject universe    : {len(universe)}  "
          f"(train {counts_sp['train']} / val {counts_sp['val']} / test {counts_sp['test']})")

    splits = defaultdict(list)
    for e in pool:
        splits[split_of_name(subproject(e))].append(e)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[{OUT_DIR.name}]  target = {TARGET_FIELD}")
    counts = {}
    for s in ("train", "val", "test"):
        items = splits[s]
        n_sp = len({subproject(e) for e in items})
        counts[s] = {"entries": len(items), "subprojects": n_sp}
        print(f"  {s:5s}: {len(items):5d} entries from {n_sp:3d} subprojects")
        with open(OUT_DIR / f"{s}.jsonl", "w") as f:
            for e in items:
                f.write(json.dumps(e) + "\n")

    with open(OUT_DIR / "split_info.json", "w") as f:
        json.dump({
            "track": OUT_DIR.name,
            "seed": SEED,
            "split_unit": "subproject = project/module_root",
            "filter": f"both tools pass; def<{MAX_DEF_LEN}, elixir_type<{MAX_TYPE_LEN}; spec present",
            "counts": counts,
            "target_field": TARGET_FIELD,
            "target_field_note": (
                "train/val/test target is the ORIGINAL @spec. `elixir_type` is carried "
                "through unchanged as the scoring reference: a prediction is translated "
                "into an Elixir Type (mix eval_typespec_predictions) and compared against "
                "it with compatible?/2. The elixir_type length filter is applied here too, "
                "purely to keep this track's splits identical to track2_both_pass."
            ),
            "auxiliary_fields": ["elixir_type", "type", "return_expressions", "argument_patterns"],
            "train_subprojects": sorted(sp for sp in universe if split_of_name(sp) == "train"),
            "val_subprojects": sorted(sp for sp in universe if split_of_name(sp) == "val"),
            "test_subprojects": sorted(sp for sp in universe if split_of_name(sp) == "test"),
        }, f, indent=2)

    check_against_descr_track(splits)


if __name__ == "__main__":
    main()
