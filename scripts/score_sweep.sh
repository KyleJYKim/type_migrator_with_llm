#!/usr/bin/env bash
# Score every checkpoint's validation predictions with the set-theoretic
# evaluator (compatible?/2, subtype?/2). Run LOCALLY: it needs the custom
# Elixir toolchain in type_migrator.
#
#   scripts/score_sweep.sh runs/sweep/qwen_completion_instr [path/to/type_migrator]
#
# Writes val_predictions.evaluated.jsonl beside each val_predictions.jsonl and
# skips files already scored. Then run scripts/collect_sweep.py on the run dir.
set -euo pipefail
RUN_DIR="$(cd "$1" && pwd)"
TM_DIR="$(cd "${2:-$(dirname "$0")/../../type_migrator}" && pwd)"

shopt -s nullglob
files=("$RUN_DIR"/sweep/*/val_predictions.jsonl)
[ ${#files[@]} -gt 0 ] || { echo "no sweep predictions under $RUN_DIR/sweep"; exit 1; }

for f in "${files[@]}"; do
  out="${f%.jsonl}.evaluated.jsonl"
  if [ -s "$out" ] && [ "$(wc -l < "$out")" -eq "$(wc -l < "$f")" ]; then
    echo "skip   $(basename "$(dirname "$f")")"; continue
  fi
  echo "score  $(basename "$(dirname "$f")")"
  (cd "$TM_DIR" && mix eval_predictions set-theoretic "$f" > /dev/null)
done
echo "Done. Next: python scripts/collect_sweep.py $RUN_DIR"
