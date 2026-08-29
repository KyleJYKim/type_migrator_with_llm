#!/usr/bin/env bash
# Train both tracks of the gradual-types study, then generate predictions for
# BOTH adapters on the SAME held-out test set for a fair comparison.
# MUST run on a GPU compute node (here: at lip6) — train_sft.py requires CUDA.
#
# Usage:
#   bash scripts/run_two_tracks.sh [CONFIG]
#     CONFIG default configs/qwen7b_qlora.yaml
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/qwen7b_qlora.yaml}"
TRACKS=(track1_no_gradual track2_both_pass)

# The common held-out set = the full both_pass test split. 
# It is a superset of track1's test (track1 test is exactly its non-gradual subset).
# Then, compare_tracks.py also slices the results into the non-gradual and full-pass subsets.
COMMON_TEST="data/track2_both_pass/test.jsonl"

# 1) build the two-track datasets from the current dataset.jsonl into train/val/test
python scripts/prepare_data.py 42

# 2) train each track on its own split
for t in "${TRACKS[@]}"; do
  echo "==================  TRAIN  ${t}  =================="
  python scripts/train_sft.py \
    --config "$CONFIG" \
    --data_dir "data/${t}" \
    --output_dir "runs/${t}"
done

# 3) generate predictions for BOTH adapters on the SAME common held-out set
for t in "${TRACKS[@]}"; do
  echo "==================  GENERATE   ${t}  (common test)  =================="
  python scripts/generate.py \
    --adapter_dir "runs/${t}" \
    --test_file "$COMMON_TEST" \
    --out_file "runs/${t}/eval_on_common.jsonl" \
    --constrain
done

# 4) print the comparison (overall + per subset)
echo "==================  COMPARISON  =================="
python scripts/compare_tracks.py \
  runs/track1_no_gradual/eval_on_common.jsonl \
  runs/track2_both_pass/eval_on_common.jsonl

echo "Done. Per-adapter results: runs/*/eval_on_common.jsonl"
