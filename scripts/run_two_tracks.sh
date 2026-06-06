#!/usr/bin/env bash
# Train + evaluate both tracks of the gradual-types study.
# MUST run on a GPU compute node — train_sft.py requires CUDA.
#
# Usage:
#   bash scripts/run_two_tracks.sh [CONFIG] [ELIXIR_BIN]
#     CONFIG     default configs/qwen7b_qlora.yaml
#     ELIXIR_BIN default ../type_migrator/elixir/bin  (custom typechecker for Tier C)
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/qwen7b_qlora.yaml}"
ELIXIR_BIN="${2:-../type_migrator/elixir/bin}"
TRACKS=(track1_no_gradual track2_both_pass)

# 1) (re)build the two-track datasets from the current dataset.jsonl
python scripts/prepare_data.py 42

# 2) train + evaluate each track on its own split
for t in "${TRACKS[@]}"; do
  echo "==================  TRAIN  ${t}  =================="
  python scripts/train_sft.py \
    --config "$CONFIG" \
    --data_dir "data/${t}" \
    --output_dir "runs/${t}"

  echo "==================  EVAL   ${t}  =================="
  python scripts/evaluate.py \
    --adapter_dir "runs/${t}" \
    --test_file "data/${t}/test.jsonl" \
    --elixir_bin "$ELIXIR_BIN"
done

echo "Done. Adapters + eval_results.jsonl under runs/track1_no_gradual and runs/track2_both_pass."
