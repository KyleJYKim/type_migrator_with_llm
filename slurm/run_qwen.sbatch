#!/bin/bash
#SBATCH --job-name=ElixirType training and generating
#SBATCH --qos=normal
#SBATCH --gpus=a100_7g.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/run-%j.out
#SBATCH --error=logs/run-%j.err

# Single-dataset run: fine-tune Qwen (decoder-only) on the both-pass expanded
# pool -- the raw, fully-expanded translation labels accepted by both Dialyzer
# and the typechecker. No tracks, no label variants. Seed 42 fixes the split.
# The 4-bit QLoRA 7B fits the 40 GB MIG slice; bump to a100_7g.80gb only if OOM.
# CodeT5+ (encoder-decoder) is a separate sbatch, added later.
CONFIG="${1:-configs/qwen7b_qlora.yaml}"
SEED=42
DATASET="track2_both_pass_expanded"
DATA_DIR="data/seed${SEED}/${DATASET}"
OUT_DIR="runs/seed${SEED}/${DATASET}"
TEST="${DATA_DIR}/test.jsonl"

eval "$(conda shell.bash hook)"
conda activate etr

cd ~/elixir_types_trainer/type_migrator_with_llm

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -s "$TEST" ]; then
  echo "MISSING split: $TEST -- run 'python scripts/prepare_data.py ${SEED}' first."
  exit 1
fi

echo "==================  TRAIN  ${DATASET}  (seed ${SEED})  =================="
python scripts/train_sft.py \
  --config "$CONFIG" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUT_DIR"

echo "==================  GENERATE  ${DATASET}  (grammar-constrained)  =================="
python scripts/generate.py \
  --adapter_dir "$OUT_DIR" \
  --test_file "$TEST" \
  --out_file "${OUT_DIR}/eval_on_common.jsonl" \
  --constrain

echo "==================  COMPLETION CHECK  =================="
# A complete run writes one prediction line per test entry; fail (non-zero exit)
# on a short/missing file so an incomplete run is never mistaken for a finished one.
expected=$(wc -l < "$TEST" | tr -d ' ')
if [ ! -s "${OUT_DIR}/eval_on_common.jsonl" ]; then
  echo "GENERATION INCOMPLETE -- output missing."; exit 1
fi
got=$(wc -l < "${OUT_DIR}/eval_on_common.jsonl" | tr -d ' ')
if [ "$got" -ne "$expected" ]; then
  echo "GENERATION INCOMPLETE (${got}/${expected})."; exit 1
fi
echo "OK  (${got}/${expected})"

echo "Done. Score LOCALLY (needs the custom Elixir toolchain) with:"
echo "  mix eval_predictions set-theoretic <trainer>/${OUT_DIR}/eval_on_common.jsonl"

# --- CodeT5+ (encoder-decoder) run, to be added later: a separate sbatch that
#     trains scripts/train_seq2seq.py on the same DATA_DIR and generates on TEST,
#     then scored by the same compatibility metric for a like-for-like compare. ---
