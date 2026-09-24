#!/bin/bash
#SBATCH --job-name=ET-Qwen2.5-7b
#SBATCH --qos=normal
#SBATCH --gpus=a100_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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
TEST="${DATA_DIR}/test.jsonl"


eval "$(conda shell.bash hook)"
conda activate etr

cd ~/elixir_types_trainer/type_migrator_with_llm

# Resolved AFTER `conda activate` and the cd: this reads the config with
# the env's PyYAML and a repo-relative path. Run before either, the import
# fails, OUT_DIR falls back to runs/<config-name> and the job silently
# trains into a different directory than the config names.
NAME=$(basename "$CONFIG" .yaml)
OUT_DIR=$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1])).get("output_dir", ""))' "$CONFIG")
OUT_DIR="${OUT_DIR:-runs/${NAME}}/seed${SEED}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -s "$TEST" ]; then
  echo "MISSING split: $TEST -- run 'python scripts/prepare_data.py ${SEED}' first."
  exit 1
fi

echo "==================  TRAIN  ${DATASET}  (seed ${SEED})  =================="
# Stop here if training fails. Without this the job carries on and generates
# against whatever adapter is already in OUT_DIR -- normally the PREVIOUS run's
# -- so a crashed training run still reports predictions. Python tracebacks go
# to the .err file, not to this one.
if ! python scripts/train_sft.py \
  --config "$CONFIG" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUT_DIR"; then
  echo "TRAINING FAILED -- see logs/run-${SLURM_JOB_ID}.err. Not generating."
  exit 1
fi

# Never let a previous run's predictions stand in for this one's: the completion
# check counts lines, and a stale file of the right length passes it. Archived
# rather than deleted, because these are reported results.
PREDICTIONS="${OUT_DIR}/eval_on_common.jsonl"
if [ -e "$PREDICTIONS" ]; then
  mv "$PREDICTIONS" "${PREDICTIONS}.superseded-$(date +%Y%m%d-%H%M%S)"
  echo "Archived previous predictions beside ${PREDICTIONS}"
fi

echo "==================  GENERATE  ${DATASET}  (grammar-constrained)  =================="
if ! python scripts/generate.py \
  --adapter_dir "$OUT_DIR" \
  --test_file "$TEST" \
  --out_file "$PREDICTIONS" \
  --constrain; then
  echo "GENERATION FAILED -- see logs/run-${SLURM_JOB_ID}.err."
  exit 1
fi

echo "==================  COMPLETION CHECK  =================="
# A complete run writes one prediction line per test entry; fail (non-zero exit)
# on a short/missing file so an incomplete run is never mistaken for a finished one.
expected=$(wc -l < "$TEST" | tr -d ' ')
if [ ! -s "$PREDICTIONS" ]; then
  echo "GENERATION INCOMPLETE -- output missing."; exit 1
fi
got=$(wc -l < "$PREDICTIONS" | tr -d ' ')
if [ "$got" -ne "$expected" ]; then
  echo "GENERATION INCOMPLETE (${got}/${expected})."; exit 1
fi
echo "OK  (${got}/${expected})"

echo "Done. Score LOCALLY (needs the custom Elixir toolchain) with:"
echo "  mix eval_predictions set-theoretic <trainer>/${OUT_DIR}/eval_on_common.jsonl"

# --- CodeT5+ (encoder-decoder) run, to be added later: a separate sbatch that
#     trains scripts/train_seq2seq.py on the same DATA_DIR and generates on TEST,
#     then scored by the same compatibility metric for a like-for-like compare. ---
