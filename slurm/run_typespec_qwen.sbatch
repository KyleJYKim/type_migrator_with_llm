#!/bin/bash
#SBATCH --job-name=ElixirTypespec_Qwen
#SBATCH --qos=normal
#SBATCH --gpus=a100_7g.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/run-%j.out
#SBATCH --error=logs/run-%j.err

# TYPESPEC track, decoder-only half: fine-tune Qwen2.5-Coder to write the
# original Erlang-style @spec, then generate on the held-out split. The
# predictions are translated into Elixir Types and scored afterwards, locally,
# by `mix eval_typespec_predictions` -- scoring needs the custom compiler and so
# cannot run here.
#
# The split is the same seed-42 subproject partition as the Descr track, entry
# for entry (prepare_data_typespec.py asserts this), so the two tracks compare
# directly on the same 2127 test entries.
#
# Decoding is UNCONSTRAINED: there is no TypeSpec grammar in this repo, and
# leaving both models unconstrained keeps them symmetric on this track.
CONFIG="${1:-configs/qwen7b_qlora_typespec.yaml}"
SEED=42
DATASET="typespec_both_pass"
DATA_DIR="data/seed${SEED}/${DATASET}"
TEST="${DATA_DIR}/test.jsonl"

NAME=$(basename "$CONFIG" .yaml)
OUT_DIR=$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1])).get("output_dir", ""))' "$CONFIG")
OUT_DIR="${OUT_DIR:-runs/${NAME}}/seed${SEED}"

eval "$(conda shell.bash hook)"
conda activate etr

cd ~/elixir_types_trainer/type_migrator_with_llm

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -s "$TEST" ]; then
  echo "MISSING split: $TEST -- run 'python scripts/prepare_data_typespec.py ${SEED}' first."
  exit 1
fi

echo "==================  TRAIN  ${DATASET}  (seed ${SEED})  =================="
python scripts/train_sft_typespec.py \
  --config "$CONFIG" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUT_DIR"

echo "==================  GENERATE  ${DATASET}  (unconstrained)  =================="
python scripts/generate_typespec.py \
  --adapter_dir "$OUT_DIR" \
  --test_file "$TEST" \
  --out_file "${OUT_DIR}/eval_on_common.jsonl"

echo "==================  COMPLETION CHECK  =================="
# A complete run writes one prediction line per test entry; fail on a short or
# missing file so an incomplete run is never mistaken for a finished one.
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
echo "  mix eval_typespec_predictions <trainer>/${OUT_DIR}/eval_on_common.jsonl"
