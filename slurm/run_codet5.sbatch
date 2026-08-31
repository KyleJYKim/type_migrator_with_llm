#!/bin/bash
#SBATCH --job-name=codet5-expanded
#SBATCH --qos=normal
#SBATCH --gpus=a100_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/run-%j.out
#SBATCH --error=logs/run-%j.err

# Single-dataset CodeT5+ (encoder-decoder) run on the both-pass expanded pool,
# seed 42 -- the encoder-decoder counterpart of run_qwen.sbatch: SAME dataset and
# SAME encoder input (train_seq2seq.build_prompt == generate.format_prompt), for a
# like-for-like model-family comparison scored by the same compatibility metric.
#
# Default = codet5p-770m (standard T5, safe, fits the 40 GB MIG slice). Run it
# first. For codet5p-2b (custom modeling code, trust_remote_code; predates
# transformers 5.x and may break), pass the config AND switch the GPU to 80 GB:
#   #SBATCH --gpus=a100_7g.80gb:1
#   sbatch run_codet5.sbatch configs/codet5p_2b.yaml
#
# NOTE: decoding is grammar-constrained (--constrain), the SAME descr grammar as
# the Qwen run, so the two are decode-matched. SMOKE-TEST 770m first: transformers-cfg
# with the T5 (sentencepiece) tokenizer is less battle-tested than with Qwen's; if the
# grammar processor errors on load, drop --constrain and report the caveat instead.
# The one remaining architectural difference is the 512-token encoder cap (Qwen: 1024).
CONFIG="${1:-configs/codet5p_770m.yaml}"
SEED=42
DATASET="track2_both_pass_expanded"
DATA_DIR="data/seed${SEED}/${DATASET}"
NAME=$(basename "$CONFIG" .yaml)
OUT_DIR="runs/seed${SEED}/${NAME}"
TEST="${DATA_DIR}/test.jsonl"

eval "$(conda shell.bash hook)"
conda activate etr

cd ~/elixir_types_trainer/type_migrator_with_llm

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -s "$TEST" ]; then
  echo "MISSING split: $TEST -- run 'python scripts/prepare_data.py ${SEED}' first."
  exit 1
fi

echo "==================  TRAIN  ${NAME}  (seed ${SEED})  =================="
train_start=$SECONDS
python scripts/train_seq2seq.py \
  --config "$CONFIG" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUT_DIR"
train_elapsed=$((SECONDS - train_start))

echo "==================  GENERATE  ${NAME}  =================="
gen_start=$SECONDS
python scripts/generate_seq2seq.py \
  --model_dir "$OUT_DIR" \
  --test_file "$TEST" \
  --out_file "${OUT_DIR}/eval_on_common.jsonl" \
  --max_new_tokens 512 \
  --constrain \
  --trust_remote_code
gen_elapsed=$((SECONDS - gen_start))

echo "==================  COMPLETION CHECK  =================="
expected=$(wc -l < "$TEST" | tr -d ' ')
if [ ! -s "${OUT_DIR}/eval_on_common.jsonl" ]; then
  echo "GENERATION INCOMPLETE -- output missing."; exit 1
fi
got=$(wc -l < "${OUT_DIR}/eval_on_common.jsonl" | tr -d ' ')
if [ "$got" -ne "$expected" ]; then
  echo "GENERATION INCOMPLETE (${got}/${expected})."; exit 1
fi
echo "OK  (${got}/${expected})"

echo "==================  ELAPSED  =================="
printf 'Training:   %dm %02ds\n' $((train_elapsed/60)) $((train_elapsed%60))
printf 'Generation: %dm %02ds  (%d entries, %d ms/entry)\n' \
  $((gen_elapsed/60)) $((gen_elapsed%60)) "$got" $(( gen_elapsed * 1000 / (got > 0 ? got : 1) ))

echo "Done. Score LOCALLY (needs the custom Elixir toolchain) with:"
echo "  mix eval_predictions set-theoretic <trainer>/${OUT_DIR}/eval_on_common.jsonl"
