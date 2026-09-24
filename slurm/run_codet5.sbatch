#!/bin/bash
#SBATCH --job-name=ET-Codet5p-770m
#SBATCH --qos=normal
#SBATCH --gpus=a100_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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
# Re-generate ONLY (reuse an already-trained model, skip the ~2-4h train) by passing
# "skiptrain" as the 2nd arg (env vars do NOT survive sbatch on this cluster):
#   sbatch run_codet5.sbatch configs/codet5p_770m.yaml skiptrain
#
# NOTE: decoding is UNCONSTRAINED. transformers-cfg 0.2.7 does not support CodeT5+'s
# RobertaTokenizerFast, and its GPT2 byte-mapping path is broken on transformers 5.x, so a
# decode-match would need brittle library patching. CodeT5+ is trained with an explicit EOS
# to avoid runaway, so raw greedy output is well-formed; report the null (parse-fail) rate as
# the comparison caveat. The grammar constraint remains a Qwen-side ablation.
# Other architectural difference: the 512-token encoder cap (Qwen sees 1024).
CONFIG="${1:-configs/codet5p_770m.yaml}"
SEED=42
DATASET="track2_both_pass_expanded"
DATA_DIR="data/seed${SEED}/${DATASET}"
TEST="${DATA_DIR}/test.jsonl"

# OUT_DIR follows the config's own output_dir, so the config is the single source
# of truth (train_seq2seq.py falls back to that same key when --output_dir is
# omitted). Read AFTER the cd: $CONFIG is relative to the project root. Falls back
# to the config basename if the key is missing; the seed layer is added here.

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

echo "==================  TRAIN  ${NAME}  (seed ${SEED})  =================="
train_start=$SECONDS
if { [ "${2:-}" = "skiptrain" ] || [ "${SKIP_TRAIN:-0}" = "1" ]; } && [ -f "${OUT_DIR}/config.json" ]; then
  echo "skiptrain requested and model exists at ${OUT_DIR} -- skipping training, re-generating only."
else
  python scripts/train_seq2seq.py \
    --config "$CONFIG" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUT_DIR"
  train_status=$?
  # The exit code, not the presence of a model file. A crash leaves the
  # PREVIOUS run's config.json in place, so the file check below passes and the
  # job would generate from stale weights while reporting success.
  if [ "$train_status" -ne 0 ]; then
    echo "TRAINING FAILED (exit ${train_status}) -- see logs/run-${SLURM_JOB_ID}.err. Not generating."
    exit 1
  fi
fi
train_elapsed=$((SECONDS - train_start))

# Guard: if training saved nothing (e.g. OOM before the first checkpoint),
# generate would fail with a confusing "can't load tokenizer" error instead.
if [ ! -f "${OUT_DIR}/config.json" ]; then
  echo "TRAINING FAILED -- no model saved to ${OUT_DIR} (see the .err log). Skipping generation."
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

echo "==================  GENERATE  ${NAME}  =================="
gen_start=$SECONDS
if ! python scripts/generate_seq2seq.py \
    --model_dir "$OUT_DIR" \
    --test_file "$TEST" \
    --out_file "$PREDICTIONS" \
    --max_new_tokens 512 \
    --trust_remote_code; then
  echo "GENERATION FAILED -- see logs/run-${SLURM_JOB_ID}.err."
  exit 1
fi
gen_elapsed=$((SECONDS - gen_start))

echo "==================  COMPLETION CHECK  =================="
expected=$(wc -l < "$TEST" | tr -d ' ')
if [ ! -s "$PREDICTIONS" ]; then
  echo "GENERATION INCOMPLETE -- output missing."; exit 1
fi
got=$(wc -l < "$PREDICTIONS" | tr -d ' ')
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
