#!/bin/bash
#SBATCH --job-name=run-two-tracks
#SBATCH --qos=normal
#SBATCH --gpus=a100_7g.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/run-%j.out
#SBATCH --error=logs/run-%j.err

CONFIG="${1:-configs/qwen7b_qlora.yaml}"
TRACKS=(track2_both_pass_compacted track2_both_pass_hedged track2_both_pass_both track2_both_pass_expanded)
COMMON_TEST="data/track2_both_pass_expanded/test.jsonl"

eval "$(conda shell.bash hook)"
conda activate etr

cd ~/elixir_types_trainer/type_migrator_with_llm

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for t in "${TRACKS[@]}"; do
  echo "==================  TRAIN  ${t}  =================="
  python scripts/train_sft.py \
    --config "$CONFIG" \
    --data_dir "data/${t}" \
    --output_dir "runs/${t}"
done

for t in "${TRACKS[@]}"; do
  echo "==================  GENERATE   ${t}  (common test)  =================="
  python scripts/generate.py \
    --adapter_dir "runs/${t}" \
    --test_file "$COMMON_TEST" \
    --out_file "runs/${t}/eval_on_common.jsonl" \
    --constrain
done

echo "==================  COMPLETION CHECK (generation)  =================="
# A complete run writes exactly one prediction line per test entry; a crashed
# or timed-out generation leaves a short file. Fail the job (non-zero exit) if
# any variant is missing or short, so an incomplete run is never mistaken for
# a finished one.
expected=$(wc -l < "$COMMON_TEST" | tr -d ' ')
incomplete=0
for t in "${TRACKS[@]}"; do
  f="runs/${t}/eval_on_common.jsonl"
  if [ ! -s "$f" ]; then
    echo "  MISSING   ${t}"
    incomplete=1
  else
    got=$(wc -l < "$f" | tr -d ' ')
    if [ "$got" -eq "$expected" ]; then
      echo "  OK        ${t}  (${got}/${expected})"
    else
      echo "  PARTIAL   ${t}  (${got}/${expected})"
      incomplete=1
    fi
  fi
done

if [ "$incomplete" -ne 0 ]; then
  echo "GENERATION INCOMPLETE -- see above."
  exit 1
fi

echo "Done. Generation complete for all ${#TRACKS[@]} adapters: runs/*/eval_on_common.jsonl"
echo "Next, LOCALLY (needs the custom Elixir toolchain), score each variant with:"
echo "  mix eval_predictions set-theoretic <trainer>/runs/track2_both_pass_<variant>/eval_on_common.jsonl"