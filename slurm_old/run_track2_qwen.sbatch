#!/bin/bash
#SBATCH --job-name=run-track2
#SBATCH --qos=normal
#SBATCH --gpus=a100_7g.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/run-%j.out
#SBATCH --error=logs/run-%j.err

CONFIG="${1:-configs/qwen7b_qlora.yaml}"
TRACK="track2_both_pass"
COMMON_TEST="data/track2_both_pass/test.jsonl"

eval "$(conda shell.bash hook)"
conda activate etr

cd ~/elixir_types_trainer/type_migrator_with_llm

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "==================  TRAIN  $TRACK  =================="
python scripts/train_sft.py \
  --config "$CONFIG" \
  --data_dir "data/$TRACK" \
  --output_dir "runs/$TRACK"

echo "==================  GENERATE   $TRACK  (common test)  =================="
python scripts/generate.py \
  --adapter_dir "runs/$TRACK" \
  --test_file "$COMMON_TEST" \
  --out_file "runs/$TRACK/eval_on_common.jsonl" \
  --constrain

echo "Done. Per-adapter results: runs/$TRACK/eval_on_common.jsonl"

SBATCH