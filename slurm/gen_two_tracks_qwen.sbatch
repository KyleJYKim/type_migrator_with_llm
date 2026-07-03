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
TRACKS=(track1_no_gradual track2_both_pass)
COMMON_TEST="data/track2_both_pass/test.jsonl"

eval "$(conda shell.bash hook)"
conda activate etr

cd ~/elixir_types_trainer/type_migrator_with_llm

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for t in "${TRACKS[@]}"; do
  echo "==================  GENERATE   ${t}  (common test)  =================="
  python scripts/generate.py \
    --adapter_dir "runs/${t}" \
    --test_file data/track2_both_pass/test.jsonl \
    --out_file runs/$t/eval_on_common.jsonl --constrain
done

echo "==================  COMPARISON  =================="
python scripts/compare_tracks.py \
  runs/track1_no_gradual/eval_on_common.jsonl \
  runs/track2_both_pass/eval_on_common.jsonl

echo "Done. Per-adapter results: runs/*/eval_on_common.jsonl"

SBATCH