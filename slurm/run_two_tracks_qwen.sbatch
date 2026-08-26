#!/bin/bash
#SBATCH --job-name=run-two-tracks
#SBATCH --qos=normal
#SBATCH --gpus=a100_7g.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/run-%j.out
#SBATCH --error=logs/run-%j.err

# Seed-variance study: each seed reshuffles the subproject -> split assignment
# (data/seed${SEED}/...), so retraining per seed on its own test set turns a
# single-split point estimate into a mean +/- std across splits. Only the two
# branches that carry the thesis claim are trained (expanded baseline vs hedged).
#
# Usage:
#   sbatch run_two_tracks_qwen.sbatch                       # all seeds, one long job
#   sbatch run_two_tracks_qwen.sbatch configs/qwen7b_qlora.yaml "42"   # one seed
# Running one seed per job (5 jobs) keeps each within a ~10h wall clock and lets
# them run in parallel; the default all-in-one needs the larger --time above.
CONFIG="${1:-configs/qwen7b_qlora.yaml}"
TRACKS=(track2_both_pass_expanded track2_both_pass_hedged)

eval "$(conda shell.bash hook)"
conda activate etr

cd ~/elixir_types_trainer/type_migrator_with_llm

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


COMMON_TEST="data/track2_both_pass_expanded/test.jsonl"

for t in "${TRACKS[@]}"; do
  echo "==================  TRAIN ${t}  =================="
  python scripts/train_sft.py \
    --config "$CONFIG" \
    --data_dir "data/${t}" \
    --output_dir "runs/${t}"
done

for t in "${TRACKS[@]}"; do
  echo "==================  GENERATE ${t}  (own test)  =================="
  python scripts/generate.py \
    --adapter_dir "runs/${t}" \
    --test_file "$COMMON_TEST" \
    --out_file "runs/${t}/eval_on_common.jsonl" \
    --constrain
done

echo "Done. Generation complete for ${#TRACKS[@]} branches: runs/track2_both_pass_*/eval_on_common.jsonl"
