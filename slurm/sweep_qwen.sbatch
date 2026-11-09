#!/bin/bash
#SBATCH --job-name=ET-Qwen-sweep
#SBATCH --qos=normal
#SBATCH --gpus=a100_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/sweep-%A_%a.out
#SBATCH --error=logs/sweep-%A_%a.err
#
# Checkpoint-selection study, step 2: signals + grammar-constrained generation
# on the validation subset for every checkpoint of one run. Array task i takes
# every k-th checkpoint (k = array size), so --array=0-3 splits the work 4 ways.
#   sbatch --array=0-3 slurm/sweep_qwen.sbatch runs/sweep/qwen7b_completion_instr1/seed42 1
# Re-submitting resumes: finished checkpoints are skipped.
# Then, LOCALLY:  scripts/score_sweep.sh <run_dir>  &&  python scripts/collect_sweep.py <run_dir>
RUN_DIR="${1:?run dir}"
INSTR="${2:-1}"
DATA_DIR="data/seed42/track2_both_pass_expanded"
K="${SLURM_ARRAY_TASK_COUNT:-1}"
I="${SLURM_ARRAY_TASK_ID:-0}"

eval "$(conda shell.bash hook)"
conda activate etr
cd ~/elixir_types_trainer/type_migrator_with_llm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ETR_INCLUDE_INSTRUCTION="$INSTR"

python scripts/sweep_checkpoints.py --run_dir "$RUN_DIR" --data_dir "$DATA_DIR" \
  --n 400 --constrain --shard "${I}/${K}"
