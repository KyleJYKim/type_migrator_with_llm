#!/bin/bash
#SBATCH --job-name=ET-Qwen-masked
#SBATCH --qos=normal
#SBATCH --gpus=a100_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/masked-%j.out
#SBATCH --error=logs/masked-%j.err
#
# Checkpoint-selection study, step 1: train one arm, keep every checkpoint.
#   sbatch slurm/train_masked_qwen.sbatch completion 1   # completion-only loss, with instruction
#   sbatch slurm/train_masked_qwen.sbatch full 1         # full-sequence loss,  with instruction
#   sbatch slurm/train_masked_qwen.sbatch completion 0   # ... without instruction
#   sbatch slurm/train_masked_qwen.sbatch full 0
# Then: sbatch --array=0-3 slurm/sweep_qwen.sbatch <run_dir> <instr>
LOSS="${1:?loss: full|completion}"
INSTR="${2:-1}"
SEED=42
DATA_DIR="data/seed${SEED}/track2_both_pass_expanded"
RUN_DIR="runs/sweep/qwen7b_${LOSS}_instr${INSTR}/seed${SEED}"

eval "$(conda shell.bash hook)"
conda activate etr
cd ~/elixir_types_trainer/type_migrator_with_llm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ETR_INCLUDE_INSTRUCTION="$INSTR"

if [ -d "$RUN_DIR" ] && ls "$RUN_DIR"/checkpoint-* >/dev/null 2>&1; then
  echo "$RUN_DIR already has checkpoints; move it away first."; exit 1
fi

# Create the run dir here, in the shell, before Python does. Four arms submitted
# together all create runs/sweep at once; on the cluster's shared filesystem a
# node can briefly not see the folder another node just made, and Python's
# mkdir then fails with FileExistsError. `mkdir -p` (plus a short retry) is safe.
# Listing the parents first makes the node revalidate its cached view of them
# (a stale "does not exist" entry is what makes mkdir fail with EEXIST).
for _ in 1 2 3 4 5 6; do
  ls -a runs runs/sweep > /dev/null 2>&1
  mkdir -p "$RUN_DIR" 2>/dev/null && [ -d "$RUN_DIR" ] && break
  sleep 10
done
[ -d "$RUN_DIR" ] || { echo "cannot create $RUN_DIR"; exit 1; }

echo "==========  TRAIN  loss=${LOSS}  instruction=${INSTR}  ->  ${RUN_DIR}  =========="
python scripts/train_sft_masked.py --config configs/qwen7b_qlora_sweep.yaml \
  --data_dir "$DATA_DIR" --output_dir "$RUN_DIR" --loss "$LOSS" || { echo "TRAINING FAILED"; exit 1; }

# Write the shared validation subset now, once, so the sweep array never races on it.
python scripts/sweep_checkpoints.py --run_dir "$RUN_DIR" --data_dir "$DATA_DIR" --prepare_only
echo "Next:  sbatch --array=0-3 slurm/sweep_qwen.sbatch $RUN_DIR $INSTR"
