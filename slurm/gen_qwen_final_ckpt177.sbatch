#!/bin/bash
#SBATCH --job-name=gen-qwen-final
#SBATCH --qos=normal
#SBATCH --gpus=a100_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/gen-qwen-final-%j.out
#SBATCH --error=logs/gen-qwen-final-%j.err

# One-off (temp): regenerate Qwen predictions from the FINAL-weights checkpoint
# (checkpoint-177) instead of the CE-best top-level adapter (checkpoint-100).
# Rationale: metric_for_best_model=eval_loss selects too early for this 7B model,
# so the CE-best checkpoint scores WORSE on set-theoretic compatibility (36.6%)
# than the final weights (~39.5%). Everything below is identical to the normal
# Qwen generation except --adapter_dir (-> checkpoint-177) and --out_file
# (distinct name, so the checkpoint-100 outputs are not clobbered).

ADAPTER="runs/qwen7b_qlora/seed42/checkpoint-177"
TEST="data/track2_both_pass/test.jsonl"           # the 2127-entry common test set
OUT="runs/qwen7b_qlora/seed42/eval_on_common.final.jsonl"

eval "$(conda shell.bash hook)"
conda activate etr

cd ~/elixir_types_trainer/type_migrator_with_llm

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fail early if the final-weights checkpoint is missing (e.g. save_total_limit
# rotated it out on a run where it wasn't null).
if [ ! -d "$ADAPTER" ]; then
  echo "ERROR: $ADAPTER not found. Available checkpoints:"
  ls -d runs/qwen7b_qlora/seed42/checkpoint-* 2>/dev/null
  exit 1
fi

echo "==================  GENERATE  qwen final-weights ($ADAPTER, grammar-constrained)  =================="
python scripts/generate.py \
  --adapter_dir "$ADAPTER" \
  --base_model Qwen/Qwen2.5-Coder-7B \
  --test_file "$TEST" \
  --out_file "$OUT" \
  --constrain

echo "Done. Raw predictions: $OUT"
echo "Score LOCALLY (needs the Elixir toolchain):"
echo "  mix eval_predictions set-theoretic <trainer>/$OUT"
