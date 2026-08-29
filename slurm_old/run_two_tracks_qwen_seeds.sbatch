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
SEEDS=(42 1 7 123 2024)
if [ -n "$2" ]; then SEEDS=($2); fi
TRACKS=(track2_both_pass_expanded track2_both_pass_hedged)

eval "$(conda shell.bash hook)"
conda activate etr

cd ~/elixir_types_trainer/type_migrator_with_llm

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for SEED in "${SEEDS[@]}"; do
  COMMON_TEST="data/seed${SEED}/track2_both_pass_expanded/test.jsonl"
  if [ ! -s "$COMMON_TEST" ]; then
    echo "MISSING SPLIT for seed ${SEED}: ${COMMON_TEST} -- run 'python scripts/prepare_data.py ${SEED}' first."
    exit 1
  fi

  for t in "${TRACKS[@]}"; do
    echo "==================  TRAIN  seed ${SEED}  ${t}  =================="
    python scripts/train_sft.py \
      --config "$CONFIG" \
      --data_dir "data/seed${SEED}/${t}" \
      --output_dir "runs/seed${SEED}/${t}"
  done

  for t in "${TRACKS[@]}"; do
    echo "==================  GENERATE  seed ${SEED}  ${t}  (own test)  =================="
    python scripts/generate.py \
      --adapter_dir "runs/seed${SEED}/${t}" \
      --test_file "$COMMON_TEST" \
      --out_file "runs/seed${SEED}/${t}/eval_on_common.jsonl" \
      --constrain
  done
done

echo "==================  COMPLETION CHECK (generation)  =================="
# A complete run writes exactly one prediction line per test entry; a crashed
# or timed-out generation leaves a short file. Fail the job (non-zero exit) if
# any variant is missing or short, so an incomplete run is never mistaken for
# a finished one.
incomplete=0
for SEED in "${SEEDS[@]}"; do
  expected=$(wc -l < "data/seed${SEED}/track2_both_pass_expanded/test.jsonl" | tr -d ' ')
  for t in "${TRACKS[@]}"; do
    f="runs/seed${SEED}/${t}/eval_on_common.jsonl"
    if [ ! -s "$f" ]; then
      echo "  MISSING   seed ${SEED}  ${t}"
      incomplete=1
    else
      got=$(wc -l < "$f" | tr -d ' ')
      if [ "$got" -eq "$expected" ]; then
        echo "  OK        seed ${SEED}  ${t}  (${got}/${expected})"
      else
        echo "  PARTIAL   seed ${SEED}  ${t}  (${got}/${expected})"
        incomplete=1
      fi
    fi
  done
done

if [ "$incomplete" -ne 0 ]; then
  echo "GENERATION INCOMPLETE -- see above."
  exit 1
fi

echo "Done. Generation complete for ${#SEEDS[@]} seed(s) x ${#TRACKS[@]} branches: runs/seed*/track2_both_pass_*/eval_on_common.jsonl"
echo "Next, LOCALLY (needs the custom Elixir toolchain), score each seed x branch with:"
echo "  mix eval_predictions set-theoretic <trainer>/runs/seed<SEED>/track2_both_pass_<variant>/eval_on_common.jsonl"
echo "then aggregate compatibility across seeds as mean +/- std per branch."