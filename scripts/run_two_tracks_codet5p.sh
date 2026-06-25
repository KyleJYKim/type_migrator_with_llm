#!/usr/bin/env bash
# CodeT5+ (seq2seq) counterpart of run_two_tracks.sh.
# Trains BOTH tracks with full fine-tuning, then generates predictions for BOTH
# on the SAME common held-out set, for an apples-to-apples comparison with each
# other and with the Qwen runs.
#
# IMPORTANT: this reuses the EXISTING data/track{1,2} splits (seed 42) that the
# Qwen models were trained/evaluated on -> it does NOT re-run prepare_data, so
# the held-out functions are identical across all models.
#
# Run on a GPU compute node (full FT needs CUDA). e.g. on LIP6:
#   salloc -p ... --gres=gpu:a100_7g.80gb:1 -c 8 --mem=64G -t 04:00:00
#   srun --pty bash
#   bash scripts/run_two_tracks_codet5p.sh [CONFIG]
#
# The typecheck (the paper's TC%/S&P% numbers) is run SEPARATELY and locally
# afterwards, the same way as for Qwen: inject generated_elixir_type from each
# eval_on_common.jsonl into the real project and recompile.
set -euo pipefail
cd "$(dirname "$0")/.."

# Reduce CUDA fragmentation (the "reserved but unallocated" OOM headroom).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG="${1:-configs/codet5p_2b.yaml}"
TRACKS=(track1_no_gradual track2_both_pass)
COMMON_TEST="data/track2_both_pass/test.jsonl"   # superset; sliced by compare_tracks.py

# codet5p-2b/6b need trust_remote_code; detect from the config.
TRUST=""
grep -q 'trust_remote_code: *true' "$CONFIG" && TRUST="--trust_remote_code"

# 1) train each track (full fine-tuning) on its own split
for t in "${TRACKS[@]}"; do
  echo "==================  TRAIN  codet5p  ${t}  =================="
  python scripts/train_seq2seq.py \
    --config "$CONFIG" \
    --data_dir "data/${t}" \
    --output_dir "runs/codet5p_${t}"
done

# 2) generate predictions for BOTH on the SAME common held-out set
for t in "${TRACKS[@]}"; do
  echo "==================  GENERATE   codet5p  ${t}  (common test)  =================="
  python scripts/generate_seq2seq.py \
    --model_dir "runs/codet5p_${t}" \
    --test_file "$COMMON_TEST" \
    --out_file "runs/codet5p_${t}/eval_on_common.jsonl" \
    $TRUST
done

# 3) comparison (exact-match now; TC%/S&P% filled later).
# NOTE: TC%/S&P% columns will read 0 here because type_check_pass is still null;
# they are filled by the separate typecheck step before final comparison.
echo "==================  COMPARISON  =================="
python scripts/compare_tracks.py \
  runs/codet5p_track1_no_gradual/eval_on_common.jsonl \
  runs/codet5p_track2_both_pass/eval_on_common.jsonl

echo
echo "Predictions written to runs/codet5p_*/eval_on_common.jsonl"
echo "Next (locally): run the typecheck on those two files to fill"
echo "type_check_pass, then re-run compare_tracks.py for TC% / S&P% / emit-dyn%."
