"""
Evaluate a fine-tuned seq2seq model (CodeT5+) on a held-out test set.

Encoder-decoder counterpart of evaluate.py. Produces the SAME output schema
(eval_on_common.jsonl with reference_elixir_type / generated_elixir_type /
exact_match / semantic_distance / type_check_pass / ...), so compare_tracks.py
and the local extrinsic typecheck (type_migrator `mix eval_predictions`) work
unchanged across the Qwen and CodeT5+ runs.

  Tier A — exact match
  Tier B — semantic distance (0/1/2/3), via distance.py
  Tier C — executable typecheck (optional; --skip_typecheck to defer to the
           real-project injection step run locally)

Usage:
    python scripts/evaluate_seq2seq.py \\
        --model_dir runs/codet5p_track1_no_gradual \\
        --test_file data/track2_both_pass/test.jsonl \\
        --out_file runs/codet5p_track1_no_gradual/eval_on_common.jsonl \\
        --skip_typecheck
"""
import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from distance import semantic_distance
# Reuse the identical prompt + typecheck harness from the causal-LM evaluator.
from evaluate import format_prompt, parse_generated_type, build_mix_project, run_type_checker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="fine-tuned seq2seq model dir")
    ap.add_argument("--test_file", default="data/track2_both_pass/test.jsonl")
    ap.add_argument("--out_file", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--n_samples", type=int, default=0)
    ap.add_argument("--trust_remote_code", action="store_true",
                    help="needed for codet5p-2b and larger (custom modeling code)")
    ap.add_argument("--elixir_bin", default=None)
    ap.add_argument("--skip_typecheck", action="store_true",
                    help="skip in-process mix compile (defer to local injection step)")
    args = ap.parse_args()

    if not args.skip_typecheck and not args.elixir_bin:
        ap.error("--elixir_bin is required unless --skip_typecheck is given")

    out_file = args.out_file or Path(args.model_dir) / "eval_results.jsonl"

    print(f"=== Loading seq2seq model: {args.model_dir} ===")
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=args.trust_remote_code)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, trust_remote_code=args.trust_remote_code
    )

    # codet5p-2b lacks decoder_start_token_id/pad_token_id in its config;
    # generate() needs them. Mirror the values used at training time.
    def _first_set(*vals):
        for v in vals:
            if v is not None:
                return v
        return None
    c = model.config
    if getattr(c, "decoder_start_token_id", None) is None:
        c.decoder_start_token_id = _first_set(
            getattr(c, "bos_token_id", None),
            tok.bos_token_id, tok.pad_token_id, tok.eos_token_id,
        )
    if getattr(c, "pad_token_id", None) is None:
        c.pad_token_id = _first_set(tok.pad_token_id, tok.eos_token_id)

    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    with open(args.test_file) as f:
        test = [json.loads(l) for l in f if l.strip()]
    if args.n_samples > 0:
        test = test[: args.n_samples]
    n = len(test)
    print(f"=== Evaluating {n} entries ===")

    results = []
    em_count = pass_count = compile_error_count = timeout_count = 0

    with open(out_file, "w") as fout:
        for i, ex in enumerate(test):
            prompt = format_prompt(ex)
            inputs = tok(prompt, return_tensors="pt", truncation=True,
                         max_length=512).to(model.device)

            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                )
            # seq2seq output decodes to the TARGET only (no prompt echo).
            decoded = tok.decode(gen[0], skip_special_tokens=True)
            generated_type = parse_generated_type(decoded)

            reference = ex.get("elixir_type") or ""
            em = generated_type.strip() == reference.strip()
            em_count += int(em)

            distance, similarity, explanation = semantic_distance(generated_type, reference)

            if args.skip_typecheck:
                passed, tc_output = None, ""
            else:
                with tempfile.TemporaryDirectory(prefix="eval_") as tmp_root:
                    build_mix_project(tmp_root, ex, generated_type)
                    passed, tc_output = run_type_checker(tmp_root, args.elixir_bin)

            if passed == "compile_error":
                compile_error_count += 1
            elif passed in ("timeout", "error"):
                timeout_count += 1
            elif passed is True:
                pass_count += 1

            entry_result = {
                **ex,
                "reference_elixir_type": reference,
                "generated_elixir_type": generated_type,
                "exact_match": em,
                "semantic_distance": distance,
                "semantic_similarity": similarity,
                "distance_reason": explanation,
                "type_check_pass": passed,
                "type_check_output": tc_output[:500] if tc_output else "",
            }
            results.append(entry_result)
            fout.write(json.dumps(entry_result) + "\n")
            fout.flush()

            if (i + 1) % 20 == 0:
                d01 = sum(1 for r in results if r["semantic_distance"] <= 1)
                print(f"  [{i+1}/{n}] EM: {em_count} ({100*em_count/(i+1):.1f}%), "
                      f"D0+D1: {d01} ({100*d01/(i+1):.1f}%)")

    d_counts = Counter(r["semantic_distance"] for r in results)
    mpd = sum(r["semantic_distance"] for r in results) / n
    mss = sum(r["semantic_similarity"] for r in results) / n
    success_rate = 100 * (d_counts[0] + d_counts[1]) / n
    verifiable = n - compile_error_count - timeout_count

    print(f"\n=== Final ===")
    print(f"  Total:                   {n}")
    print(f"  Exact match:             {em_count} ({100*em_count/n:.1f}%)")
    print(f"  Distance 0 (perfect):    {d_counts[0]} ({100*d_counts[0]/n:.1f}%)")
    print(f"  Distance 1 (good):       {d_counts[1]} ({100*d_counts[1]/n:.1f}%)")
    print(f"  Distance 2 (partial):    {d_counts[2]} ({100*d_counts[2]/n:.1f}%)")
    print(f"  Distance 3 (failed):     {d_counts[3]} ({100*d_counts[3]/n:.1f}%)")
    print(f"  Mean Predicted Distance: {mpd:.3f}")
    print(f"  Success Rate (D0+D1):    {success_rate:.1f}%")
    if not args.skip_typecheck:
        print(f"  Compile errors:          {compile_error_count} (excluded)")
        print(f"  Timeouts/errors:         {timeout_count} (excluded)")
        print(f"  TypeCheck@1:             {pass_count}/{verifiable} "
              f"({100*pass_count/max(verifiable,1):.1f}%)")
    print(f"\n  Saved to: {out_file}")


if __name__ == "__main__":
    main()
