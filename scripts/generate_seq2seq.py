"""
Generate type predictions with a fine-tuned seq2seq model (CodeT5+) on a held-out
test set. Encoder-decoder counterpart of generate.py: produces the SAME predictions
dataset schema (elixir_type / generated_elixir_type / reference_over_budget / ...),
so the downstream steps work unchanged across the Qwen and CodeT5+ runs. This script
only predicts; scoring (typecheck, set-theoretic distance) is done separately.

Usage:
    python scripts/generate_seq2seq.py \\
        --model_dir runs/codet5p_track1_no_gradual \\
        --test_file data/track2_both_pass/test.jsonl \\
        --out_file runs/codet5p_track1_no_gradual/predictions.jsonl
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Reuse the identical prompt + parsing from the causal-LM generator.
from generate import format_prompt, parse_generated_type


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="fine-tuned seq2seq model dir")
    ap.add_argument("--test_file", default="data/track2_both_pass/test.jsonl")
    ap.add_argument("--out_file", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--n_samples", type=int, default=0)
    ap.add_argument("--trust_remote_code", action="store_true",
                    help="needed for codet5p-2b and larger (custom modeling code)")
    args = ap.parse_args()

    out_file = args.out_file or Path(args.model_dir) / "predictions.jsonl"

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

    # generate() -> _prepare_generation_config() also calls
    # config._get_non_default_generation_parameters() -> self.__class__(), which
    # codet5p's custom config asserts on. Bypass it (same as in train_seq2seq.py).
    try:
        type(c)._get_non_default_generation_parameters = lambda self: {}
    except Exception:
        pass

    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    with open(args.test_file) as f:
        test = [json.loads(l) for l in f if l.strip()]
    if args.n_samples > 0:
        test = test[: args.n_samples]
    n = len(test)
    print(f"=== Generating predictions for {n} entries ===")

    over_budget_count = 0

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
            ref_token_len = len(tok(reference).input_ids)
            reference_over_budget = ref_token_len > args.max_new_tokens
            if reference_over_budget:
                over_budget_count += 1

            record = {
                **ex,
                "generated_elixir_type": generated_type,
                "reference_token_len":   ref_token_len,
                "reference_over_budget": reference_over_budget,
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{n}] generated")

    print(f"\n=== Done ===")
    print(f"  Total:                              {n}")
    print(f"  Over budget (ref > {args.max_new_tokens} tok): {over_budget_count}")
    print(f"  Saved to: {out_file}")


if __name__ == "__main__":
    main()
