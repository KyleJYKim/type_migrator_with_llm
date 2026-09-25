"""
Generate TypeSpec predictions with a fine-tuned seq2seq model (CodeT5+).

Encoder-decoder counterpart of generate_typespec.py, producing the SAME record
schema (`generated_spec` beside the full source entry), so the scoring step is
identical for both models:

    mix eval_typespec_predictions runs/<run>/eval_on_common.jsonl

Both models on this track decode unconstrained, so unlike the Descr track there
is no decoding asymmetry between them to qualify the comparison.

Usage:
    python scripts/generate_seq2seq_typespec.py \\
        --model_dir runs/codet5p_770m_typespec/seed42 \\
        --test_file data/seed42/track2_both_pass_expanded/test.jsonl \\
        --out_file runs/codet5p_770m_typespec/seed42/eval_on_common.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Stream progress live even when stdout is redirected to a SLURM .out file.
sys.stdout.reconfigure(line_buffering=True)

# Shared prompt and spec parsing, so the two generators of this track cannot drift.
import prompt_typespec
from prompt_typespec import TARGET_FIELD, build_prompt as format_prompt
from generate_typespec import parse_generated_spec
from prompt_logger import GenerationPromptLog, write_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="fine-tuned seq2seq model dir")
    ap.add_argument("--test_file", default="data/seed42/track2_both_pass_expanded/test.jsonl")
    ap.add_argument("--out_file", default=None)
    # Encoder cap. The longest prompt in the corpus is ~1,320 tokens (3,952
    # characters at a conservative 3 chars/token), so 1536 truncates nothing --
    # the point is to STOP truncating. It was hardcoded to 512, which silently
    # cut the tail off ~3-5% of prompts; T5 uses relative position embeddings so
    # it has no architectural 512 limit, but the TRAINING config must use the
    # same value or the model meets longer inputs than it ever saw.
    ap.add_argument("--max_source_length", type=int, default=1536,
                    help="encoder cap; must match the training config")
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
    # config._get_non_default_generation_parameters(), which codet5p's custom
    # config asserts on. Bypass it (same as in the trainer).
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
    print(f"=== Generating {n} TypeSpec predictions (unconstrained decoding) ===")

    empty_count = 0
    truncated_count = 0

    # Same logging as the causal generator, so both halves of the track are
    # auditable the same way. Note the encoder truncation below: a prompt longer
    # than 512 tokens is cut, and the log records the prompt BEFORE truncation,
    # so a silently shortened input is visible by comparing the two.
    log_dir = Path(out_file).parent
    write_manifest(
        log_dir, prompt_typespec,
        phase="generate:seq2seq",
        extra={
            "model_dir": args.model_dir,
            "test_file": args.test_file,
            "out_file": str(out_file),
            "entries": n,
            "constrained_decoding": False,
            "max_new_tokens": args.max_new_tokens,
            "max_source_length": args.max_source_length,
        },
        example=format_prompt(test[0]) if test else None,
    )

    with open(out_file, "w") as fout, GenerationPromptLog(log_dir) as plog:
        for i, ex in enumerate(test):
            prompt = format_prompt(ex)
            full_len = len(tok(prompt).input_ids)
            if full_len > args.max_source_length:
                truncated_count += 1
                print(f"  [{i+1}/{n}] TRUNCATED: prompt {full_len} tokens "
                      f"> --max_source_length {args.max_source_length}")
            inputs = tok(prompt, return_tensors="pt", truncation=True,
                         max_length=args.max_source_length).to(model.device)

            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                )
            # seq2seq output decodes to the TARGET only (no prompt echo).
            decoded = tok.decode(gen[0], skip_special_tokens=True)
            generated_spec = parse_generated_spec(decoded)
            if not generated_spec:
                empty_count += 1

            plog.write(index=i, prompt=prompt, raw_output=decoded,
                       parsed=generated_spec, reference=ex.get(TARGET_FIELD), entry=ex)

            record = {
                **ex,
                "generated_spec": generated_spec,
                "reference_spec_token_len": len(tok(ex.get(TARGET_FIELD) or "").input_ids),
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{n}] generated")

    print(f"\n=== Done ===")
    print(f"  Total:            {n}")
    print(f"  Empty prediction: {empty_count}")
    print(f"  Truncated prompts: {truncated_count}  (should be 0; raise --max_source_length and RETRAIN if not)")
    print(f"  Saved to: {out_file}")
    print(f"  Score with: mix eval_typespec_predictions {out_file}")


if __name__ == "__main__":
    main()
