"""
Generate TypeSpec predictions with a fine-tuned causal-LM (Qwen) and write the
predictions dataset. This script only predicts; scoring happens in the
type_migrator project, which translates each predicted `@spec` into an Elixir
Type and compares it against the reference annotation:

    mix eval_typespec_predictions runs/<run>/eval_on_common.jsonl

Counterpart of generate.py (same model, Descr target). Two differences beyond
the prompt and the output field:

  * No grammar-constrained decoding. The Descr track constrains Qwen to
    descr_type.gbnf, which CodeT5+'s tokenizer cannot use -- an asymmetry that
    had to be reported as a limitation there. TypeSpec syntax has no grammar in
    this repo, so both models on this track decode unconstrained and the
    asymmetry disappears. The price is that malformed output is possible here
    for both, and it is counted the same way for both: a prediction that will
    not translate is a null, and a null is incompatible.

  * The output is parsed as a spec (see parse_generated_spec), not as a bare
    type expression.

Usage:
    python scripts/generate_typespec.py \\
        --adapter_dir runs/qwen7b_typespec/seed42 \\
        --test_file data/seed42/track2_both_pass_expanded/test.jsonl \\
        --out_file runs/qwen7b_typespec/seed42/eval_on_common.jsonl
"""
import argparse
import json
import re
import sys
from pathlib import Path

# Stream progress live even when stdout is redirected to a SLURM .out file
# (Python block-buffers a non-tty stdout, which makes a running job look hung).
sys.stdout.reconfigure(line_buffering=True)

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import prompt_typespec
from prompt_typespec import TARGET_FIELD, build_prompt as format_prompt
from prompt_logger import GenerationPromptLog, write_manifest

# A line that can only belong to the NEXT thing the model started writing, not
# to the spec: another attribute, a definition, or the module's end.
_NEXT_CONSTRUCT = re.compile(r"\n\s*(?:@(?:doc|moduledoc|type|opaque|typep|impl)|def\b|defp\b|end\b)")


def parse_generated_spec(generated_text):
    """The first complete `@spec` in the model's continuation.

    A causal model does not stop at the end of the spec: it carries on with the
    function, the next attribute, or a second spec. Everything after the first
    spec is cut, and the `@spec` prefix is re-attached when the model omitted it
    (having been given `### Output:` it sometimes writes the signature alone).
    Multi-line specs -- wrapped argument lists, `when` guards -- are preserved:
    only a blank line or the start of another construct terminates the spec.
    """
    text = generated_text
    for marker in ["<|endoftext|>", "<|im_end|>", "\n###"]:
        idx = text.find(marker)
        if idx >= 0:
            text = text[:idx]

    # Strip a code fence if the model wrapped its answer in one.
    text = re.sub(r"```\w*\n?", "", text)

    # Start at the first @spec; keep only that one.
    first = text.find("@spec")
    if first >= 0:
        text = text[first:]
        second = text.find("@spec", len("@spec"))
        if second > 0:
            text = text[:second]

    # Stop at a blank line or at whatever construct comes next.
    blank = text.find("\n\n")
    if blank > 0:
        text = text[:blank]
    if (m := _NEXT_CONSTRUCT.search(text)) is not None:
        text = text[: m.start()]

    text = text.strip()
    if text and not text.startswith("@spec"):
        text = "@spec " + text
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-Coder-7B")
    ap.add_argument("--test_file", default="data/seed42/track2_both_pass_expanded/test.jsonl")
    ap.add_argument("--out_file", default=None)
    # Specs are short (median 62 characters, longest 1178 in the both-pass pool),
    # so 256 tokens is ample; the Descr track needs 1024 for its expanded types.
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--n_samples", type=int, default=0)
    # Damp degenerate enumeration loops (the model extending a union or keyword
    # list until it is cut mid-token). Mild values leave real specs intact.
    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0)
    args = ap.parse_args()

    out_file = args.out_file or Path(args.adapter_dir) / "predictions.jsonl"

    print(f"=== Loading base model: {args.base_model} ===")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb,
        device_map="auto", torch_dtype=torch.bfloat16,
    )
    print(f"=== Loading adapter: {args.adapter_dir} ===")
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    with open(args.test_file) as f:
        test = [json.loads(l) for l in f if l.strip()]
    if args.n_samples > 0:
        test = test[: args.n_samples]
    n = len(test)
    print(f"=== Generating {n} TypeSpec predictions (unconstrained decoding) ===")

    empty_count = 0

    # Prompts are logged beside the predictions, with the raw continuation kept
    # next to the parsed spec: a parser bug then shows up as a sound
    # continuation beside a bad parse, which the predictions file alone cannot
    # distinguish from a bad prediction.
    log_dir = Path(out_file).parent
    write_manifest(
        log_dir, prompt_typespec,
        phase="generate:causal",
        extra={
            "adapter_dir": args.adapter_dir,
            "base_model": args.base_model,
            "test_file": args.test_file,
            "out_file": str(out_file),
            "entries": n,
            "constrained_decoding": False,
            "max_new_tokens": args.max_new_tokens,
        },
        example=format_prompt(test[0]) if test else None,
    )

    with open(out_file, "w") as fout, GenerationPromptLog(log_dir) as plog:
        for i, ex in enumerate(test):
            prompt = format_prompt(ex)
            inputs = tok(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                )
            full = tok.decode(gen[0], skip_special_tokens=False)
            prompt_len = len(tok.decode(inputs.input_ids[0], skip_special_tokens=False))
            raw_output = full[prompt_len:]
            generated_spec = parse_generated_spec(raw_output)
            if not generated_spec:
                empty_count += 1

            plog.write(index=i, prompt=prompt, raw_output=raw_output,
                       parsed=generated_spec, reference=ex.get(TARGET_FIELD), entry=ex)

            # Carry the FULL source entry through: scoring needs `elixir_type`
            # (the reference annotation), `spec` (the reference spec, for the
            # ceiling), `type` and `module` (to rebuild the translation context).
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
    print(f"  Saved to: {out_file}")
    print(f"  Score with: mix eval_typespec_predictions {out_file}")


if __name__ == "__main__":
    main()
