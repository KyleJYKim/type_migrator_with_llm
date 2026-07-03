"""
Generate type predictions for a fine-tuned causal-LM (Qwen) on test.jsonl and
write the predictions dataset. This script only predicts; scoring is done by
separate steps:
  - the typecheck (type_migrator: inject generated_elixir_type into the real
    project and recompile), and
  - the set-theoretic distance (SetTheoreticEvaluator, Descr module).
"""
import argparse
import json
import sys
from pathlib import Path

# Stream progress live even when stdout is redirected to a SLURM .out file
# (Python block-buffers a non-tty stdout, which makes a running job look hung).
sys.stdout.reconfigure(line_buffering=True)

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LogitsProcessorList,
)


def format_prompt(example):
    type_block = ""
    if example.get("type"):
        if isinstance(example["type"], list):
            type_block = "\n".join(example["type"])
        else:
            type_block = str(example["type"])
    return (
        f"### Module: {example['module']}\n"
        f"### Types in scope:\n{type_block}\n\n"
        f"### Definition:\n{example['definition']}\n\n"
        f"### Elixir type:\n"
    )


def parse_generated_type(generated_text):
    for marker in ["<|endoftext|>", "<|im_end|>", "\n###", "\n\n"]:
        idx = generated_text.find(marker)
        if idx > 0:
            generated_text = generated_text[:idx]
    return generated_text.strip()


def build_grammar_processor(grammar_path, hf_tokenizer):
    """Return a LogitsProcessor that constrains generation to the descr type
    grammar (scripts/descr_type.lark), so the model can only emit well-formed,
    descr-only types --- no missing delimiters, no unterminated atoms, no drift to
    TypeSpec forms. Uses `outlines` (pip install outlines lark). The tokenizer-wrapper
    API is version-sensitive, so it is isolated here for easy adjustment.
    """
    from outlines.processors import CFGLogitsProcessor
    try:
        from outlines.models.transformers import TransformerTokenizer
        otok = TransformerTokenizer(hf_tokenizer)
    except Exception:
        otok = hf_tokenizer  # newer outlines accept the HF tokenizer directly
    return CFGLogitsProcessor(open(grammar_path).read(), otok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-Coder-7B")
    ap.add_argument("--test_file", default="data/test.jsonl")
    ap.add_argument("--out_file", default=None)
    # Generation budget. Qwen trains at max_seq_length=1024 (prompt+completion), so it
    # can emit types well past the old 256 cap; 1024 covers long-but-reasonable types
    # without inviting runaway greedy generation. Override per-run as needed.
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--n_samples", type=int, default=0)
    # Break the degenerate-repetition loops (the model enumerating an ever-growing
    # struct/keyword list until it is cut mid-token, yielding an unparseable type).
    # repetition_penalty>1 discourages reusing tokens; no_repeat_ngram_size forbids
    # repeating any n-gram (0 = off). Mild values keep legitimately repetitive struct
    # types intact; raise them if truncation persists.
    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0)
    # Grammar-constrained decoding: force the output to be a well-formed descr type
    # (scripts/descr_type.lark). Eliminates malformed/unparseable output and TypeSpec
    # drift by construction, so the repetition_penalty above is bypassed when this is on.
    ap.add_argument("--constrain", action="store_true",
                    help="constrain decoding to the descr type grammar (needs `outlines`)")
    ap.add_argument("--grammar",
                    default=str(Path(__file__).resolve().parent / "descr_type.lark"))
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

    grammar_processor = None
    if args.constrain:
        print(f"=== Grammar-constrained decoding: {args.grammar} ===")
        grammar_processor = build_grammar_processor(args.grammar, tok)

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
            inputs = tok(prompt, return_tensors="pt").to(model.device)

            gen_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
            if grammar_processor is not None:
                # Grammar guarantees validity; the repetition_penalty hack is unneeded
                # (and harmful to structured output), so it is omitted on this path.
                gen_kwargs["logits_processor"] = LogitsProcessorList([grammar_processor])
            else:
                gen_kwargs["repetition_penalty"] = args.repetition_penalty
                gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size

            with torch.no_grad():
                gen = model.generate(**inputs, **gen_kwargs)
            full = tok.decode(gen[0], skip_special_tokens=False)
            prompt_len = len(tok.decode(inputs.input_ids[0], skip_special_tokens=False))
            generated_type = parse_generated_type(full[prompt_len:])

            reference = ex.get("elixir_type") or ""
            # A reference longer than the generation budget cannot be emitted in full
            # by any model; flag it so scoring can bucket it rather than count it as a
            # plain miss.
            ref_token_len = len(tok(reference).input_ids)
            reference_over_budget = ref_token_len > args.max_new_tokens
            if reference_over_budget:
                over_budget_count += 1

            # Carry the FULL source entry (file, definition, type, spec, line locators,
            # elixir_type, ...) so this jsonl doubles as the predictions dataset for the
            # typecheck step: injecting generated_elixir_type back into the real project
            # and recompiling.
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
