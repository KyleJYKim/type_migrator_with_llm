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
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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
    print(f"=== Generating predictions for {n} entries ===")

    over_budget_count = 0

    with open(out_file, "w") as fout:
        for i, ex in enumerate(test):
            prompt = format_prompt(ex)
            inputs = tok(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
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
