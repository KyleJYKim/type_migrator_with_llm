"""
Evaluate the fine-tuned model on test.jsonl using executable type-checking.

For each test entry:
  1. Build the same prompt format as training
  2. Generate the elixir_type completion
  3. Write a temporary Elixir file with the original definition + generated type
  4. Run the type checker on it
  5. Record pass/fail

Outputs results to runs/<run_name>/eval_results.jsonl
"""
import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def format_prompt(example):
    """Same as training but without the completion."""
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
    """The model continues after '### Elixir type:\n'.
    Extract everything up to <|endoftext|> or first blank line."""
    # Remove EOT and anything after
    for marker in ["<|endoftext|>", "<|im_end|>", "\n###", "\n\n"]:
        idx = generated_text.find(marker)
        if idx > 0:
            generated_text = generated_text[:idx]
    return generated_text.strip()


def run_type_checker(definition, generated_type, module_name, type_block, elixir_bin):
    """Build a minimal Elixir module and run the type checker on it.
    Returns (pass: bool, output: str)."""
    safe_module = re.sub(r"[^\w]", "_", module_name) + "_Eval"

    source = f"""defmodule {safe_module} do
  {type_block}

  @assert_type_form {generated_type}
  {definition}
end
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ex", delete=False) as f:
        f.write(source)
        path = f.name

    try:
        env = os.environ.copy()
        env["PATH"] = f"{elixir_bin}:{env.get('PATH', '')}"
        result = subprocess.run(
            ["elixir", path],
            capture_output=True, text=True, timeout=30, env=env,
        )
        # Heuristic: type checker emits "type warning found at" on failures
        passed = "type warning found at" not in result.stderr and \
                 "type warning found at" not in result.stdout
        return passed, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, f"ERROR: {e}"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True,
                    help="Path to checkpoint directory, e.g. runs/qwen7b_qlora_v1/checkpoint-20")
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-Coder-7B")
    ap.add_argument("--test_file", default="data/test.jsonl")
    ap.add_argument("--out_file", default=None,
                    help="Defaults to <adapter_dir>/eval_results.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--n_samples", type=int, default=0,
                    help="If > 0, evaluate only the first N test entries")
    ap.add_argument("--elixir_bin", default="",
                    help="Path to custom elixir bin; empty means use system elixir")
    args = ap.parse_args()

    out_file = args.out_file or Path(args.adapter_dir) / "eval_results.jsonl"

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
    print(f"=== Evaluating {len(test)} entries ===")

    pass_count = 0
    em_count = 0  # exact match against reference elixir_type

    with open(out_file, "w") as fout:
        for i, ex in enumerate(test):
            prompt = format_prompt(ex)
            inputs = tok(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    pad_token_id=tok.eos_token_id,
                )
            full = tok.decode(gen[0], skip_special_tokens=False)
            after_prompt = full[len(tok.decode(inputs.input_ids[0], skip_special_tokens=False)):]
            generated_type = parse_generated_type(after_prompt)

            # Exact match
            em = generated_type.strip() == (ex.get("elixir_type") or "").strip()
            if em:
                em_count += 1

            # Executable check
            type_block = ""
            if ex.get("type"):
                type_block = "\n".join(ex["type"]) if isinstance(ex["type"], list) else str(ex["type"])
            passed, output = run_type_checker(
                ex["definition"], generated_type, ex["module"], type_block, args.elixir_bin
            )
            if passed:
                pass_count += 1

            fout.write(json.dumps({
                "module": ex["module"],
                "function": ex["function"],
                "arity": ex["arity"],
                "project": ex["project"],
                "reference_elixir_type": ex.get("elixir_type"),
                "generated_elixir_type": generated_type,
                "exact_match": em,
                "type_check_pass": passed,
                "type_check_output": output[:500],  # truncate for file size
            }) + "\n")
            fout.flush()

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(test)}] EM: {em_count}, TypeCheck pass: {pass_count}")

    print(f"\n=== Final ===")
    print(f"  Total:           {len(test)}")
    print(f"  Exact match:     {em_count} ({100*em_count/len(test):.1f}%)")
    print(f"  TypeCheck@1:     {pass_count} ({100*pass_count/len(test):.1f}%)")
    print(f"  Results saved:   {out_file}")


if __name__ == "__main__":
    main()