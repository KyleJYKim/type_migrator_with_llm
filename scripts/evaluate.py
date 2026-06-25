"""
Evaluate the fine-tuned model on test.jsonl using:
  Tier A — exact match
  Tier C — executable typecheck via mix compile (auxiliary signal)
(Distance is scored separately by the set-theoretic evaluator.)
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


def build_mix_project(tmp_root, example, generated_type):
    project_dir = Path(tmp_root)
    (project_dir / "lib").mkdir(parents=True, exist_ok=True)

    module_name = example["module"]
    safe_proj = re.sub(r"[^\w]", "_", module_name).lower()

    type_block = ""
    if example.get("type"):
        type_block = "\n  ".join(example["type"]) if isinstance(example["type"], list) else str(example["type"])

    lib_content = f"""defmodule {module_name} do
  {type_block}

  @assert_type_form {generated_type}
  {example["definition"]}
end
"""
    mix_content = f"""defmodule {safe_proj.title().replace('_', '')}.MixProject do
  use Mix.Project

  def project do
    [
      app: :{safe_proj},
      version: "0.1.0",
      elixir: "~> 1.19.5",
      deps: []
    ]
  end

  def application, do: [extra_applications: [:logger]]
end
"""
    (project_dir / "lib" / "eval.ex").write_text(lib_content)
    (project_dir / "mix.exs").write_text(mix_content)


def run_type_checker(tmp_root, elixir_bin):
    """Returns one of: True (pass), False (typecheck warning), 'compile_error', 'timeout', 'error'."""
    abs_elixir_bin = str(Path(elixir_bin).resolve())
    env = os.environ.copy()
    env["PATH"] = f"{abs_elixir_bin}:{env.get('PATH', '')}"
    env["MIX_HOME"] = str(Path(tmp_root) / ".mix")
    env["MIX_INSTALL_DIR"] = str(Path(tmp_root) / ".mix-install")

    try:
        result = subprocess.run(
            ["bash", "-c", f"PATH={abs_elixir_bin}:$PATH mix compile --force"],
            cwd=tmp_root,
            capture_output=True, text=True, timeout=60, env=env,
        )
        output = result.stdout + result.stderr

        if "type warning found at" in output:
            return False, output
        if "** (CompileError)" in output or "** (SyntaxError)" in output:
            return "compile_error", output
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return "timeout", "TIMEOUT"
    except Exception as e:
        return "error", f"ERROR: {e}"


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
    # Only needed for the in-process Tier C check; on the remote (generate +
    # exact-match + semantic-distance) we pass --skip_typecheck and the
    # executable typecheck is run separately on the local side.
    ap.add_argument("--elixir_bin", default=None)
    ap.add_argument("--skip_typecheck", action="store_true",
                    help="Skip the executable mix compile check (much faster)")
    args = ap.parse_args()

    if not args.skip_typecheck and not args.elixir_bin:
        ap.error("--elixir_bin is required unless --skip_typecheck is given")

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
    n = len(test)
    print(f"=== Evaluating {n} entries ===")

    results = []
    em_count = 0
    pass_count = 0
    compile_error_count = 0
    timeout_count = 0
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
            after_prompt = full[prompt_len:]
            generated_type = parse_generated_type(after_prompt)

            reference = ex.get("elixir_type") or ""
            # Reproducibility bucket: a reference longer than the generation budget
            # cannot be emitted in full by any model, so it is bucketed (not dropped)
            # and reported separately rather than silently depressing the pass rate.
            ref_token_len = len(tok(reference).input_ids)
            reference_over_budget = ref_token_len > args.max_new_tokens
            if reference_over_budget:
                over_budget_count += 1

            # Tier A — exact match
            em = generated_type.strip() == reference.strip()
            if em:
                em_count += 1

            # Tier C — executable typecheck (optional)
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
                # Carry the FULL source entry (file, definition, type, spec, line
                # locators, elixir_type, ...) so this jsonl doubles as the
                # predictions dataset for the local typecheck step: injecting
                # generated_elixir_type back into the real project and recompiling.
                # The reference type is the entry's own elixir_type (carried by **ex).
                **ex,
                "generated_elixir_type":   generated_type,
                "reference_token_len":     ref_token_len,
                "reference_over_budget":   reference_over_budget,
                "exact_match":             em,
                "type_check_pass":         passed,
                "type_check_output":       tc_output[:500] if tc_output else "",
            }
            results.append(entry_result)
            fout.write(json.dumps(entry_result) + "\n")
            fout.flush()

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{n}] EM: {em_count} ({100*em_count/(i+1):.1f}%)")

    # ----------------------------- Summary -----------------------------
    verifiable = n - compile_error_count - timeout_count

    print(f"\n=== Final ===")
    print(f"  Total:                   {n}")
    print(f"  Exact match:             {em_count} ({100*em_count/n:.1f}%)")

    if not args.skip_typecheck:
        print()
        print(f"  Compile errors:          {compile_error_count} (excluded)")
        print(f"  Timeouts/errors:         {timeout_count} (excluded)")
        print(f"  TypeCheck@1:             {pass_count}/{verifiable} "
              f"({100*pass_count/max(verifiable,1):.1f}%)")

    # Reproducible subset: exclude references that exceed the generation budget
    # (unreproducible by construction) and report the rates with them bucketed out.
    repro = [r for r in results if not r["reference_over_budget"]]
    nb = len(repro)
    print()
    print(f"  Unreproducible (ref > {args.max_new_tokens} tok): {over_budget_count} (bucketed)")
    if nb and nb != n:
        em_b = sum(1 for r in repro if r["exact_match"])
        print(f"  -- excluding unreproducible ({nb} entries) --")
        print(f"     Exact match:          {em_b} ({100*em_b/nb:.1f}%)")
        if not args.skip_typecheck:
            verif_b = sum(1 for r in repro if r["type_check_pass"] in (True, False))
            pass_b = sum(1 for r in repro if r["type_check_pass"] is True)
            print(f"     TypeCheck@1:          {pass_b}/{verif_b} "
                  f"({100*pass_b/max(verif_b,1):.1f}%)")

    print(f"\n  Saved to: {out_file}")


if __name__ == "__main__":
    main()