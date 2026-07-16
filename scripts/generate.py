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
    StoppingCriteriaList,
)
try:
    from transformers import MaxTimeCriteria
except ImportError:  # older/newer layouts keep it in the submodule
    from transformers.generation.stopping_criteria import MaxTimeCriteria


def format_prompt(example):
    type_block = ""
    if example.get("type"):
        if isinstance(example["type"], list):
            type_block = "\n".join(example["type"])
        else:
            type_block = str(example["type"])

    # Must match train_sft.py's format_prompt exactly -- a train/inference
    # prompt mismatch would silently degrade generation quality.
    return_expr_block = ""
    if example.get("return_expressions"):
        return_expr_block = "\n".join(example["return_expressions"])

    return (
        f"### Module: {example['module']}\n"
        f"### Types in scope:\n{type_block}\n\n"
        f"### Definition:\n{example['definition']}\n\n"
        f"### Return expressions:\n{return_expr_block}\n\n"
        f"### Elixir type:\n"
    )


def parse_generated_type(generated_text):
    for marker in ["<|endoftext|>", "<|im_end|>", "\n###", "\n\n"]:
        idx = generated_text.find(marker)
        if idx > 0:
            generated_text = generated_text[:idx]
    return generated_text.strip()


def build_grammar_processor(grammar_path, hf_tokenizer):
    """Return a LogitsProcessor that constrains generation to the descr type grammar
    (GBNF, scripts/descr_type.gbnf), so the model can only emit well-formed, descr-only
    types --- no missing delimiters, no unterminated atoms, no drift to TypeSpec forms.
    Uses transformers-cfg (pip install transformers-cfg).
    """
    from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
    from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
    with open(grammar_path) as f:
        grammar_str = f.read()
    constraint = IncrementalGrammarConstraint(grammar_str, "root", hf_tokenizer)
    return GrammarConstrainedLogitsProcessor(constraint)


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
    # (scripts/descr_type.gbnf). Eliminates malformed/unparseable output and TypeSpec
    # drift by construction, so the repetition_penalty above is bypassed when this is on.
    ap.add_argument("--constrain", action="store_true",
                    help="constrain decoding to the descr type grammar (needs transformers-cfg)")
    ap.add_argument("--grammar",
                    default=str(Path(__file__).resolve().parent / "descr_type.gbnf"))
    # Per-entry wall-clock cap. Grammar-constrained decoding recomputes a vocab-wide
    # mask every token, so a single entry whose type runs long can grind for many
    # minutes (esp. on a MIG slice) and stall the whole run. This bounds each entry;
    # a capped entry yields a (likely truncated) prediction and the run continues.
    ap.add_argument("--max_time", type=float, default=60.0,
                    help="seconds per entry before generation is cut (constrained path)")
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
            # Damp degenerate enumeration loops (the model emitting a union/struct far
            # longer than any real target, e.g. ":a or :b or :c ..." or "integer(),
            # integer(), ..."). Applied on BOTH paths: the grammar keeps every token
            # valid but a huge union/struct is perfectly valid, so the grammar has no
            # notion of "long enough" and cannot stop the loop on its own. Mild values
            # leave legitimately-repetitive struct types intact.
            gen_kwargs["repetition_penalty"] = args.repetition_penalty
            gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
            if grammar_processor is not None:
                # Reset the incremental parser state between examples.
                grammar_processor.reset()
                gen_kwargs["logits_processor"] = LogitsProcessorList([grammar_processor])
                # Fresh timer per entry so no single (slow-mask) generation stalls the run.
                gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                    [MaxTimeCriteria(max_time=args.max_time)]
                )

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
