"""
Checkpoint sweep for the decoder-only model: for every saved checkpoint of one
run, compute cheap teacher-forced signals AND generate on a fixed validation
subset, so the checkpoint can be chosen by the task metric (compatible?/2)
instead of by eval_loss.

    python scripts/sweep_checkpoints.py --run_dir runs/sweep/qwen_completion_instr \
        --data_dir data/seed42/track2_both_pass_expanded --n 400 --constrain

Writes, per checkpoint, into <run_dir>/sweep/<checkpoint-N>/:
  signals.json          teacher-forced: full-sequence loss, type-only loss,
                        type-token accuracy, whole-type exact under teacher forcing
  val_predictions.jsonl generated types for the subset (same format as
                        generate.py, so `mix eval_predictions set-theoretic`
                        scores it unchanged)
Re-running skips checkpoints whose outputs are complete.

The validation subset is stratified by arity bucket and definition-length
tertile and written ONCE to <data_dir>/val_sweep_n<N>_s<seed>.jsonl, so every
run of the study is scored on the same functions. The test set is never read.

The prompt (instruction on/off) comes from prompt.py / ETR_INCLUDE_INSTRUCTION,
which must match the run being swept: the script refuses to run if the run's
prompt_manifest.json disagrees.
"""
import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LogitsProcessorList,
    StoppingCriteriaList,
)
try:
    from transformers import MaxTimeCriteria
except ImportError:
    from transformers.generation.stopping_criteria import MaxTimeCriteria

import prompt as prompt_module
from prompt import TARGET_FIELD, build_prompt
from generate import build_grammar_processor, parse_generated_type

EOS = "<|endoftext|>"


# ---------------------------------------------------------------- subset
def stratified_subset(rows, n, seed):
    """Sample n rows, proportionally over (arity bucket, length tertile)."""
    lens = sorted(len(r["definition"]) for r in rows)
    cut1, cut2 = lens[len(lens) // 3], lens[2 * len(lens) // 3]
    def key(r):
        a = min(int(r.get("arity", 0)), 3)
        L = len(r["definition"])
        return (a, 0 if L < cut1 else 1 if L < cut2 else 2)
    strata = defaultdict(list)
    for r in rows:
        strata[key(r)].append(r)
    rnd = random.Random(seed)
    picked = []
    for k in sorted(strata):
        group = strata[k]
        rnd.shuffle(group)
        picked += group[: round(n * len(group) / len(rows))]
    rnd.shuffle(picked)
    return picked[:n]


def load_or_make_subset(data_dir, n, seed):
    path = Path(data_dir) / f"val_sweep_n{n}_s{seed}.jsonl"
    if path.exists():
        rows = [json.loads(l) for l in open(path) if l.strip()]
        print(f"=== Reusing validation subset {path} ({len(rows)}) ===")
        return rows
    val = [json.loads(l) for l in open(Path(data_dir) / "val.jsonl") if l.strip()]
    rows = stratified_subset(val, n, seed)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"=== Wrote validation subset {path} ({len(rows)} of {len(val)}) ===")
    return rows


# ---------------------------------------------------------------- signals
@torch.no_grad()
def teacher_forced_signals(model, tok, rows, max_len):
    tot_all = n_all = tot_typ = n_typ = hit = whole = counted = 0
    for r in rows:
        p = tok(build_prompt(r), add_special_tokens=False).input_ids
        c = tok(r[TARGET_FIELD] + EOS, add_special_tokens=False).input_ids
        ids = p + c
        if len(ids) > max_len:
            continue
        x = torch.tensor([ids], device=model.device)
        logits = model(input_ids=x).logits[0, :-1].float()
        tgt = x[0, 1:]
        nll = torch.nn.functional.cross_entropy(logits, tgt, reduction="none")
        is_type = torch.zeros_like(tgt, dtype=torch.bool)
        is_type[len(p) - 1:] = True               # positions predicting completion tokens
        pred = logits.argmax(-1)
        tot_all += nll.sum().item(); n_all += nll.numel()
        tot_typ += nll[is_type].sum().item(); n_typ += int(is_type.sum())
        ok = (pred[is_type] == tgt[is_type])
        hit += int(ok.sum()); whole += int(ok.all()); counted += 1
    return {
        "full_seq_loss": tot_all / max(n_all, 1),
        "type_loss": tot_typ / max(n_typ, 1),
        "type_token_acc": hit / max(n_typ, 1),
        "type_exact_teacher_forced": whole / max(counted, 1),
        "n_scored": counted,
    }


# ---------------------------------------------------------------- generation
@torch.no_grad()
def generate_subset(model, tok, rows, out_path, args, grammar):
    with open(out_path, "w") as fout:
        for i, ex in enumerate(rows):
            inputs = tok(build_prompt(ex), return_tensors="pt", add_special_tokens=False).to(model.device)
            kw = dict(max_new_tokens=args.max_new_tokens, do_sample=False,
                      pad_token_id=tok.eos_token_id,
                      repetition_penalty=args.repetition_penalty)
            if grammar is not None:
                grammar.reset()
                kw["logits_processor"] = LogitsProcessorList([grammar])
                kw["stopping_criteria"] = StoppingCriteriaList([MaxTimeCriteria(max_time=args.max_time)])
            gen = model.generate(**inputs, **kw)
            raw = tok.decode(gen[0, inputs.input_ids.shape[-1]:], skip_special_tokens=False)
            fout.write(json.dumps({**ex, "generated_elixir_type": parse_generated_type(raw)}) + "\n")
            if (i + 1) % 50 == 0:
                print(f"    [{i+1}/{len(rows)}] generated")


def step_of(p):
    m = re.search(r"checkpoint-(\d+)$", p.name)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-Coder-7B")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--checkpoints", nargs="*", help="only these checkpoint dir names")
    ap.add_argument("--only", choices=["both", "signals", "generate"], default="both")
    ap.add_argument("--constrain", action="store_true")
    ap.add_argument("--grammar", default=str(Path(__file__).resolve().parent / "descr_type.gbnf"))
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    ap.add_argument("--max_time", type=float, default=60.0)
    ap.add_argument("--prepare_only", action="store_true",
                    help="only write the shared validation subset, then exit "
                         "(run once before an array job, so tasks never race on it)")
    ap.add_argument("--shard", default=None,
                    help="i/k: process only checkpoints whose index %% k == i (SLURM arrays)")
    args = ap.parse_args()

    if args.prepare_only:
        load_or_make_subset(args.data_dir, args.n, args.seed)
        return

    run = Path(args.run_dir)
    man = run / "prompt_manifest.json"
    if man.exists():
        want = json.load(open(man)).get("toggles", {}).get("INCLUDE_INSTRUCTION", True)
        if want != prompt_module.INCLUDE_INSTRUCTION:
            sys.exit(f"Prompt mismatch: run trained with INCLUDE_INSTRUCTION={want}, "
                     f"now {prompt_module.INCLUDE_INSTRUCTION}. Set ETR_INCLUDE_INSTRUCTION accordingly.")

    ckpts = sorted((p for p in run.glob("checkpoint-*") if p.is_dir()), key=step_of)
    if (run / "final").is_dir():
        ckpts.append(run / "final")
    if args.checkpoints:
        ckpts = [c for c in ckpts if c.name in set(args.checkpoints)]
    if args.shard:
        i, k = map(int, args.shard.split("/"))
        ckpts = [c for j, c in enumerate(ckpts) if j % k == i]
    print(f"=== {len(ckpts)} checkpoints: {[c.name for c in ckpts]} ===")

    rows = load_or_make_subset(args.data_dir, args.n, args.seed)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, device_map="auto", torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16),
    )
    grammar = build_grammar_processor(args.grammar, tok) if args.constrain else None

    model, prev = None, None
    for ck in ckpts:
        out = run / "sweep" / ck.name
        out.mkdir(parents=True, exist_ok=True)
        sig_f, pred_f = out / "signals.json", out / "val_predictions.jsonl"
        need_sig = args.only in ("both", "signals") and not sig_f.exists()
        need_gen = args.only in ("both", "generate") and not (
            pred_f.exists() and sum(1 for _ in open(pred_f)) == len(rows))
        if not (need_sig or need_gen):
            print(f"--- {ck.name}: done already, skipping")
            continue
        print(f"--- {ck.name}")
        name = ck.name.replace("-", "_")
        if model is None:
            model = PeftModel.from_pretrained(base, str(ck), adapter_name=name)
        else:
            model.load_adapter(str(ck), adapter_name=name)
        model.set_adapter(name)
        if prev:
            model.delete_adapter(prev)
        prev = name
        model.eval()
        if need_sig:
            s = teacher_forced_signals(model, tok, rows, args.max_len)
            s["checkpoint"] = ck.name
            s["step"] = step_of(ck)
            json.dump(s, open(sig_f, "w"), indent=1)
            print(f"    signals: {s}")
        if need_gen:
            generate_subset(model, tok, rows, pred_f, args, grammar)
    print("=== Sweep done. Score locally:  scripts/score_sweep.sh", run, "===")


if __name__ == "__main__":
    main()
