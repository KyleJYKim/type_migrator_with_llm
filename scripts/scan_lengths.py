#!/usr/bin/env python3
"""
Report the tokenized prompt+completion length distribution for the SFT data,
using the SAME format_prompt and the real Qwen tokenizer, so you can see how many
entries exceed max_seq_length (and would be shredded/truncated under packing)
before deciding the filter threshold.

Usage:
    python scripts/scan_lengths.py --config configs/qwen7b_qlora.yaml \
        --data_dir data --files train.jsonl val.jsonl test.jsonl
"""
import argparse, json
from pathlib import Path

import yaml
from transformers import AutoTokenizer


def format_text(e):
    # Mirror train_sft.format_prompt exactly.
    tb = e.get("type") or ""
    if isinstance(tb, list):
        tb = "\n".join(tb)
    prompt = (
        f"### Module: {e.get('module')}\n"
        f"### Types in scope:\n{tb}\n\n"
        f"### Definition:\n{e.get('definition')}\n\n"
        f"### Elixir type:\n"
    )
    return prompt + (e.get("elixir_type") or "") + "<|endoftext|>"


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--files", nargs="+", default=["train.jsonl", "val.jsonl", "test.jsonl"])
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    max_seq = cfg["training"]["max_seq_length"]
    tok = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    print(f"model={cfg['model_name_or_path']}  max_seq_length={max_seq}\n")

    for fname in args.files:
        path = Path(args.data_dir) / fname
        if not path.exists():
            print(f"-- {fname}: not found, skipping")
            continue
        rows = [json.loads(l) for l in open(path) if l.strip()]
        lens = [len(tok(format_text(e)).input_ids) for e in rows]
        n = len(lens)
        over = [(l, e) for l, e in zip(lens, rows) if l > max_seq]
        print(f"== {fname}: {n} entries ==")
        print(f"   tokens  p50={pct(lens,50)}  p90={pct(lens,90)}  p99={pct(lens,99)}  max={max(lens)}")
        print(f"   > {max_seq}: {len(over)} ({100*len(over)/n:.2f}%)")
        for l, e in sorted(over, key=lambda x: -x[0])[:8]:
            print(f"      {l:>7} tok  {e.get('module')}.{e.get('function')}/{e.get('arity')}")
        print()


if __name__ == "__main__":
    main()
