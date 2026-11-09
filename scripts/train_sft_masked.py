"""
QLoRA fine-tuning of the decoder-only model (Qwen) with an explicit choice of
which tokens the loss is computed on -- the checkpoint-selection study.

    python scripts/train_sft_masked.py --config configs/qwen7b_qlora_sweep.yaml \
        --data_dir data/seed42/track2_both_pass_expanded \
        --output_dir runs/sweep/qwen_completion_instr --loss completion

--loss full        every token is scored: prompt + type + EOS. This is what
                   train_sft.py does (SFTTrainer on a plain "text" field).
--loss completion  only the type + EOS are scored; prompt tokens get label -100,
                   exactly as the CodeT5+ run scores only its decoder target.

Differences from train_sft.py, on purpose:
  * Tokenisation is done here, prompt and completion separately and then
    concatenated, so the prompt/completion boundary is exact and identical to
    generation (generate.py tokenises the prompt alone). This does not depend on
    the installed TRL version's completion-only / packing behaviour.
  * No packing. Packing mixes several examples in one block, which makes a
    per-example mask fragile across TRL versions. Both --loss arms use the same
    unpacked pipeline, so the only difference between them is the mask.
    (Unpacked steps are smaller, so the step count is ~4x train_sft.py's.)
  * Checkpoints are kept, not pruned to a "best by eval_loss": selection is
    done afterwards by scripts/sweep_checkpoints.py on the task metric.
"""
import argparse
import inspect
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    DataCollatorForSeq2Seq, Trainer, TrainingArguments,
)

import prompt as prompt_module
from prompt import TARGET_FIELD, build_prompt
from prompt_logger import log_training_prompts, write_manifest

EOS = "<|endoftext|>"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def encode(example, tok, loss, max_len):
    """Token ids and labels for one example, or None if it is over max_len."""
    p_ids = tok(build_prompt(example), add_special_tokens=False).input_ids
    c_ids = tok(example[TARGET_FIELD] + EOS, add_special_tokens=False).input_ids
    ids = p_ids + c_ids
    if len(ids) > max_len:
        return None
    labels = list(ids) if loss == "full" else [-100] * len(p_ids) + list(c_ids)
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def build_split(rows, tok, loss, max_len):
    out, dropped = [], 0
    for r in rows:
        e = encode(r, tok, loss, max_len)
        if e is None:
            dropped += 1
        else:
            out.append(e)
    return Dataset.from_list(out), dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--loss", choices=["full", "completion"], required=True)
    ap.add_argument("--max_train_examples", type=int, default=0,
                    help="smoke test: train on the first N examples only")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    t = cfg["training"]
    out = Path(args.output_dir)
    print(f"=== loss={args.loss}  instruction={prompt_module.INCLUDE_INSTRUCTION}  out={out} ===")

    tok = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    q = cfg["quantization"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"], bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"], quantization_config=bnb,
        device_map="auto", torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    l = cfg["lora"]
    model = get_peft_model(model, LoraConfig(
        r=l["r"], lora_alpha=l["alpha"], lora_dropout=l["dropout"],
        target_modules=l["target_modules"], bias="none", task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    data_dir = Path(args.data_dir)
    max_len = t["max_seq_length"]
    train_rows = load_jsonl(data_dir / "train.jsonl")
    val_rows = load_jsonl(data_dir / "val.jsonl")
    if args.max_train_examples:
        train_rows = train_rows[: args.max_train_examples]
    # The in-training eval is only a log; a fixed random subset keeps it cheap.
    n_eval = t.get("eval_subset", 0)
    if n_eval and n_eval < len(val_rows):
        val_rows = random.Random(t["seed"]).sample(val_rows, n_eval)
    train_ds, d_tr = build_split(train_rows, tok, args.loss, max_len)
    val_ds, d_va = build_split(val_rows, tok, args.loss, max_len)
    print(f"Train {len(train_ds)} (dropped {d_tr} over {max_len} tok) | Eval {len(val_ds)} (dropped {d_va})")

    write_manifest(out, prompt_module, phase=f"train:sft-masked:{args.loss}", extra={
        "data_dir": str(data_dir), "config": args.config, "model": cfg["model_name_or_path"],
        "loss": args.loss, "train_examples": len(train_ds), "eval_examples": len(val_ds),
        "dropped_over_length": {"train": d_tr, "eval": d_va}, "packing": False,
    }, example=build_prompt(train_rows[0]) + train_rows[0][TARGET_FIELD] + EOS)
    log_training_prompts(out, [build_prompt(r) + r[TARGET_FIELD] + EOS for r in train_rows[:200]],
                         seed=t["seed"])

    ta = dict(
        output_dir=str(out),
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        num_train_epochs=t["num_train_epochs"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        bf16=t["bf16"], optim=t["optim"], seed=t["seed"],
        eval_steps=t["eval_steps"], save_steps=t["save_steps"],
        save_strategy="steps", save_total_limit=t.get("save_total_limit"),
        save_only_model=t.get("save_only_model", True),
        logging_steps=t["logging_steps"], report_to=t["report_to"],
        group_by_length=True, remove_unused_columns=False,
        load_best_model_at_end=False,
    )
    # transformers renamed evaluation_strategy -> eval_strategy (4.41+).
    params = inspect.signature(TrainingArguments.__init__).parameters
    ta["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "steps"
    args_tr = TrainingArguments(**ta)

    collator = DataCollatorForSeq2Seq(tok, label_pad_token_id=-100, pad_to_multiple_of=8)
    kw = dict(model=model, args=args_tr, train_dataset=train_ds, eval_dataset=val_ds,
              data_collator=collator)
    # transformers renamed Trainer(tokenizer=) -> processing_class (4.46+).
    tparams = inspect.signature(Trainer.__init__).parameters
    kw["processing_class" if "processing_class" in tparams else "tokenizer"] = tok
    trainer = Trainer(**kw)
    trainer.train()

    trainer.save_model(str(out / "final"))
    tok.save_pretrained(str(out / "final"))
    with open(out / "train_log_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=1)
    print(f"=== Done. Checkpoints in {out}/checkpoint-*, final adapter in {out}/final ===")


if __name__ == "__main__":
    main()
