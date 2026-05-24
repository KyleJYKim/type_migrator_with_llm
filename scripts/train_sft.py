"""
SFT (Supervised Fine-Tuning) training script for elixir_type prediction.
Usage:
    python scripts/train_sft.py --config configs/qwen7b_qlora.yaml
"""
import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def format_prompt(example):
    """Build the training text. We train on the full text and rely on
    `completion_only_loss` in SFTConfig to ignore the prompt tokens."""
    type_block = ""
    if example.get("type"):
        if isinstance(example["type"], list):
            type_block = "\n".join(example["type"])
        else:
            type_block = str(example["type"])

    prompt = (
        f"### Module: {example['module']}\n"
        f"### Types in scope:\n{type_block}\n\n"
        f"### Definition:\n{example['definition']}\n\n"
        f"### Elixir type:\n"
    )
    completion = example["elixir_type"]
    return {"text": prompt + completion + "<|endoftext|>"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = args.output_dir or cfg.get("output_dir", "runs/default")

    print(f"=== Config ===\n{json.dumps(cfg, indent=2)}")
    print(f"=== Output: {output_dir} ===")

    # --- Model + tokenizer ---
    bnb = BitsAndBytesConfig(
        load_in_4bit=cfg["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=cfg["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=cfg["quantization"]["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, cfg["quantization"]["bnb_4bit_compute_dtype"]),
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # --- LoRA ---
    lora = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # --- Data ---
    data_dir = Path(args.data_dir)
    raw = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "val.jsonl"),
        },
    )
    ds = raw.map(format_prompt, remove_columns=raw["train"].column_names)
    print(f"Train: {len(ds['train'])}, Val: {len(ds['validation'])}")
    print(f"Sample text[0]:\n{ds['train'][0]['text'][:600]}")

    # --- Trainer ---
    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        learning_rate=cfg["training"]["learning_rate"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        weight_decay=cfg["training"]["weight_decay"],
        bf16=cfg["training"]["bf16"],
        optim=cfg["training"]["optim"],
        packing=cfg["training"]["packing"],
        max_seq_length=cfg["training"]["max_seq_length"],
        seed=cfg["training"]["seed"],
        eval_strategy="steps",
        eval_steps=cfg["training"]["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["training"]["save_steps"],
        save_total_limit=cfg["training"]["save_total_limit"],
        logging_steps=cfg["training"]["logging_steps"],
        report_to=cfg["training"]["report_to"],
        completion_only_loss=False,  # we keep the loss over the full text for now
        dataset_text_field="text",
        dataset_num_proc=4,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        peft_config=lora,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"=== Done. Adapter saved to {output_dir} ===")


if __name__ == "__main__":
    main()