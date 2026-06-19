"""
SFT (Supervised Fine-Tuning) training script for elixir_type prediction.
Usage:
    python scripts/train_sft.py --config configs/qwen7b_qlora.yaml
    1. Load config from a YAML file.
    2. Load the pre-trained LLM (Qwen2.5-Coder-7B) in 4-bit to save GPU memory.
    3. Attach LoRA adapters so only a tiny fraction of weights are trainable.
    4. Format training data into prompt/completion strings.
    5. Train for several epochs, evaluating on validation data periodically.
    6. Save the LoRA adapter -> later merged with the base model for inference.
libraries:
    transformers: Load any pre-trained LLM from HuggingFace
    peft: Apply LoRA (parameter-efficient fine-tuning)
    bitsandbytes: 4-bit quantization to reduce GPU memory
    trl: `SFTTrainer`, a high-level training loop built for fine-tuning
    datasets: Load and process training data
"""
import argparse
import json
import os
from pathlib import Path

import torch
if not torch.cuda.is_available():
    raise RuntimeError(
        "No GPU available. Are you running on the frontend? "
        "Use `salloc + srun --pty bash` to get onto a compute node first."
    )
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
    """
    All hyperparameters (learning rate, batch size, model name, LoRA settings) live in a separate YAML file.
    This keeps the training script generic and reusable - just swap the config file
    """
    with open(path) as f:
        return yaml.safe_load(f)


def format_prompt(example):
    """
    Build the training text. 
    Train on the full text.
    Each raw data example gets formatted into on big string:
        • Prompt = the context the model sees (module name, types, function definition)
        • Completion = the answer the model must learn to produce
        • `<|endoftext|>` = a special token telling the model "this is where the answer end"
    """
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

    # Model + tokenizer
    """
    A 7B parameter model normally needs ~14GB GPU RAM (in float16).
    4-bit quantization compresses weights so it fits in ~5-6GB,
    making fine-tuning possible on a single GPU.
    * `nf4` = "NormalFloat 4-bit", a special format that works well for LLMs.
    """
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

    # LoRA: Train only a tiny fraction of weights
    """
    Instead of updating all 7B weights (very expensive indeed), LoRA injects small trainable matrics into specific layers.
    Only ~0.1-1% of weights are trained, and yet results are nearly as good.
    Parameter:
        • `r`: Rank of LoRA matrices - higher = more capacity but more memory
        • `lora_alpha`: Scaling factor for LoRA updates
        • `target_modules`: Which layers to inject LoRA into (usually attention layers)
    """
    lora = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load Data
    """
    JSONL files (one JSON object per line) are loaded and each example is converted to a single `"text"` field using `format_prompt()`.
    """
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

    # Trainer: Training hyperparameters
    """
    Setting:
        • `per_device_train_batch_size: Examples per GPU per step (small = less memory)
        • `gradient_accumulation_steps`: Simulate a larger batch by accumulating gradients over N steps
        • `learning_rate`: How fast the model updates weights
        • `num_train_epochs`: How many times to loop over the full dataset
        • `bf16=-True`: Use bfloat16 precision for faster math
        • `packing`: Concatenate short examples to fill the full context window (efficiency trick)
        
    """
    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["training"].get("per_device_eval_batch_size", 1),
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

    """
    `trainer.train()` runs the full training loop.
    Afterwards, only the small LoRA adapter is saved (not the full model); 
    which might be just a few hundred MB instead of 14GB.
    """
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"=== Done. Adapter saved to {output_dir} ===")


if __name__ == "__main__":
    main()