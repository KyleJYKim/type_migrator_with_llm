"""
SFT (Supervised Fine-Tuning) for the TYPESPEC track: the decoder-only model
(Qwen2.5-Coder) learns to write the original Erlang-style `@spec`, which is
translated into an Elixir Type afterwards and only then scored.

Counterpart of train_sft.py, which trains the same model to emit the
set-theoretic annotation directly. Kept as its own script (rather than a flag on
that one) so the Descr-track runs stay reproducible byte for byte while this
track is developed.

Everything below is the Descr track's regime -- QLoRA over the same modules, the
same packing, the same optimiser -- with two changes:
  * the prompt comes from prompt_typespec.py (asks for a @spec);
  * the completion is example["spec"], not example["elixir_type"].

Usage:
    python scripts/train_sft_typespec.py \\
        --config configs/qwen7b_qlora_typespec.yaml \\
        --data_dir data/seed42/track2_both_pass_expanded \\
        --output_dir runs/qwen7b_typespec/seed42
"""
import argparse
import json
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

import prompt_typespec
from prompt_typespec import TARGET_FIELD, build_prompt
from prompt_logger import log_training_prompts, write_manifest


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def format_prompt(example):
    """Training text = the shared prompt + the target @spec + EOS."""
    return {"text": build_prompt(example) + example[TARGET_FIELD] + "<|endoftext|>"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", default="data/seed42/track2_both_pass_expanded")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = args.output_dir or cfg.get("output_dir", "runs/qwen7b_typespec")

    print(f"=== Config ===\n{json.dumps(cfg, indent=2)}")
    print(f"=== Target field: {TARGET_FIELD} (TypeSpec track) ===")
    print(f"=== Output: {output_dir} ===")

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

    lora = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    data_dir = Path(args.data_dir)
    raw = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "val.jsonl"),
        },
    )
    ds = raw.map(format_prompt, remove_columns=raw["train"].column_names)

    # Drop examples whose tokenized text exceeds max_seq_length. Under packing an
    # over-length example is shredded across blocks (or truncated, losing its
    # trailing EOS), teaching the model to emit non-terminating fragments.
    # Specs are far shorter than Descr annotations, so this should drop close to
    # nothing here -- the count is printed to confirm that per run.
    max_seq = cfg["training"]["max_seq_length"]
    before = {k: len(ds[k]) for k in ds}
    ds = ds.filter(lambda ex: len(tokenizer(ex["text"]).input_ids) <= max_seq)
    dropped = {k: before[k] - len(ds[k]) for k in ds}
    print(f"Dropped over-length (> {max_seq} tok) examples: {dropped}")

    print(f"Train: {len(ds['train'])}, Val: {len(ds['validation'])}")
    print(f"Sample text[0]:\n{ds['train'][0]['text'][:600]}")

    # Preserve what this run actually trained on: the format in effect, and a
    # sample of the exact texts. Written before training so they survive a run
    # that later fails or is cut short by the scheduler.
    write_manifest(
        output_dir, prompt_typespec,
        phase="train:sft",
        extra={
            "data_dir": str(data_dir),
            "config": args.config,
            "model": cfg["model_name_or_path"],
            "train_examples": len(ds["train"]),
            "val_examples": len(ds["validation"]),
            "dropped_over_length": dropped,
        },
        example=ds["train"][0]["text"],
    )
    log_training_prompts(output_dir, ds["train"]["text"], seed=cfg["training"]["seed"])

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
        load_best_model_at_end=cfg["training"].get("load_best_model_at_end", False),
        metric_for_best_model=cfg["training"].get("metric_for_best_model", "eval_loss"),
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

    trainer.train()
    # With load_best_model_at_end=True the model in memory here is the best
    # checkpoint, not the final weights; a None below means that reload did NOT
    # happen and save_model would persist the final weights instead.
    print(f"=== best_model_checkpoint={trainer.state.best_model_checkpoint} "
          f"best_metric(eval_loss)={trainer.state.best_metric} "
          f"global_step={trainer.state.global_step} ===")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"=== Done. Adapter saved to {output_dir} ===")


if __name__ == "__main__":
    main()
