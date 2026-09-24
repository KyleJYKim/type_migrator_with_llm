"""
Seq2seq (encoder-decoder) FULL fine-tuning for the TYPESPEC track: CodeT5+
learns to write the original Erlang-style `@spec`, translated into an Elixir
Type afterwards and only then scored.

Counterpart of train_seq2seq.py (same model, Descr target) and the
encoder-decoder half of this track, opposite train_sft_typespec.py.

Mirrors the Descr track's regime exactly -- full fine-tuning, no adapters or
quantisation, DataCollatorForSeq2Seq with -100 label padding -- changing only:
  * the prompt, which comes from prompt_typespec.py;
  * the decoder target, example["spec"] instead of example["elixir_type"].

Usage:
    python scripts/train_seq2seq_typespec.py \\
        --config configs/codet5p_770m_typespec.yaml \\
        --data_dir data/seed42/typespec_both_pass \\
        --output_dir runs/codet5p_770m_typespec/seed42
"""
import argparse
import json
from pathlib import Path

import torch
if not torch.cuda.is_available():
    raise RuntimeError(
        "No GPU available. Are you on the frontend? "
        "Use `salloc + srun --pty bash` to get a compute node first."
    )
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

import prompt_typespec
from prompt_typespec import TARGET_FIELD, build_prompt
from prompt_logger import LazyTexts, log_training_prompts, write_manifest


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", default="data/seed42/typespec_both_pass")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = args.output_dir or cfg.get("output_dir", "runs/codet5p_typespec")
    trust = bool(cfg.get("trust_remote_code", False))
    max_src = cfg["seq2seq"]["max_source_length"]
    max_tgt = cfg["seq2seq"]["max_target_length"]
    tr = cfg["training"]

    print(f"=== Config ===\n{json.dumps(cfg, indent=2)}")
    print(f"=== Target field: {TARGET_FIELD} (TypeSpec track) ===")
    print(f"=== Output: {output_dir} ===")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name_or_path"], trust_remote_code=trust
    )
    # Full fine-tuning: load in fp32 so AdamW keeps fp32 master/states;
    # bf16=True below does mixed-precision autocast for the forward/backward.
    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg["model_name_or_path"], trust_remote_code=trust
    )

    # codet5p-2b (custom modeling code) ships without decoder_start_token_id /
    # pad_token_id, which DataCollatorForSeq2Seq's shift_tokens_right and
    # generate() both need. Derive them so training and inference agree.
    def _first_set(*vals):
        for v in vals:
            if v is not None:
                return v
        return None

    c = model.config
    if getattr(c, "decoder_start_token_id", None) is None:
        c.decoder_start_token_id = _first_set(
            getattr(c, "bos_token_id", None),
            tokenizer.bos_token_id, tokenizer.pad_token_id, tokenizer.eos_token_id,
        )
    if getattr(c, "pad_token_id", None) is None:
        c.pad_token_id = _first_set(tokenizer.pad_token_id, tokenizer.eos_token_id)
    print(f"decoder_start_token_id={c.decoder_start_token_id}, pad_token_id={c.pad_token_id}")

    # transformers >=4.45 calls config._get_non_default_generation_parameters()
    # during save_pretrained, which codet5p's custom config asserts on.
    try:
        type(model.config)._get_non_default_generation_parameters = lambda self: {}
    except Exception as e:
        print(f"(note: could not patch generation-param check: {e})")

    model.config.use_cache = False

    data_dir = Path(args.data_dir)
    raw = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "val.jsonl"),
        },
    )

    eos = tokenizer.eos_token_id

    def preprocess(ex):
        model_inputs = tokenizer(
            build_prompt(ex), max_length=max_src, truncation=True
        )
        lab = tokenizer(
            text_target=ex[TARGET_FIELD], max_length=max_tgt, truncation=True
        )["input_ids"]
        # Ensure the target ends with EOS so the model learns to STOP: codet5p's
        # GPT2-style tokenizer does not auto-append it (T5 would), which
        # otherwise causes runaway repetition until max_new_tokens at inference.
        if eos is not None and (not lab or lab[-1] != eos):
            if len(lab) >= max_tgt:
                lab = lab[: max_tgt - 1]
            lab = lab + [eos]
        model_inputs["labels"] = lab
        return model_inputs

    ds = raw.map(preprocess, remove_columns=raw["train"].column_names)
    print(f"Train: {len(ds['train'])}, Val: {len(ds['validation'])}")
    print(f"Sample input ids len: {len(ds['train'][0]['input_ids'])}, "
          f"label len: {len(ds['train'][0]['labels'])}")

    # Preserve what this run actually trained on. The encoder input and the
    # decoder target are separate streams here, so they are logged as separate
    # fields rather than concatenated -- the log then matches how the model is
    # actually fed. Written before training, so they survive a failed run.
    render = lambda ex: {"encoder_input": build_prompt(ex), "target": ex[TARGET_FIELD]}
    write_manifest(
        output_dir, prompt_typespec,
        phase="train:seq2seq",
        extra={
            "data_dir": str(data_dir),
            "config": args.config,
            "model": cfg["model_name_or_path"],
            "train_examples": len(ds["train"]),
            "val_examples": len(ds["validation"]),
            "max_source_length": max_src,
            "max_target_length": max_tgt,
        },
        example=render(raw["train"][0]),
    )
    log_training_prompts(
        output_dir, LazyTexts(raw["train"], render), seed=tr["seed"]
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    sargs = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=tr["per_device_train_batch_size"],
        per_device_eval_batch_size=tr.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=tr["gradient_accumulation_steps"],
        num_train_epochs=tr["num_train_epochs"],
        learning_rate=tr["learning_rate"],
        lr_scheduler_type=tr["lr_scheduler_type"],
        warmup_ratio=tr["warmup_ratio"],
        weight_decay=tr["weight_decay"],
        bf16=tr["bf16"],
        optim=tr.get("optim", "adamw_torch"),
        gradient_checkpointing=tr.get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=tr["seed"],
        eval_strategy="steps",
        eval_steps=tr["eval_steps"],
        save_strategy="steps",
        save_steps=tr["save_steps"],
        save_total_limit=tr["save_total_limit"],
        logging_steps=tr["logging_steps"],
        report_to=tr["report_to"],
        load_best_model_at_end=tr["load_best_model_at_end"],
        metric_for_best_model=tr["metric_for_best_model"],
        predict_with_generate=False,   # generation is done in generate_seq2seq_typespec.py
    )

    # transformers >=4.46 renamed Trainer's `tokenizer` arg to `processing_class`.
    import inspect
    tk_kwargs = (
        {"processing_class": tokenizer}
        if "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters
        else {"tokenizer": tokenizer}
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=sargs,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        **tk_kwargs,
    )

    trainer.train()
    print(f"=== best_model_checkpoint={trainer.state.best_model_checkpoint} "
          f"best_metric(eval_loss)={trainer.state.best_metric} "
          f"global_step={trainer.state.global_step} ===")
    trainer.save_model(output_dir)         # saves the FULL fine-tuned model
    tokenizer.save_pretrained(output_dir)
    print(f"=== Done. Model saved to {output_dir} ===")


if __name__ == "__main__":
    main()
