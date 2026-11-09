# elixir_types_trainer

Fine-tunes and evaluates LLMs to predict **Elixir Types** (and TypeSpec)
annotations for unspec'd Elixir functions, using the dataset produced by
[`type_migrator`](../type_migrator). This is the LLM type-prediction study
of the accompanying Master's thesis (see `../type_migrator/thesis/main.pdf`,
Chapter "Type Prediction with LLMs").

## The two-track study

`scripts/prepare_data.py` builds splits from
`../type_migrator/results/dataset.jsonl` along two axes, so every model is
trained and compared identically on both:

- **Track 1 — `track1_no_gradual`**: both Dialyzer and the typechecker pass,
  and the label contains no `dynamic()` — precise, "safe" labels only.
- **Track 2 — `track2_both_pass`**: both tools pass (`dynamic()` allowed) —
  the larger pool, including gradual-typed labels.

Each track additionally comes in variants describing how the target type
string is shaped (e.g. `expanded`, `compacted`, `hedged`), controlled by
`target_field` in the split — see the per-split `split_info.json` for exact
filters and counts.

Both tracks share **one** subproject → split assignment (computed over
track2's superset of subprojects), so results are directly comparable and no
subproject leaks across train/val/test. A subproject is `project/module_root`
— a GitHub org can contain several independent codebases. Splits are
re-derivable per `seed` (`data/seed<N>/...`) to support a seed-variance study
across models trained on independently reshuffled splits.

**Never compare pass rates across two rounds whose test-set subprojects
differ** — re-run `prepare_data.py` for both rounds under the same seed
before comparing.

## Two model families, two architectures

- **Qwen2.5-Coder** (causal LM) — QLoRA fine-tuning via `train_sft.py` /
  `train_sft_typespec.py`, generation via `generate.py` /
  `generate_typespec.py`. Configs: `configs/qwen7b_qlora*.yaml`,
  `configs/qwen05b_smoke.yaml` (fast local smoke test).
- **CodeT5+** (encoder-decoder) — full fine-tuning via `train_seq2seq.py` /
  `train_seq2seq_typespec.py`, generation via `generate_seq2seq.py` /
  `generate_seq2seq_typespec.py`. Configs: `configs/codet5p_770m*.yaml`,
  `configs/codet5p_2b.yaml`.

Each family has a **Descr track** (target: `elixir_type`, the set-theoretic
annotation; prompt built by `scripts/prompt.py`) and a **TypeSpec track**
(target: `spec`, an Erlang-style `@spec`; prompt built by
`scripts/prompt_typespec.py`). The four training/generation scripts per
family exist so the two tracks can never drift from each other — see the
module docstring in `prompt.py` / `prompt_typespec.py` for the shared prompt
layout.

## Requirements

- Python 3.13, CUDA GPU (training/generation scripts hard-require
  `torch.cuda.is_available()`).
- Key packages: `transformers`, `peft`, `bitsandbytes`, `trl`, `accelerate`,
  `datasets`, `lark` (for `descr_type.lark`/`.gbnf` grammar-constrained
  decoding). No `requirements.txt` is checked in yet — install from
  `.venv`'s resolved set if reproducing:
  ```bash
  python3.13 -m venv .venv && source .venv/bin/activate
  pip install torch transformers peft bitsandbytes trl accelerate datasets lark pyyaml
  ```
- On the training cluster, the SLURM scripts (`slurm/*.sbatch`) activate a
  conda environment named `etr` instead of `.venv` — adjust to your setup.

## Pipeline

```bash
# 1. Build the two-track splits from type_migrator's dataset
python scripts/prepare_data.py

# 2. (optional) Check token-length distribution before picking max_seq_length
python scripts/scan_lengths.py --config configs/qwen7b_qlora.yaml \
    --data_dir data/seed42/track2_both_pass_expanded --files train.jsonl val.jsonl test.jsonl

# 3. Train
python scripts/train_sft.py --config configs/qwen7b_qlora.yaml \
    --data_dir data/seed42/track2_both_pass_expanded --output_dir runs/qwen7b_qlora/track2_both_pass_expanded
# or, encoder-decoder:
python scripts/train_seq2seq.py --config configs/codet5p_770m.yaml \
    --data_dir data/seed42/track2_both_pass_expanded --output_dir runs/codet5p_770m/track2_both_pass_expanded

# 4. Generate predictions on the held-out test set
python scripts/generate.py --model_dir runs/qwen7b_qlora/track2_both_pass_expanded \
    --test_file data/seed42/track2_both_pass_expanded/test.jsonl \
    --out_file runs/qwen7b_qlora/track2_both_pass_expanded/predictions.jsonl
# or:
python scripts/generate_seq2seq.py --model_dir runs/codet5p_770m/track2_both_pass_expanded \
    --test_file data/seed42/track2_both_pass_expanded/test.jsonl \
    --out_file runs/codet5p_770m/track2_both_pass_expanded/predictions.jsonl
```

Scoring happens back in `type_migrator`, since it needs the real project
checkouts and the custom typechecker:

```bash
cd ../type_migrator
mix eval_predictions set-theoretic <predictions.jsonl> [prjs_dir]   # Descr track
mix eval_typespec_predictions <predictions.jsonl> [out.jsonl]       # TypeSpec track
```

Compare two adapters/models evaluated on the same common test set:

```bash
python scripts/compare_tracks.py runs/A/eval_on_common.jsonl runs/B/eval_on_common.jsonl
```

reporting exact-match %, typecheck-pass %, safe&precise %, and emit-`dynamic()`
%, split overall / dynamic-free / with-dynamic by the reference target.

### Cluster (SLURM)

`slurm/*.sbatch` wrap the same scripts for batch submission, e.g.:

```bash
sbatch slurm/run_two_tracks_qwen.sbatch                              # all seeds
sbatch slurm/run_two_tracks_qwen.sbatch configs/qwen7b_qlora.yaml 42 # one seed
```

See the comment header of each `.sbatch` file for what it trains and why
(wall-clock budgeting, which track/variant combinations it runs).

## Repo layout

| Path | Contents |
|---|---|
| `scripts/` | Data prep, prompts, training, generation, comparison/scoring utilities. |
| `configs/` | Per-model YAML configs (quantization, LoRA, training hyperparameters). |
| `data/seed<N>/<track>/` | Generated splits (`train.jsonl`, `val.jsonl`, `test.jsonl`, `split_info.json`); not committed — regenerate with `prepare_data.py`. |
| `runs/` | Training outputs (checkpoints, adapters, predictions); gitignored. |
| `slurm/` | SLURM batch scripts for the university cluster. |
| `logs/` | Local run logs; gitignored. |

## Notes

- `data/`, `runs/`, and `logs/` are gitignored — everything under them is
  regenerable from `type_migrator`'s dataset plus these scripts.
- `.evaluated.jsonl` / `.typechecked.jsonl` outputs go stale if
  `../type_migrator/results/dataset.jsonl` is re-synced afterward —
  re-score rather than reuse them after any raw dataset refresh.
- The `.gbnf` / `.lark` grammars in `scripts/` constrain decoding to
  syntactically valid Elixir Types during generation.

## Related

- [`type_migrator`](../type_migrator) — the translator and dataset pipeline
  this trainer consumes; also home to the thesis document itself.

## License

No license file is included; this is an academic thesis project. Contact
the author for reuse beyond the thesis's own citation.

## Checkpoint-selection study (follow-up)

The thesis selected each checkpoint by lowest validation loss. For Qwen that
loss is computed over the whole prompt + type (`train_sft.py`), so it mostly
measures predicting source code, and it did not track compatibility. This study
selects by the task metric instead and asks which cheap signal could stand in.

Four arms, a 2x2 of loss (full / completion-only) x instruction (on / off):

```bash
# on the cluster: train an arm, keeping every checkpoint (~15)
sbatch slurm/train_masked_qwen.sbatch completion 1       # args: <full|completion> <instr 1|0>
# then signals + constrained generation on a fixed 400-function validation subset
sbatch --array=0-3 slurm/sweep_qwen.sbatch runs/sweep/qwen7b_completion_instr1/seed42 1

# locally (needs type_migrator's Elixir toolchain): score, then collect
scripts/score_sweep.sh runs/sweep/qwen7b_completion_instr1/seed42
python scripts/collect_sweep.py runs/sweep/qwen7b_*_instr*/seed42
```

`collect_sweep.py` writes `sweep/summary.md` per run: compatibility (± one
standard error), exact match and every signal per checkpoint; which checkpoint
each rule picks (validation compatibility, training eval_loss, full-sequence
loss, type-only loss, type-token accuracy); and each signal's Spearman
correlation with compatibility. Only after choosing, generate on the test set
once with the chosen checkpoint (`generate.py --adapter_dir <run>/checkpoint-N`).

The instruction is toggled with `ETR_INCLUDE_INSTRUCTION=0|1` (read by
`prompt.py`; recorded in `prompt_manifest.json`). The sweep refuses to run if
it does not match the run's manifest.

Smoke test before the real runs (0.5B model, 300 examples, 20-function subset):

```bash
python scripts/train_sft_masked.py --config configs/qwen05b_smoke.yaml --loss completion \
  --data_dir data/seed42/track2_both_pass_expanded --output_dir runs/sweep/smoke --max_train_examples 300
python scripts/sweep_checkpoints.py --run_dir runs/sweep/smoke --base_model Qwen/Qwen2.5-Coder-0.5B \
  --data_dir data/seed42/track2_both_pass_expanded --n 20 --constrain
```
(The smoke subset is written as `val_sweep_n20_s42.jsonl`, separate from the real one.)
