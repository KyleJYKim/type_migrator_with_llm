"""Record the exact prompts a run trained on and generated from.

Nothing else in the pipeline preserves the text the model actually saw: the
trainers tokenize and discard it, and the generators keep only the prediction.
That leaves questions like "were the types in scope translated or original?"
answerable only by re-deriving the prompt from the data, which is a different
computation from the one the run performed. These logs answer them from the run
itself.

Two artefacts per run directory:

  prompt_manifest.json   what the format WAS -- the instruction text, the
                         INCLUDE_* toggles, the target field, and a sha256 of
                         the prompt module's source. Two runs whose manifests
                         agree were built by the same prompt code; a differing
                         hash is prompt drift, the failure mode prompt.py's
                         single-source-of-truth rule exists to prevent.

  prompts_train.jsonl    a sample of training texts, exactly as handed to the
  prompts_generate.jsonl trainer, and every generation prompt with the raw
                         continuation beside the parsed answer (so a parser bug
                         is visible as a good continuation next to a bad parse).

Both are written beside the model, so a run is self-describing when it is read
back months later.
"""
import hashlib
import inspect
import json
import random
from datetime import datetime, timezone
from pathlib import Path


def _module_fingerprint(prompt_module):
    """sha256 of the prompt module's source, to detect format drift."""
    try:
        src = inspect.getsource(prompt_module)
    except (OSError, TypeError):
        return None
    return hashlib.sha256(src.encode()).hexdigest()


def write_manifest(out_dir, prompt_module, *, phase, extra=None, example=None):
    """Record the prompt format in effect. Returns the manifest dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "phase": phase,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prompt_module": getattr(prompt_module, "__name__", str(prompt_module)),
        "prompt_module_sha256": _module_fingerprint(prompt_module),
        "instruction": getattr(prompt_module, "INSTRUCTION", None),
        "target_field": getattr(prompt_module, "TARGET_FIELD", None),
        "types_field": getattr(prompt_module, "TYPES_FIELD", None),
        "types_char_budget": getattr(prompt_module, "TYPES_CHAR_BUDGET", None),
        "toggles": {
            name: getattr(prompt_module, name)
            for name in ("INCLUDE_MODULE", "INCLUDE_FUNCTION", "INCLUDE_TYPES",
                         "INCLUDE_GROUNDING")
            if hasattr(prompt_module, name)
        },
    }
    if extra:
        manifest.update(extra)
    if example is not None:
        manifest["example_prompt"] = example

    path = out / "prompt_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"=== Prompt manifest: {path} ===")
    print(f"    instruction : {manifest['instruction']}")
    print(f"    target      : {manifest['target_field']}")
    print(f"    types field : {manifest['types_field']}")
    print(f"    toggles     : {manifest['toggles']}")
    print(f"    module sha  : {manifest['prompt_module_sha256']}")
    return manifest


class LazyTexts:
    """Sequence view that renders an item only for the indices actually logged.

    The seq2seq trainer keeps its examples tokenized, so the text has to be
    rebuilt to be logged. Rebuilding all of them to keep twenty is wasteful;
    this renders on access, preserving true dataset indices in the log.
    `render` may return a string or a dict of fields.
    """

    def __init__(self, source, render):
        self._source = source
        self._render = render

    def __len__(self):
        return len(self._source)

    def __getitem__(self, i):
        return self._render(self._source[i])


def log_training_prompts(out_dir, texts, *, n_samples=20, seed=42, extra_per_record=None):
    """Write a sample of training texts verbatim.

    `texts` is any sequence of what the trainer consumes: strings, or dicts of
    fields (for an encoder-decoder, whose prompt and target are separate
    streams). The first example is always included -- it is the one printed to
    the run log, so the two can be compared -- and the rest are drawn at random
    for a spread across projects.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    total = len(texts)
    if total == 0:
        return None

    n = min(n_samples, total)
    rng = random.Random(seed)
    idx = [0] + rng.sample(range(1, total), n - 1) if total > 1 and n > 1 else [0]
    idx = sorted(set(idx))

    path = out / "prompts_train.jsonl"
    with open(path, "w") as f:
        for i in idx:
            rendered = texts[i]
            record = {"index": i}
            record.update(rendered if isinstance(rendered, dict) else {"text": rendered})
            if extra_per_record:
                record.update(extra_per_record(i))
            f.write(json.dumps(record) + "\n")
    print(f"=== Logged {len(idx)} of {total} training prompts to {path} ===")
    return path


class GenerationPromptLog:
    """Append one record per generated entry: prompt in, raw text out, parse.

    Used as a context manager so the file is closed even if generation is cut
    short by a time limit, leaving the entries completed so far readable.
    """

    def __init__(self, out_dir, filename="prompts_generate.jsonl"):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.path = out / filename
        self._f = None
        self.n = 0

    def __enter__(self):
        self._f = open(self.path, "w")
        return self

    def write(self, *, index, prompt, raw_output, parsed, reference=None, entry=None):
        record = {
            "index": index,
            "prompt": prompt,
            "raw_output": raw_output,
            "parsed": parsed,
            "reference": reference,
        }
        if entry is not None:
            # Enough to find the entry in the dataset without duplicating it.
            record["locator"] = {
                k: entry.get(k) for k in ("project", "file", "module", "function", "arity")
            }
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()
        self.n += 1

    def __exit__(self, *exc):
        if self._f is not None:
            self._f.close()
        print(f"=== Logged {self.n} generation prompts to {self.path} ===")
        return False
