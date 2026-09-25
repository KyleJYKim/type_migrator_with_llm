"""Single source of truth for the LLM prompt / encoder-input format.

Imported by train_sft.py, generate.py, train_seq2seq.py and generate_seq2seq.py
so the four can never drift (a prompt mismatch between training and inference
silently degrades generation). Switch prompt variants for ALL of them at once by
flipping the INCLUDE_* flags below -- no more commenting the same lines in three
files.

`build_prompt(example)` returns the whole prompt up to and including
`### Output:\n`. For the decoder-only (Qwen) run that string is the model's
prompt and the completion (`elixir_type` + EOS) is appended by the caller; for
the encoder-decoder (CodeT5+) run it is the encoder input and `elixir_type` is
the decoder target. The completion/EOS handling therefore stays in each caller;
only the prompt is shared.

Layout invariants (keep these when adding a field):
  * every block is introduced by a `Label:` tag, including the definition --
    the types block and the definition are both Elixir source, so a blank line
    alone is not a reliable boundary for the model;
  * a block whose content is empty is omitted entirely, tag included -- a bare
    `Types in scope:` with nothing under it teaches the tag to mean "nothing
    follows";
  * every block ends in a blank line, so toggling one INCLUDE_* flag does not
    reflow the others.
"""

INSTRUCTION = (
    "For the given Elixir function definition, infer the most precise correct "
    "function type, written as an Elixir set-theoretic type annotation. "
    "Respond only with Elixir Types (Descr) syntax."
)

# The supervised target, and the field the `Types in scope` block reads.
#
# TYPES_FIELD is `translated_type`, NOT `type`. An entry's `type` field holds
# the @type declarations as they appear in the source -- TypeSpec syntax -- so
# using it here had this track reading its context in one notation while writing
# its answer in another. `translated_type` is the same declarations rendered as
# Elixir Types (added by `mix translate_dataset_types`), which matches the
# target. The TypeSpec track keeps reading `type`, where source syntax is what
# matches ITS target.
TARGET_FIELD = "elixir_type"
TYPES_FIELD = "translated_type"

# Character budget for the `Types in scope` block.
#
# Translation EXPANDS a type: a compact source reference becomes its fully
# resolved Descr form, and a recursive or variable-arity one can blow up without
# bound. In the current dataset the source block tops out at 1,867 characters
# while its translation reaches 531,964 (Membrane's SchemeParser) -- enough to
# push a single prompt past 133k tokens and OOM a generation run mid-way.
#
# The dataset already bounds the cross-module part of the source block at
# construction time (TranslationRunner's @max_referenced_chars); this is the
# same idea applied after translation, where the expansion actually happens.
# 2,000 keeps the block at the scale the source one had, and costs 4.3% of
# entries part of their context -- against a 1,024-token training budget that a
# larger block would exhaust on its own.
TYPES_CHAR_BUDGET = 2000

# Prompt-variant toggles -- flip in this one place to switch every script.
# v1   = definition only.
# v1.5 = module + user types.
# v1.6 = v1.5 + `Definition:` tag, empty blocks dropped, uniform blank lines.
INCLUDE_MODULE = True
INCLUDE_FUNCTION = False
INCLUDE_TYPES = True
INCLUDE_GROUNDING = False  # argument patterns + return expressions


def _block(example, key):
    """Join a list field (or stringify a scalar) into a block; '' if absent."""
    v = example.get(key)
    if not v:
        return ""
    return "\n".join(v) if isinstance(v, list) else str(v)


def _tagged(parts, label, body):
    """Append a `label:\\n<body>\\n\\n` block, or nothing if body is empty."""
    if body:
        parts.append(f"{label}:\n{body}\n\n")


def _budgeted(declarations, budget):
    """Join declarations up to `budget` characters, keeping source order.

    Whole declarations only: a type cut in half is worse than an absent one,
    since it teaches the model a syntax that never denotes anything.

    An oversized declaration is SKIPPED rather than ending the block, because
    the one construct that blows up here -- a variable-arity `(... -> T)`,
    expanded by the translation into a union over arities 0..255 -- can appear
    early among several small, useful ones. Declarations are independent lines,
    so keeping a later one without an earlier one loses nothing.
    """
    kept, used = [], 0
    for decl in declarations:
        need = len(decl) + (1 if kept else 0)
        if used + need > budget:
            continue
        kept.append(decl)
        used += need
    return "\n".join(kept)


def _types_block(example):
    """The types in scope, in Elixir Types notation.

    Fails loudly on a dataset built before the translation existed rather than
    falling back to the source-syntax `type` field: that fallback is the bug
    this field was added to fix, and silently training on the wrong notation
    costs a full run to discover.
    """
    if TYPES_FIELD in example:
        return _budgeted(example.get(TYPES_FIELD) or [], TYPES_CHAR_BUDGET)
    if example.get("type"):
        raise KeyError(
            f"Entry carries 'type' but no {TYPES_FIELD!r}: these splits predate the "
            "type-in-scope translation.\n"
            "\n"
            "  On the machine with the Elixir toolchain, rebuild them:\n"
            "    mix translate_dataset_types data/dataset.jsonl\n"
            "    mv data/dataset.with_translated_types.jsonl data/dataset.jsonl\n"
            "    python scripts/prepare_data.py 42\n"
            "\n"
            "  On a training node, COPY the rebuilt splits across -- do NOT run\n"
            "  prepare_data.py here. It would regenerate them from this machine's\n"
            "  dataset.jsonl, which lacks the field, overwriting good splits:\n"
            "    rsync -av <local>/data/seed42/<split-dir> $PWD/data/seed42/\n"
            "\n"
            "  Verify with:\n"
            "    head -1 data/seed42/<split-dir>/train.jsonl | "
            f"python -c \"import json,sys; print({TYPES_FIELD!r} in json.load(sys.stdin))\""
        )
    return ""


def build_prompt(example):
    """The full prompt / encoder input, up to and including '### Output:\\n'."""
    parts = [
        "### Instruction:\n",
        INSTRUCTION,
        "\n\n",
        "### Input:\n",
    ]

    # One-line metadata fields share a single block so that enabling both does
    # not put a blank line between two one-liners.
    meta = []
    if INCLUDE_MODULE and (module := _block(example, "module")):
        meta.append(f"Module: {module}")
    if INCLUDE_FUNCTION and (function := _block(example, "function")):
        meta.append(f"Function: {function}/{example['arity']}")
    if meta:
        parts.append("\n".join(meta) + "\n\n")

    # Definition FIRST, types in scope after it. If anything ever truncates this
    # prompt -- an encoder cap, a context limit -- truncation takes the tail, and
    # the tail must be the expendable part. With types first, an oversized type
    # block could consume the whole window and cut away the function itself and
    # the `### Output:` marker, leaving the model to answer from a fragment of a
    # type declaration; that happened, silently, at a 512-token encoder cap.
    # Losing trailing type declarations is a graceful degradation instead.
    #
    # The definition is the one required field: an example without it is a data
    # bug, so index rather than .get() and let the KeyError surface.
    _tagged(parts, "Definition", example["definition"])

    if INCLUDE_TYPES:
        _tagged(parts, "Types in scope", _types_block(example))

    if INCLUDE_GROUNDING:
        _tagged(parts, "Argument patterns", _block(example, "argument_patterns"))
        _tagged(parts, "Return expressions", _block(example, "return_expressions"))

    parts.append("### Output:\n")
    return "".join(parts)
