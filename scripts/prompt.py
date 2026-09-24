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


def _types_block(example):
    """The types in scope, in Elixir Types notation.

    Fails loudly on a dataset built before the translation existed rather than
    falling back to the source-syntax `type` field: that fallback is the bug
    this field was added to fix, and silently training on the wrong notation
    costs a full run to discover.
    """
    if TYPES_FIELD in example:
        return _block(example, TYPES_FIELD)
    if example.get("type"):
        raise KeyError(
            f"Entry carries 'type' but no {TYPES_FIELD!r}: this dataset predates the "
            "type-in-scope translation. Rebuild it with\n"
            "    mix translate_dataset_types data/dataset.jsonl\n"
            "    mv data/dataset.with_translated_types.jsonl data/dataset.jsonl\n"
            "    python scripts/prepare_data.py 42"
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

    if INCLUDE_TYPES:
        _tagged(parts, "Types in scope", _types_block(example))

    # The definition is the one required field: an example without it is a data
    # bug, so index rather than .get() and let the KeyError surface.
    _tagged(parts, "Definition", example["definition"])

    if INCLUDE_GROUNDING:
        _tagged(parts, "Argument patterns", _block(example, "argument_patterns"))
        _tagged(parts, "Return expressions", _block(example, "return_expressions"))

    parts.append("### Output:\n")
    return "".join(parts)
