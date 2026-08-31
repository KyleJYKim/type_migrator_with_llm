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
"""

INSTRUCTION = (
    "For the given Elixir function definition, infer the most precise correct "
    "function type, written as an Elixir set-theoretic type annotation. "
    "Respond only with Elixir Types (Descr) syntax."
)

# Prompt-variant toggles -- flip in this one place to switch every script.
# v1 = definition only. Current = v1.5 (module + user types).
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


def build_prompt(example):
    """The full prompt / encoder input, up to and including '### Output:\\n'."""
    parts = [
        "### Instruction:\n",
        INSTRUCTION,
        "\n\n",
        "### Input:\n",
    ]
    if INCLUDE_MODULE:
        parts.append(f"Module: {example['module']}\n")
    if INCLUDE_FUNCTION:
        parts.append(f"Function: {example['function']}/{example['arity']}\n")
    if INCLUDE_TYPES:
        parts.append(f"Types in scope:\n{_block(example, 'type')}\n\n")
    parts.append(f"{example['definition']}\n\n")
    if INCLUDE_GROUNDING:
        parts.append(f"Argument patterns:\n{_block(example, 'argument_patterns')}\n\n")
        parts.append(f"Return expressions:\n{_block(example, 'return_expressions')}\n\n")
    parts.append("### Output:\n")
    return "".join(parts)
