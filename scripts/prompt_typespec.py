"""Prompt / encoder-input format for the TYPESPEC track.

Counterpart of prompt.py, which targets the set-theoretic annotation. Kept as a
separate module rather than a flag on that one so the two tracks cannot disturb
each other: prompt.py is the single source of truth for the four Descr-track
scripts, and this file is the single source of truth for the four TypeSpec-track
scripts (train_sft_typespec.py, train_seq2seq_typespec.py, generate_typespec.py,
generate_seq2seq_typespec.py).

What differs from prompt.py:
  * the INSTRUCTION asks for an Erlang-style `@spec`, not Descr syntax;
  * the target is the entry's `spec` field, not `elixir_type` (TARGET_FIELD).

What is deliberately IDENTICAL: the blocks, their order, their tags, and the
`### Output:` seam. The input side of the two tracks is therefore the same text,
so a difference in results is a difference in what the model was asked to write.

Note that the `Types in scope` block already carries TypeSpec syntax on BOTH
tracks -- the entries' `type` field is the original `@type` declarations
(`@type t :: Hound.BrowserLike.t()`), never their translation. On this track the
prompt and the target are consequently in one and the same notation.

Layout invariants are those of prompt.py: every block is introduced by a
`Label:` tag, an empty block is omitted tag and all, and every block ends in a
blank line.
"""

INSTRUCTION = (
    "For the given Elixir function definition, infer the most precise correct "
    "function type, written as an Erlang-style Elixir TypeSpec. "
    "Respond only with a single @spec attribute."
)

# The supervised target. Read by the two trainers and by the two generators (to
# record the reference alongside each prediction), so the track's target is
# named in exactly one place.
TARGET_FIELD = "spec"

# The field the `Types in scope` block reads. This track keeps `type`, the
# declarations in their original source form, because TypeSpec syntax is what
# matches its target. The Descr track reads `translated_type` instead, for the
# same reason in the other notation.
TYPES_FIELD = "type"

# Same budget as prompt.py, for the same reason. Inert on this track today --
# the source declarations top out at 1,867 characters, well inside it -- but it
# keeps the two modules' behaviour identical, so a prompt difference between the
# tracks can only ever come from the fields they read.
TYPES_CHAR_BUDGET = 2000

# Prompt-variant toggles -- mirror prompt.py's v1.6 settings.
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
    """
    kept, used = [], 0
    for decl in declarations:
        need = len(decl) + (1 if kept else 0)
        if used + need > budget:
            continue
        kept.append(decl)
        used += need
    return "\n".join(kept)


def build_prompt(example):
    """The full prompt / encoder input, up to and including '### Output:\\n'."""
    parts = [
        "### Instruction:\n",
        INSTRUCTION,
        "\n\n",
        "### Input:\n",
    ]

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
        _tagged(parts, "Types in scope", _budgeted(example.get(TYPES_FIELD) or [], TYPES_CHAR_BUDGET))

    if INCLUDE_GROUNDING:
        _tagged(parts, "Argument patterns", _block(example, "argument_patterns"))
        _tagged(parts, "Return expressions", _block(example, "return_expressions"))

    parts.append("### Output:\n")
    return "".join(parts)
