"""
Semantic distance for Elixir set-theoretic types.
Operates on the canonical (already-expanded) form produced by the type translator.

All entries in our dataset are function types: (T, ... -> T).
Distance is computed by comparing function arity, then recursively comparing
input types position-by-position and return types.

Inspired by H.B.Mengesha, Ca'Foscari University of Venice (2026).
"""
import re

# ── Key types: the flat leaf vocabulary (no hierarchy among them) ──────────
KEY_TYPES = frozenset({
    "atom", "pid", "port", "reference",
    "float", "integer",
    "bitstring", "binary",
    "tuple", "open_map", "fun",
    "list", "empty_list", "non_empty_list",
    "term", "none", "dynamic",
})


# ── Bracket-aware splitting ────────────────────────────────────────────────
def top_level_split(s, sep):
    """Split string on `sep` only at bracket depth 0."""
    parts, depth, buf = [], 0, []
    for ch in s:
        if ch in '({[%':
            depth += 1
        elif ch in ')}]':
            depth -= 1
        if depth == 0 and _matches_sep(buf, ch, sep, s):
            token = ''.join(buf).strip()
            if sep == ' or ':
                # The separator includes the space; we consumed 'o' as trigger
                # Needs special handling — use regex approach instead
                pass
            parts.append(token)
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append(''.join(buf).strip())
    return [p for p in parts if p]


def _matches_sep(buf, ch, sep, full_str):
    """Helper — not practical for multi-char seps. Use regex split instead."""
    return ch == sep if len(sep) == 1 else False


def split_top_level(s, sep):
    """Split on multi-char separator (like ' or ', ', ') at bracket depth 0."""
    parts = []
    depth = 0
    buf = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in '({[':
            depth += 1
        elif ch in ')}]':
            depth -= 1
        # Check for separator at depth 0
        if depth == 0 and s[i:i+len(sep)] == sep:
            parts.append(''.join(buf).strip())
            buf = []
            i += len(sep)
            continue
        buf.append(ch)
        i += 1
    if buf:
        parts.append(''.join(buf).strip())
    return [p for p in parts if p]


# ── Normalisation ──────────────────────────────────────────────────────────
def normalise(s):
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)
    return s


# ── Parse function type ───────────────────────────────────────────────────
def parse_function(s):
    """Parse '(T1, T2, ... -> Tout)' into ([inputs], output).
    Returns None if not a function type."""
    s = normalise(s)
    if not s.startswith('(') or '->' not in s:
        return None
    # Strip outer parens
    inner = s[1:]
    if inner.endswith(')'):
        inner = inner[:-1]
    # Split on ' -> ' at top level
    parts = split_top_level(inner, ' -> ')
    if len(parts) != 2:
        # Might have nested function types; take last ' -> ' at depth 0
        arrow_parts = split_top_level(inner, ' -> ')
        if len(arrow_parts) < 2:
            return None
        output = arrow_parts[-1]
        inputs_str = ' -> '.join(arrow_parts[:-1])
        parts = [inputs_str, output]

    inputs_str, output_str = parts[0].strip(), parts[1].strip()
    if not inputs_str:
        return ([], output_str)
    inputs = split_top_level(inputs_str, ', ')
    return (inputs, output_str)


# ── Parse union ────────────────────────────────────────────────────────────
def parse_union(s):
    """Split a type into union arms at the top level. Returns list of arms."""
    arms = split_top_level(normalise(s), ' or ')
    return arms if len(arms) > 1 else [normalise(s)]


# ── Parse tuple ────────────────────────────────────────────────────────────
def parse_tuple(s):
    """Parse '{T1, T2, ...}' into list of element types. Returns None if not tuple."""
    s = normalise(s)
    if not s.startswith('{') or not s.endswith('}'):
        return None
    inner = s[1:-1]
    return split_top_level(inner, ', ')


# ── Shape categorisation ──────────────────────────────────────────────────
def shape(s):
    s = normalise(s)
    if s.startswith('(') and '->' in s:
        return 'function'
    if s.startswith('{'):
        return 'tuple'
    if s.startswith('%{'):
        return 'map'
    if s.startswith('non_empty_list(') or s == 'empty_list()':
        return 'list'
    # Check for top-level union
    arms = split_top_level(s, ' or ')
    if len(arms) > 1:
        return 'union'
    if s.startswith(':'):
        return 'atom_literal'
    return 'atomic'


# ── Recursive type similarity ─────────────────────────────────────────────
def type_similarity(a, b):
    """Compute similarity between two type strings, recursively.
    Returns float in [0, 1]."""
    a, b = normalise(a), normalise(b)

    if a == b:
        return 1.0

    sa, sb = shape(a), shape(b)

    # Shape mismatch — low but not zero if either is 'term' or 'dynamic'
    if sa != sb:
        if a in ('term()', 'dynamic()') or b in ('term()', 'dynamic()'):
            return 0.2  # gradual/top type matches anything weakly
        return 0.0

    # ── Function types ──
    if sa == 'function':
        fa, fb = parse_function(a), parse_function(b)
        if fa is None or fb is None:
            return 0.0
        inputs_a, out_a = fa
        inputs_b, out_b = fb
        # Arity mismatch → very low
        if len(inputs_a) != len(inputs_b):
            return 0.05
        # Compare inputs position-by-position
        if inputs_a:
            input_sim = sum(type_similarity(ia, ib)
                           for ia, ib in zip(inputs_a, inputs_b)) / len(inputs_a)
        else:
            input_sim = 1.0
        output_sim = type_similarity(out_a, out_b)
        # Weight output slightly more (it's what callers depend on)
        return 0.4 * input_sim + 0.6 * output_sim

    # ── Tuples ──
    if sa == 'tuple':
        ta, tb = parse_tuple(a), parse_tuple(b)
        if ta is None or tb is None:
            return 0.0
        if len(ta) != len(tb):
            return 0.1  # different arity tuple
        if not ta:
            return 1.0  # both empty tuples
        return sum(type_similarity(ea, eb) for ea, eb in zip(ta, tb)) / len(ta)

    # ── Unions ──
    if sa == 'union':
        arms_a = set(parse_union(a))
        arms_b = set(parse_union(b))
        if arms_a == arms_b:
            return 1.0
        # Greedy bipartite matching (like Mengesha's approach)
        matched_sim = _greedy_match_similarity(list(arms_a), list(arms_b))
        return matched_sim

    # ── Maps ── (structural comparison is complex; fall back to token overlap)
    if sa == 'map':
        return _token_jaccard(a, b)

    # ── Lists ──
    if sa == 'list':
        # Both are list-shaped; compare inner types if possible
        inner_a = _extract_list_inner(a)
        inner_b = _extract_list_inner(b)
        if inner_a is not None and inner_b is not None:
            return type_similarity(inner_a, inner_b)
        return _token_jaccard(a, b)

    # ── Atom literals ──
    if sa == 'atom_literal':
        return 1.0 if a == b else 0.0

    # ── Atomic key types ──
    # In the set-theoretic system, key types are flat — equal or not
    a_base = re.sub(r'\(\)', '', a)
    b_base = re.sub(r'\(\)', '', b)
    return 1.0 if a_base == b_base else 0.0


def _greedy_match_similarity(arms_a, arms_b):
    """Greedy bipartite matching between union arms.
    Match each arm in the smaller set to its best match in the larger set."""
    if not arms_a or not arms_b:
        return 0.0

    # Compute pairwise similarities
    sims = []
    used_b = set()
    for aa in arms_a:
        best_sim, best_j = 0.0, -1
        for j, bb in enumerate(arms_b):
            if j not in used_b:
                s = type_similarity(aa, bb)
                if s > best_sim:
                    best_sim, best_j = s, j
        sims.append(best_sim)
        if best_j >= 0:
            used_b.add(best_j)

    # Normalise by the size of the larger set (penalise missing/extra arms)
    return sum(sims) / max(len(arms_a), len(arms_b))


def _extract_list_inner(s):
    """Extract the element type from non_empty_list(T, T') or empty_list()."""
    m = re.match(r'non_empty_list\((.+),\s*(.+)\)$', s)
    if m:
        return m.group(1).strip()
    if s == 'empty_list()':
        return 'none()'  # empty list has no elements
    return None


def _token_jaccard(a, b):
    """Fallback: extract all type tokens and compute Jaccard."""
    pattern = r'\b(' + '|'.join(sorted(KEY_TYPES, key=len, reverse=True)) + r')\b'
    ta = set(re.findall(pattern, a))
    tb = set(re.findall(pattern, b))
    ta.update(re.findall(r':\w+', a))  # atom literals
    tb.update(re.findall(r':\w+', b))
    if not ta and not tb:
        return 0.5
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union > 0 else 0.0


# ── Public API ─────────────────────────────────────────────────────────────
def semantic_distance(predicted, reference):
    """Return (distance, similarity, reason)."""
    p = normalise(predicted)
    r = normalise(reference)

    if not p or not r:
        return 3, 0.0, "empty type"

    if p == r:
        return 0, 1.0, "exact"

    sim = type_similarity(p, r)

    # Map continuous similarity to discrete distance
    if sim >= 0.85:
        return 0, sim, f"equivalent ({sim:.3f})"
    elif sim >= 0.5:
        return 1, sim, f"close ({sim:.3f})"
    elif sim >= 0.25:
        return 2, sim, f"partial ({sim:.3f})"
    else:
        return 3, sim, f"distant ({sim:.3f})"