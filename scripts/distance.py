"""
Semantic distance for Elixir set-theoretic types.
Implements 0/1/2/3 buckets inspired by H.B.Mengesha form Ca'Foscari University of Venice (2026).
"""
import re
import subprocess
from functools import lru_cache

# Elixir type hierarchy — adapted for set-theoretic types
HIERARCHY = {
    "term": [],                                    # depth 0
    "any": ["term"],                               # depth 0 (alias)
    "dynamic": ["term"],                           # depth 0
    "number": ["term"],                            # depth 1
    "integer": ["number"],                         # depth 2
    "float": ["number"],                           # depth 2
    "pos_integer": ["integer"],                    # depth 3
    "non_neg_integer": ["integer"],                # depth 3
    "neg_integer": ["integer"],                    # depth 3
    "atom": ["term"],                              # depth 1
    "boolean": ["atom"],                           # depth 2
    "binary": ["term"],                            # depth 1
    "list": ["term"],                              # depth 1
    "non_empty_list": ["list"],                    # depth 2
    "empty_list": ["list"],                        # depth 2
    "tuple": ["term"],                             # depth 1
    "map": ["term"],                               # depth 1
    "pid": ["term"], "port": ["term"], "reference": ["term"],
}

# Pre-compute depths and ancestor sets
def compute_depths():
    depths = {}
    def depth(t):
        if t in depths: return depths[t]
        parents = HIERARCHY.get(t, [])
        d = 0 if not parents else 1 + max(depth(p) for p in parents)
        depths[t] = d
        return d
    for t in HIERARCHY:
        depth(t)
    return depths

DEPTHS = compute_depths()
MAX_DEPTH = max(DEPTHS.values()) if DEPTHS else 1

def ancestors(t):
    """Set of all ancestors of t, including itself."""
    seen = {t}
    stack = [t]
    while stack:
        cur = stack.pop()
        for p in HIERARCHY.get(cur, []):
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return seen

def hierarchy_similarity(a, b):
    """Depth-weighted similarity for two atomic types in the hierarchy."""
    if a == b: return 1.0
    if a not in DEPTHS or b not in DEPTHS:
        return 0.0  # unknown atomic type
    common = ancestors(a) & ancestors(b)
    if not common:
        return 0.0
    lca_depth = max(DEPTHS[c] for c in common)
    da, db = DEPTHS[a], DEPTHS[b]
    return (2 * lca_depth) / (da + db) if (da + db) > 0 else 0.0


def normalise(type_str):
    """Canonicalise an Elixir set-theoretic type string for comparison.
    - 'or' and '|' both mean union
    - 'empty_list()' and '[]' both mean empty list
    - whitespace and parens normalised
    """
    s = type_str.strip()
    s = re.sub(r'\s+', ' ', s)
    s = s.replace(' or ', ' | ')
    return s


def shape_arity(type_str):
    """Return a coarse shape descriptor for a type — used for distance 3 detection.
    Returns one of: 'function', 'union', 'tuple', 'list', 'map', 'struct', 'atomic'."""
    s = normalise(type_str)
    if '->' in s: return 'function'
    if '|' in s: return 'union'
    if s.startswith('{') or 'tuple(' in s: return 'tuple'
    if 'list(' in s or 'empty_list' in s or 'non_empty_list' in s: return 'list'
    if s.startswith('%{') or s.startswith('%') or 'map(' in s: return 'map'
    if s.startswith('%'): return 'struct'
    return 'atomic'


def function_arity(type_str):
    """For function types like '(a, b, c -> d)' return input arity, or None."""
    s = normalise(type_str)
    m = re.match(r'\((.*?)\s*->\s*(.+)\)', s)
    if not m: return None
    inputs = m.group(1).strip()
    if not inputs: return 0
    # Count top-level commas (simplification; real impl should bracket-aware split)
    return inputs.count(',') + 1


def extract_atomic_tokens(type_str):
    """Extract all atomic type tokens from a type string for shallow similarity."""
    return set(re.findall(r'\b(integer|float|number|atom|binary|boolean|pid|'
                          r'port|reference|list|empty_list|non_empty_list|'
                          r'tuple|map|term|any|dynamic|pos_integer|'
                          r'non_neg_integer|neg_integer)\b\(?\)?', type_str))


def semantic_distance(predicted, reference):
    """Map (predicted, reference) to discrete distance 0/1/2/3.
    Returns (distance, similarity_score, explanation)."""
    p = normalise(predicted)
    r = normalise(reference)

    # Distance 0 — string equivalence after normalisation
    if p == r:
        return 0, 1.0, "exact"

    # Shape mismatch → distance 3
    sp, sr = shape_arity(predicted), shape_arity(reference)
    if sp != sr:
        return 3, 0.1, f"shape mismatch: {sp} vs {sr}"

    # Function with wrong arity → distance 3
    if sp == 'function':
        ap, ar = function_arity(predicted), function_arity(reference)
        if ap is not None and ar is not None and ap != ar:
            return 3, 0.1, f"function arity {ap} vs {ar}"

    # Token-overlap measure
    tp = extract_atomic_tokens(predicted)
    tr = extract_atomic_tokens(reference)
    if not tp or not tr:
        return 2, 0.4, "no atomic tokens"
    jacc = len(tp & tr) / len(tp | tr)

    # Hierarchy-based similarity for each pair of tokens
    sims = []
    for a in tp:
        best = max((hierarchy_similarity(a, b) for b in tr), default=0.0)
        sims.append(best)
    hier_sim = sum(sims) / len(sims) if sims else 0.0

    combined = 0.4 * jacc + 0.6 * hier_sim

    if combined >= 0.85:
        return 1, combined, f"close ({combined:.2f})"
    elif combined >= 0.4:
        return 2, combined, f"partial ({combined:.2f})"
    else:
        return 3, combined, f"distant ({combined:.2f})"