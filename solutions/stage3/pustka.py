from typing import List, Tuple
import numpy as np

def solve_release_set(
    activations,
    rows,
    w,
    b,
) -> List[Tuple[str, str]]:
    """Find five hidden (concept, value) triggers from the linear-head margins."""
    activations = np.asarray(activations, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    logits = activations @ w + float(np.asarray(b))
    positive = logits >= 0.0

    pair_masks = {}
    for row_index, row in enumerate(rows):
        for concept, value in row["concepts"].items():
            pair = (concept, value)
            if pair not in pair_masks:
                pair_masks[pair] = np.zeros(len(rows), dtype=bool)
            pair_masks[pair][row_index] = True

    # A true OR-trigger cannot occur in a negative report. Among the remaining
    # correlated candidates, true triggers have the larger mean positive logit.
    candidates = []
    for pair, present in pair_masks.items():
        present_positive = present & positive
        if present_positive.any() and not (present & ~positive).any():
            candidates.append((pair, float(logits[present_positive].mean())))

    # If a release set has fewer than five perfectly positive-only candidates,
    # retain a penalized fallback so the function still returns five pairs.
    if len(candidates) < 5:
        candidates = []
        for pair, present in pair_masks.items():
            if not present.any():
                continue
            pos_mean = float(logits[present & positive].mean()) if (present & positive).any() else -1e6
            neg_rate = float((present & ~positive).sum()) / float(present.sum())
            candidates.append((pair, pos_mean - 10.0 * neg_rate))

    candidates.sort(key=lambda item: (-item[1], str(item[0][0]), str(item[0][1])))
    return [pair for pair, _ in candidates[:5]]
