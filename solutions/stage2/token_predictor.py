"""Reference selector for the Poland 2026 Stage 2 Token Predictor task.

The fixed coefficients below were fit on synthetic 512 -> 128 windows from the
released ``train_data`` stream. Each synthetic negative pool is the unique token
IDs in a separate 512-token train-stream passage. No validation labels or test
data are used by this implementation. The optional GPT score is disabled by
default because the linked checkpoint was unavailable in development.
"""

import numpy as np
import sys

try:  # PyTorch is present in the competition notebook; keep local import optional.
    import torch
except Exception:  # pragma: no cover - only used by the no-Torch fallback.
    torch = None


_FEATURE_MEAN = np.array([
    4.439068193316061, 7.5808674621186505, 0.32660577188298817,
    0.011596554301948505, 0.02197783436966874, 0.04064807202856513,
    0.0723129446508735, 0.1235428903836694, 0.20539516229626634,
    0.2708107025597128, 0.32660577188298817, 0.018205440508726144,
    0.035132483759573965, 0.06873787981149397, 0.1351611388203299,
    0.2618426371948405, 0.4566788500004205, 0.6638684132018945,
    0.2647733219987104, 0.14929028489806068,
], dtype=np.float64)

_FEATURE_SCALE = np.array([
    1.9453688139198702, 1.8419604652077692, 0.6482562535473356,
    0.09165644305498111, 0.13064430140027666, 0.18731052398013506,
    0.26483061425757753, 0.3652494606517419, 0.49312811124244105,
    0.5807374601567306, 0.6482562535473356, 0.1152347974453774,
    0.18517550220255316, 0.3221801734017333, 0.5918317114819062,
    1.107257373779291, 1.9077164632841024, 2.7679248635975813,
    0.4412124317784936, 0.35637437580744263,
], dtype=np.float64)

_LOGIT_COEF = np.array([
    0.036170994594721985, -0.008155684753416471, 0.11274541700691562,
    0.0006105046836685159, 0.003234422842235215, 0.046435253263629205,
    0.10919321103044169, 0.0812843312483594, 0.06522753787115525,
    0.0038248940943025334, 0.11274541700691562, 0.03161033431627779,
    0.01647783389316946, -0.07417918247656942, -0.16562777234102322,
    -0.06635250660690528, 0.2100115080949274, 0.4410114644265936,
    0.11645010143341361, 0.0221769163112388,
], dtype=np.float64)

_LOGIT_INTERCEPT = -0.0291854165435196
_VOCAB_SIZE = 50257
_GOOD_COUNT_PRIOR = 83  # round(mean unique 128-token horizon size: 82.53)


def _to_numpy(value, dtype=None):
    """Convert Python, NumPy, or CPU/GPU tensor values into a NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


class TokenPredictorSolution:
    """Rank candidate tokens using a train-stream-fitted logistic model.

    ``model_logit_blend`` is an experimental option.  It can add a standardized
    score aggregated from the GPT model's next-token log probabilities at the
    last 128 prompt positions.  Leave it at 0 unless that feature is validated
    on the official checkpoint and within the five-minute budget.
    """

    def __init__(self, model_logit_blend=0.0):
        self.model_logit_blend = float(model_logit_blend)
        self._token_counts = None

    @staticmethod
    def _notebook_global(name):
        """Find data/model globals whether this code is pasted or imported."""
        value = globals().get(name)
        if value is None:
            main_module = sys.modules.get("__main__")
            value = getattr(main_module, name, None) if main_module is not None else None
        return value

    def _get_token_counts(self, token_list):
        if self._token_counts is None:
            stream = self._notebook_global("train_data")
            if stream is None:
                return np.zeros(max(_VOCAB_SIZE, max(token_list, default=-1) + 1), dtype=np.int64)
            values = _to_numpy(stream).reshape(-1).astype(np.int64, copy=False)
            size = max(_VOCAB_SIZE, int(values.max()) + 1 if values.size else _VOCAB_SIZE)
            self._token_counts = np.bincount(values, minlength=size)
        if len(token_list) and max(token_list) >= len(self._token_counts):
            pad = np.zeros(max(token_list) + 1 - len(self._token_counts), dtype=np.int64)
            self._token_counts = np.concatenate([self._token_counts, pad])
        return self._token_counts

    def _candidate_features(self, prompt, candidates, token_counts):
        p = _to_numpy(prompt, dtype=np.int64).reshape(-1)
        p = p[-512:]
        c = np.asarray(candidates, dtype=np.int64)
        vocab_size = max(len(token_counts), _VOCAB_SIZE, int(c.max()) + 1 if c.size else _VOCAB_SIZE)

        # The column order matches the synthetic-window fit described above.
        columns = [np.log1p(token_counts[c]), np.log1p(c)]
        for width in (0, 8, 16, 32, 64, 128, 256, 384, 512):
            window = p[-(width or len(p)):]
            freq = np.bincount(window, minlength=vocab_size)
            columns.append(np.log1p(freq[c]))
        positions = np.arange(len(p), dtype=np.float64)
        for decay in (8, 16, 32, 64, 128, 256, 512):
            weights = np.exp(-(len(p) - 1 - positions) / float(decay))
            weighted = np.bincount(p, weights=weights, minlength=vocab_size)
            columns.append(weighted[c])
        prompt_freq = np.bincount(p, minlength=vocab_size)[c]
        columns.extend([(prompt_freq > 0).astype(np.float64),
                        (prompt_freq > 1).astype(np.float64)])
        return np.column_stack(columns)

    def _model_context_score(self, model, prompt, candidates):
        """Return mean next-token log probability over the prompt's last 128 steps."""
        if torch is None or model is None or not callable(model) or not candidates:
            return None
        try:
            try:
                device = next(model.parameters()).device
            except (AttributeError, StopIteration, TypeError):
                device = None
            p = prompt
            if not torch.is_tensor(p):
                p = torch.as_tensor(p, dtype=torch.long, device=device)
            else:
                p = p.to(device=device, dtype=torch.long)
            p = p.reshape(-1)[-512:]
            with torch.no_grad():
                logits = model(p.unsqueeze(0))[0]
                count = min(128, int(logits.shape[0]))
                log_probs = torch.log_softmax(logits[-count:].float(), dim=-1)
                candidate_ids = torch.as_tensor(candidates, dtype=torch.long, device=logits.device)
                selected = log_probs.index_select(1, candidate_ids)
                aggregate = torch.logsumexp(selected, dim=0) - np.log(float(count))
            return _to_numpy(aggregate, dtype=np.float64)
        except Exception:
            # Fall back cleanly if a platform wrapper exposes a nonstandard model.
            return None

    def classify(self, model, prompt=None, token_list=None):
        """Return disjoint ``(bad_answer, good_answer)`` candidate partitions.

        The three-argument form matches the official evaluator.  The two-argument
        form is also accepted for the starter notebook's ``classify(prompt, list)``
        call; in that case the model is read from the notebook global if present.
        """
        if token_list is None:
            token_list = prompt
            prompt = model
            model = self._notebook_global("model")

        # The task normally supplies unique IDs, but collapse duplicates before
        # ranking so an ID can never be placed in both output partitions.
        original = [int(x) for x in _to_numpy(token_list).reshape(-1).tolist()]
        candidates = np.unique(np.asarray(original, dtype=np.int64))
        if not candidates.size:
            return [], []

        token_counts = self._get_token_counts(candidates.tolist())
        x = self._candidate_features(prompt, candidates, token_counts)
        standardized = (x - _FEATURE_MEAN) / _FEATURE_SCALE
        scores = standardized @ _LOGIT_COEF + _LOGIT_INTERCEPT

        if self.model_logit_blend != 0.0:
            context_score = self._model_context_score(model, prompt, candidates.tolist())
            if context_score is not None and np.all(np.isfinite(context_score)):
                spread = float(np.std(context_score))
                if spread > 1e-8:
                    lm_z = (context_score - float(np.mean(context_score))) / spread
                    scores = scores + self.model_logit_blend * np.clip(lm_z, -5.0, 5.0)

        # Estimate the number of distinct IDs in the next 128-token horizon
        # from 1,178 synthetic train-stream windows (mean 82.53). This avoids
        # assuming that the negative candidate pool is the same size as the
        # positive set.
        n_good = max(1, min(len(candidates), _GOOD_COUNT_PRIOR))
        good_idx = set(np.argsort(scores, kind="stable")[-n_good:].tolist())
        good_ids = set(candidates[list(good_idx)].tolist())
        good = [int(value) for value in candidates if int(value) in good_ids]
        bad = [int(value) for value in candidates if int(value) not in good_ids]
        return bad, good


# Name used by the official starter notebook.
YourSolution = TokenPredictorSolution
