"""Predict which experts fire, from SAE latents, single residual coordinates, a PCA of the
residual, or token n-gram counts.

Each of those is fit on the train tokens of a finished collect run.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
from loguru import logger
from scipy import sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from routerinterp import core
from routerinterp.artifacts import Corpus, load_corpus
from routerinterp.core import NgramType

# --- collect.py shard contracts consumed here --------------------------------
RESIDUALS_KEY = "residuals"  # residual shards: dense [T, H] activations

PROBE_RESULTS_FILENAME = "probe_results.parquet"
NGRAM_DIR = "ngrams"
TEST_SPLIT = "test"

PROBE_RESULT_SCHEMA = pa.schema(
    [
        ("basis", pa.string()),  # sae | neuron | pca | ngram
        ("hookpoint", pa.string()),
        ("expert_id", pa.int64()),
        ("metric", pa.string()),  # auroc | ap | f1
        ("value", pa.float64()),
        ("split", pa.string()),  # test
        ("n_train", pa.int64()),  # train positives for this expert
        ("n_test", pa.int64()),  # test positives for this expert
    ]
)

# ---------------------------------------------------------------------------
# N-gram lookup tables
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NgramEntry:
    """Token ids of one n-gram and how often it was routed to an expert."""

    token_ids: tuple[int, ...]
    count: int


@dataclass(frozen=True, slots=True, order=True)
class ExpertAtHook:
    """One expert at one hookpoint."""

    hookpoint: str
    expert_id: int


type Bigram = tuple[int, int]
type NgramTable = dict[ExpertAtHook, list[NgramEntry]]
type BigramExpertCounts = dict[Bigram, Counter[int]]
ARTIFACT_VERSION = 1
PAD_ID = -1


@dataclass
class NgramLookupTables:
    """Compact-artifact tables used by explanation lookup / scoring."""

    primary: NgramTable
    unigrams: NgramTable
    global_selected_bigrams: set[Bigram] = field(default_factory=set)
    ngram_type: NgramType = "bigram"


def activation_position(ctx_len: int) -> int:
    """Return the index of the routed token inside a window of length ``ctx_len``."""
    return ctx_len - ctx_len // 4


def ngram_tuple_at_index(
    tokens: Sequence[int] | np.ndarray,
    index: int,
    ngram_type: NgramType,
) -> tuple[int, ...]:
    """Extract the unigram or bigram token-id tuple at ``index``."""
    token_id = int(tokens[index])
    if ngram_type == "unigram" or index == 0:
        return (token_id,)
    return (int(tokens[index - 1]), token_id)


def _top_k_unigram_matrices(
    unigram_expert_counts: dict[int, Counter[int]],
    num_experts: int,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    per_expert: dict[int, Counter[int]] = {e: Counter() for e in range(num_experts)}
    for token_id, expert_counter in unigram_expert_counts.items():
        for expert_id, count in expert_counter.items():
            per_expert[int(expert_id)][int(token_id)] = int(count)

    ids = np.full((num_experts, top_k), PAD_ID, dtype=np.int64)
    counts = np.zeros((num_experts, top_k), dtype=np.int64)
    for expert_id, counter in per_expert.items():
        for rank, (token_id, count) in enumerate(counter.most_common(top_k)):
            ids[expert_id, rank] = token_id
            counts[expert_id, rank] = count
    return ids, counts


def _top_k_bigram_matrices(
    bigram_expert_counts: BigramExpertCounts,
    num_experts: int,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    per_expert: dict[int, Counter[Bigram]] = {e: Counter() for e in range(num_experts)}
    for bigram, expert_counter in bigram_expert_counts.items():
        for expert_id, count in expert_counter.items():
            per_expert[int(expert_id)][bigram] = int(count)

    ids = np.full((num_experts, top_k, 2), PAD_ID, dtype=np.int64)
    counts = np.zeros((num_experts, top_k), dtype=np.int64)
    for expert_id, counter in per_expert.items():
        for rank, (bigram, count) in enumerate(counter.most_common(top_k)):
            ids[expert_id, rank, 0] = bigram[0]
            ids[expert_id, rank, 1] = bigram[1]
            counts[expert_id, rank] = count
    return ids, counts


def build_ngram_artifact(
    *,
    ngram_type: NgramType,
    num_experts: int,
    top_k: int,
    unigram_expert_counts: dict[int, Counter[int]],
    bigram_expert_counts: BigramExpertCounts,
    layer: int,
    hookpoint: str,
    model_name: str,
    vocab_size: int,
) -> dict[str, Any]:
    """Build arrays for a compact per-expert top-K n-gram ``.npz`` artifact."""
    uni_ids, uni_counts = _top_k_unigram_matrices(
        unigram_expert_counts, num_experts, top_k
    )
    payload: dict[str, Any] = {
        "version": ARTIFACT_VERSION,
        "ngram_type": np.asarray(ngram_type),
        "num_experts": np.asarray(num_experts),
        "top_k": np.asarray(top_k),
        "layer": np.asarray(layer),
        "hookpoint": np.asarray(hookpoint),
        "model_name": np.asarray(model_name),
        "vocab_size": np.asarray(vocab_size),
        "unigram_token_ids": uni_ids,
        "unigram_counts": uni_counts,
    }
    if ngram_type == "bigram":
        bi_ids, bi_counts = _top_k_bigram_matrices(
            bigram_expert_counts, num_experts, top_k
        )
        payload["bigram_token_ids"] = bi_ids
        payload["bigram_counts"] = bi_counts
    return payload


def load_ngram_artifact(path: Path) -> dict[str, Any]:
    """Load a compact n-gram ``.npz`` artifact."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def artifact_to_ngram_tables(
    payload: dict[str, Any],
    *,
    hookpoint: str,
    top_k: int | None = None,
) -> tuple[NgramTable, NgramTable, set[Bigram]]:
    """Build per-expert unigram and bigram tables, plus every bigram any expert saved."""
    saved_k = int(payload["top_k"])
    request_k = saved_k if top_k is None else top_k
    if request_k > saved_k:
        msg = (
            f"Requested top_k={request_k} exceeds saved top_k={saved_k} "
            "in n-gram artifact"
        )
        raise ValueError(msg)

    ngram_type = str(payload["ngram_type"])
    num_experts = int(payload["num_experts"])
    uni_ids = payload["unigram_token_ids"]
    uni_counts = payload["unigram_counts"]

    unigram_table: NgramTable = {}
    for expert_id in range(num_experts):
        entries: list[NgramEntry] = []
        for rank in range(request_k):
            tok = int(uni_ids[expert_id, rank])
            if tok == PAD_ID:
                break
            entries.append(NgramEntry((tok,), int(uni_counts[expert_id, rank])))
        unigram_table[ExpertAtHook(hookpoint, expert_id)] = entries

    global_bigrams: set[Bigram] = set()
    if ngram_type == "unigram":
        return unigram_table, unigram_table, global_bigrams

    bi_ids = payload["bigram_token_ids"]
    bi_counts = payload["bigram_counts"]
    bigram_table: NgramTable = {}
    for expert_id in range(num_experts):
        entries = []
        for rank in range(request_k):
            a = int(bi_ids[expert_id, rank, 0])
            b = int(bi_ids[expert_id, rank, 1])
            if a == PAD_ID or b == PAD_ID:
                break
            pair: Bigram = (a, b)
            entries.append(NgramEntry(pair, int(bi_counts[expert_id, rank])))
            global_bigrams.add(pair)
        bigram_table[ExpertAtHook(hookpoint, expert_id)] = entries
    return bigram_table, unigram_table, global_bigrams


def ngram_set_for_expert(
    ngrams: NgramTable,
    hookpoint: str,
    expert_id: int,
    top_k: int,
) -> set[tuple[int, ...]]:
    """Return one expert's top-K n-gram token ids."""
    entries = ngrams.get(ExpertAtHook(hookpoint, expert_id), [])
    return {entry.token_ids for entry in entries[:top_k]}


def lookup_matches_expert(
    tokens: Sequence[int] | np.ndarray,
    act_offset: int,
    ngram_type: NgramType,
    expert_ngrams: set[tuple[int, ...]],
    expert_unigrams: set[tuple[int, ...]] | None = None,
    *,
    global_selected_bigrams: set[Bigram] | None = None,
) -> bool:
    """Return whether this token's context is in the expert's n-gram table.

    A bigram that appears in the saved bigram tables is not also checked as a unigram.
    A bigram that does not appear there is checked as the current token alone.
    """
    if not expert_ngrams and not expert_unigrams:
        return False

    if ngram_type == "unigram":
        return ngram_tuple_at_index(tokens, act_offset, "unigram") in expert_ngrams

    primary = ngram_tuple_at_index(tokens, act_offset, "bigram")
    if len(primary) == 2:
        pair = (primary[0], primary[1])
        if global_selected_bigrams is not None and pair in global_selected_bigrams:
            return primary in expert_ngrams
        if primary in expert_ngrams:
            return True
    elif primary in expert_ngrams:
        return True

    if expert_unigrams:
        unigram = ngram_tuple_at_index(tokens, act_offset, "unigram")
        return unigram in expert_unigrams
    return False


# ---------------------------------------------------------------------------
# Corpus / shard loading
# ---------------------------------------------------------------------------


@dataclass
class ProbeSample:
    """The collect corpus, and the token positions this probe stage scores.

    Those positions count from the start of the corpus.
    """

    corpus: Corpus
    train_mask_T: np.ndarray  # bool [N * S]
    selected: np.ndarray  # int64 [n] sorted flat token indices

    @property
    def tokens(self) -> np.ndarray:
        return self.corpus.tokens

    @property
    def seq_len(self) -> int:
        return self.corpus.seq_len

    @property
    def n_tokens(self) -> int:
        return self.corpus.n_tokens


def _shard_paths(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("shard-*.safetensors"))
    if not paths:
        msg = f"No shard-*.safetensors under {directory}"
        raise FileNotFoundError(msg)
    return paths


def _shard_row_selection(
    shard_sizes: list[int], selected: np.ndarray
) -> list[np.ndarray]:
    """Find which rows of each shard hold the chosen corpus positions."""
    starts = np.concatenate([[0], np.cumsum(shard_sizes)])
    out: list[np.ndarray] = []
    for i in range(len(shard_sizes)):
        lo, hi = starts[i], starts[i + 1]
        mask = (selected >= lo) & (selected < hi)
        out.append(selected[mask] - lo)
    return out


def _probe_sample(corpus: Corpus, config: core.ProbeConfig) -> ProbeSample:
    """Build the train mask and the seeded token subsample shared by every probe."""
    train_mask_T = np.repeat(corpus.is_train, corpus.seq_len)
    total = corpus.n_tokens
    if total > config.n_tokens:
        rng = np.random.default_rng(config.seed)
        selected = np.sort(
            rng.choice(total, size=config.n_tokens, replace=False)
        ).astype(np.int64)
    else:
        selected = np.arange(total, dtype=np.int64)
    return ProbeSample(corpus=corpus, train_mask_T=train_mask_T, selected=selected)


def _load_labels(
    input_dir: Path, layer: int, sample: ProbeSample
) -> tuple[np.ndarray, int, int]:
    """Return, for each sampled token, whether each expert was in the router's top-k."""
    router_dir = input_dir / "router" / f"layers_{layer}"
    parts: list[np.ndarray] = []
    num_experts = 0
    for path in _shard_paths(router_dir):
        tensors = core.read_safetensors(path)
        topk_ids = np.asarray(tensors[core.ROUTER_TOPK_IDS], dtype=np.int64)
        parts.append(topk_ids)
        if topk_ids.size:
            num_experts = max(num_experts, int(topk_ids.max()) + 1)
    topk = np.concatenate(parts, axis=0)
    if topk.shape[0] != sample.n_tokens:
        msg = (
            f"Token/router misalignment at layer {layer}: "
            f"tokens={sample.n_tokens:,} vs router={topk.shape[0]:,}"
        )
        raise ValueError(msg)

    sel_topk = topk[sample.selected]
    top_k = int(sel_topk.shape[1])
    labels = np.zeros((sel_topk.shape[0], num_experts), dtype=np.bool_)
    np.put_along_axis(labels, sel_topk, True, axis=1)
    return labels, top_k, num_experts


def _load_latents(
    input_dir: Path, hookpoint: str, sample: ProbeSample
) -> sp.csr_matrix:
    """Load SAE latents for the selected token rows."""
    latents_dir = input_dir / "latents" / core.hookpoint_slug(hookpoint)
    paths = core.csr_shard_paths(latents_dir)
    if not paths:
        msg = f"No CSR latent shards under {latents_dir}"
        raise FileNotFoundError(msg)
    shards = [core.read_csr_shard(path) for path in paths]
    local_rows = _shard_row_selection(
        [shard.n_tokens for shard in shards], sample.selected
    )

    num_features = 0
    for shard in shards:
        if shard.nnz:
            num_features = max(num_features, int(shard.feature_ids.max()) + 1)

    indptr_parts = [np.zeros(1, dtype=np.int64)]
    ids_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    nnz = 0
    for shard, rows in zip(shards, local_rows, strict=True):
        counts = shard.offsets[rows + 1] - shard.offsets[rows]
        sub_indptr = np.zeros(rows.shape[0] + 1, dtype=np.int64)
        np.cumsum(counts, out=sub_indptr[1:])
        total = int(sub_indptr[-1])
        if total:
            starts = np.repeat(shard.offsets[rows] - sub_indptr[:-1], counts)
            gather = starts + np.arange(total, dtype=np.int64)
            ids_parts.append(shard.feature_ids[gather])
            val_parts.append(shard.values[gather])
        indptr_parts.append(sub_indptr[1:] + nnz)
        nnz += total

    indptr = np.concatenate(indptr_parts)
    indices = np.concatenate(ids_parts) if ids_parts else np.empty(0, dtype=np.int32)
    data = (
        np.concatenate(val_parts).astype(np.float32)
        if val_parts
        else np.empty(0, dtype=np.float32)
    )
    return sp.csr_matrix(
        (data, indices, indptr),
        shape=(sample.selected.shape[0], num_features),
    )


def _load_residuals(input_dir: Path, hookpoint: str, sample: ProbeSample) -> np.ndarray:
    """Load residual rows for the selected tokens, as float32 ``[n_selected, H]``."""
    residuals_dir = input_dir / "residuals" / core.hookpoint_slug(hookpoint)
    arrays: list[np.ndarray] = []
    for path in _shard_paths(residuals_dir):
        tensors = core.read_safetensors(path)
        if RESIDUALS_KEY not in tensors:
            msg = (
                f"Residual shard {path} missing key {RESIDUALS_KEY!r}; "
                f"found {sorted(tensors)}"
            )
            raise KeyError(msg)
        arrays.append(np.asarray(tensors[RESIDUALS_KEY]))
    local_rows = _shard_row_selection([arr.shape[0] for arr in arrays], sample.selected)
    return np.concatenate(
        [
            arr[rows].astype(np.float32)
            for arr, rows in zip(arrays, local_rows, strict=True)
        ],
        axis=0,
    )


# ---------------------------------------------------------------------------
# Metrics (F1, plus score-based AUROC and average precision)
# ---------------------------------------------------------------------------


def _f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute F1 as ``2tp / (2tp + fp + fn)``, or 0 when the denominator is 0."""
    tp = float(np.sum(y_true & y_pred))
    fp = float(np.sum(~y_true & y_pred))
    fn = float(np.sum(y_true & ~y_pred))
    denom = 2 * tp + fp + fn
    return (2 * tp) / denom if denom else 0.0


def _expert_metric_rows(
    *,
    basis: str,
    hookpoint: str,
    expert_id: int,
    y_true: np.ndarray,
    scores: np.ndarray,
    y_pred: np.ndarray,
    n_train_pos: int,
    n_test_pos: int,
) -> list[dict[str, Any]]:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)

    def _row(metric: str, value: float) -> dict[str, Any]:
        return {
            "basis": basis,
            "hookpoint": hookpoint,
            "expert_id": expert_id,
            "metric": metric,
            "value": float(value),
            "split": TEST_SPLIT,
            "n_train": n_train_pos,
            "n_test": n_test_pos,
        }

    rows: list[dict[str, Any]] = []
    n_pos = int(y_true.sum())
    if 0 < n_pos < y_true.shape[0]:
        rows.append(_row("auroc", float(roc_auc_score(y_true, scores))))
        rows.append(_row("ap", average_precision_score(y_true, scores)))
    else:
        logger.warning(
            f"{basis}/{hookpoint} expert {expert_id}: test split single-class "
            f"({n_pos} positives); skipping auroc/ap"
        )
    rows.append(_row("f1", _f1_score(y_true, y_pred)))
    return rows


# ---------------------------------------------------------------------------
# Shared linear probe (sae / neuron / pca bases)
# ---------------------------------------------------------------------------


def _run_linear_basis(
    *,
    basis: str,
    hookpoint: str,
    X_train: sp.csr_matrix | np.ndarray,
    X_test: sp.csr_matrix | np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    config: core.ProbeConfig,
    coverage: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Fit one logistic regression per expert on a shared design matrix."""
    n_train_tokens, n_experts = Y_train.shape
    rows: list[dict[str, Any]] = []
    n_skipped = 0
    for expert_id in range(n_experts):
        y_train = np.asarray(Y_train[:, expert_id]).ravel()
        n_pos = int(y_train.sum())
        if n_pos < config.min_tokens_per_expert or n_pos == n_train_tokens:
            n_skipped += 1
            continue
        clf = LogisticRegression(
            solver=config.solver,
            C=config.logreg_c,
            max_iter=config.max_iter,
            class_weight="balanced",
        )
        clf.fit(X_train, y_train)
        scores = clf.decision_function(X_test)
        preds = np.asarray(clf.predict(X_test)).ravel()
        rows.extend(
            _expert_metric_rows(
                basis=basis,
                hookpoint=hookpoint,
                expert_id=expert_id,
                y_true=Y_test[:, expert_id],
                scores=scores,
                y_pred=preds,
                n_train_pos=n_pos,
                n_test_pos=int(Y_test[:, expert_id].sum()),
            )
        )
        if coverage is not None:
            masked = np.asarray(preds).astype(bool) & np.asarray(coverage).astype(bool)
            rows.append(
                {
                    "basis": basis,
                    "hookpoint": hookpoint,
                    "expert_id": expert_id,
                    "metric": "effective_f1",
                    "value": _f1_score(Y_test[:, expert_id], masked),
                    "split": TEST_SPLIT,
                    "n_train": n_pos,
                    "n_test": int(Y_test[:, expert_id].sum()),
                }
            )
    logger.info(
        f"{basis}/{hookpoint}: probed {n_experts - n_skipped}/{n_experts} "
        f"experts (skipped {n_skipped} with < {config.min_tokens_per_expert} "
        "train positives)"
    )
    return rows


def keep_topk_features(features_TF: np.ndarray, k: int) -> np.ndarray:
    """Keep the ``k`` largest-magnitude values on each token and set the rest to zero.

    Signs stay as they are.
    A ``k`` at least as wide as the vector returns the vector unchanged.
    """
    n_features = features_TF.shape[1]
    if k >= n_features:
        return features_TF
    if k <= 0:
        return np.zeros_like(features_TF)
    # Last k columns of the partition are the largest magnitudes, unsorted.
    top = np.argpartition(np.abs(features_TF), n_features - k, axis=1)[:, -k:]
    filtered = np.zeros_like(features_TF)
    np.put_along_axis(
        filtered, top, np.take_along_axis(features_TF, top, axis=1), axis=1
    )
    return filtered


def _resolve_keep_top_s(
    config: core.ProbeConfig, resolved_config: dict[str, Any]
) -> int | None:
    """Choose how many activations to keep per token."""
    if config.keep_top_s is not None:
        return int(config.keep_top_s)
    sae = resolved_config.get("sae")
    if not isinstance(sae, dict):
        return None
    k = sae.get("k")
    return None if k is None else int(k)


def _fit_pca_on_train(
    train_TH: np.ndarray,
    test_TH: np.ndarray,
    pca_dim: int | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit PCA on training tokens only, then transform train and test."""
    from sklearn.decomposition import PCA

    hidden_size = train_TH.shape[1]
    effective_dim = min(pca_dim or hidden_size, hidden_size)
    if pca_dim is not None and effective_dim < pca_dim:
        logger.warning(
            f"Requested pca_dim={pca_dim} but hidden_size is {hidden_size}; "
            f"using {effective_dim} components"
        )
    pca = PCA(n_components=effective_dim, svd_solver="covariance_eigh")
    train_TK = pca.fit_transform(train_TH).astype(np.float32, copy=False)
    test_TK = pca.transform(test_TH).astype(np.float32, copy=False)
    return train_TK, test_TK, effective_dim


# ---------------------------------------------------------------------------
# N-gram basis (count lookup with hierarchical backoff)
# ---------------------------------------------------------------------------


@dataclass
class NgramRoutingModel:
    """Fitted unigram/bigram expert-activation lookup."""

    ngram_type: NgramType
    num_experts: int
    top_k: int
    vocab_size: int
    seq_len: int
    global_probs_E: np.ndarray  # [E] float32
    unigram_probs_VE: np.ndarray  # [V, E] float32 (0 rows where unseen)
    unigram_counts_V: np.ndarray  # [V] int64
    bigram_keys: np.ndarray  # sorted int64 [B], key = prev * V + cur
    bigram_probs_BE: np.ndarray  # [B, E] float32
    unigram_expert_counts: dict[int, Counter[int]]
    bigram_expert_counts: BigramExpertCounts


def _fit_ngram_model(
    tokens_NS: np.ndarray,
    y_train: np.ndarray,
    flat_indices: np.ndarray,
    *,
    ngram_type: NgramType,
    num_experts: int,
    top_k: int,
    seq_len: int,
    vocab_size: int,
) -> NgramRoutingModel:
    """Count n-grams on the chosen train positions, without crossing from one sequence into
    the next.
    """
    y = np.asarray(y_train, dtype=np.float32)
    n_train = int(flat_indices.shape[0])
    seq = flat_indices // seq_len
    pos = flat_indices % seq_len
    cur = tokens_NS[seq, pos].astype(np.int64)

    global_probs_E = y.sum(axis=0) / max(n_train, 1)

    unigram_sums = np.zeros((vocab_size, num_experts), dtype=np.float32)
    np.add.at(unigram_sums, cur, y)
    unigram_counts_V = np.bincount(cur, minlength=vocab_size).astype(np.int64)
    unigram_probs_VE = np.zeros_like(unigram_sums)
    np.divide(
        unigram_sums,
        np.maximum(unigram_counts_V, 1)[:, None],
        out=unigram_probs_VE,
    )

    unigram_expert_counts: dict[int, Counter[int]] = {}
    for tok in np.nonzero(unigram_counts_V)[0]:
        row = unigram_sums[tok]
        unigram_expert_counts[int(tok)] = Counter(
            {e: int(c) for e, c in enumerate(row) if c > 0}
        )

    bigram_keys = np.empty(0, dtype=np.int64)
    bigram_probs_BE = np.zeros((0, num_experts), dtype=np.float32)
    bigram_expert_counts: BigramExpertCounts = {}
    if ngram_type == "bigram":
        valid = pos > 0
        prev = tokens_NS[seq[valid], pos[valid] - 1].astype(np.int64)
        keys = prev * vocab_size + cur[valid]
        uniq, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        sums = np.zeros((uniq.shape[0], num_experts), dtype=np.float32)
        np.add.at(sums, inverse, y[valid])
        bigram_keys = uniq  # np.unique returns sorted keys
        bigram_probs_BE = sums / np.maximum(counts, 1)[:, None]
        for i, key in enumerate(uniq):
            a, b = divmod(int(key), vocab_size)
            row = sums[i]
            bigram_expert_counts[(a, b)] = Counter(
                {e: int(c) for e, c in enumerate(row) if c > 0}
            )

    logger.info(
        f"n-gram fit ({ngram_type}): {n_train:,} train tokens, "
        f"{int((unigram_counts_V > 0).sum()):,} unigrams, "
        f"{bigram_keys.shape[0]:,} bigrams"
    )
    return NgramRoutingModel(
        ngram_type=ngram_type,
        num_experts=num_experts,
        top_k=top_k,
        vocab_size=vocab_size,
        seq_len=seq_len,
        global_probs_E=global_probs_E.astype(np.float32),
        unigram_probs_VE=unigram_probs_VE,
        unigram_counts_V=unigram_counts_V,
        bigram_keys=bigram_keys,
        bigram_probs_BE=bigram_probs_BE.astype(np.float32),
        unigram_expert_counts=unigram_expert_counts,
        bigram_expert_counts=bigram_expert_counts,
    )


def _ngram_scores(
    model: NgramRoutingModel, tokens_NS: np.ndarray, flat_indices: np.ndarray
) -> np.ndarray:
    """Score every expert at each token.

    A token pair that was seen in training uses the bigram table.
    Otherwise the current token uses the unigram table.
    Otherwise the score is that expert's overall rate.
    """
    n = int(flat_indices.shape[0])
    seq = flat_indices // model.seq_len
    pos = flat_indices % model.seq_len
    cur = tokens_NS[seq, pos].astype(np.int64)

    scores = np.repeat(model.global_probs_E[None, :], n, axis=0)
    seen_unigram = model.unigram_counts_V[cur] > 0
    scores[seen_unigram] = model.unigram_probs_VE[cur[seen_unigram]]

    if model.ngram_type == "bigram" and model.bigram_keys.size:
        valid = pos > 0
        prev = tokens_NS[seq[valid], pos[valid] - 1].astype(np.int64)
        keys = prev * model.vocab_size + cur[valid]
        loc = np.searchsorted(model.bigram_keys, keys)
        loc = np.clip(loc, 0, model.bigram_keys.shape[0] - 1)
        found = model.bigram_keys[loc] == keys
        rows = np.nonzero(valid)[0]
        scores[rows[found]] = model.bigram_probs_BE[loc[found]]
    return scores


def _topk_predictions(scores: np.ndarray, top_k: int) -> np.ndarray:
    """Mark the ``top_k`` highest-scoring experts at each token."""
    n, num_experts = scores.shape
    k = min(top_k, num_experts)
    preds = np.zeros((n, num_experts), dtype=bool)
    top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    np.put_along_axis(preds, top, True, axis=1)
    return preds


def _run_ngram_basis(
    *,
    hookpoint: str,
    layer: int,
    sample: ProbeSample,
    labels: np.ndarray,
    top_k: int,
    num_experts: int,
    config: core.ProbeConfig,
    model_name: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    sel = sample.selected
    train_sel = sample.train_mask_T[sel]
    train_pos = np.nonzero(train_sel)[0]
    test_pos = np.nonzero(~train_sel)[0]
    vocab_size = int(sample.tokens.max()) + 1 if sample.tokens.size else 0

    model = _fit_ngram_model(
        sample.tokens,
        labels[train_pos],
        sel[train_pos],
        ngram_type=config.ngram_type,
        num_experts=num_experts,
        top_k=top_k,
        seq_len=sample.seq_len,
        vocab_size=vocab_size,
    )
    scores = _ngram_scores(model, sample.tokens, sel[test_pos])
    preds = _topk_predictions(scores, top_k)
    y_test = labels[test_pos]

    rows: list[dict[str, Any]] = []
    n_skipped = 0
    n_train_tokens = int(train_pos.shape[0])
    for expert_id in range(num_experts):
        n_pos = int(labels[train_pos, expert_id].sum())
        if n_pos < config.min_tokens_per_expert or n_pos == n_train_tokens:
            n_skipped += 1
            continue
        rows.extend(
            _expert_metric_rows(
                basis="ngram",
                hookpoint=hookpoint,
                expert_id=expert_id,
                y_true=y_test[:, expert_id],
                scores=scores[:, expert_id],
                y_pred=preds[:, expert_id],
                n_train_pos=n_pos,
                n_test_pos=int(y_test[:, expert_id].sum()),
            )
        )
    logger.info(
        f"ngram/{hookpoint}: probed {num_experts - n_skipped}/{num_experts} "
        f"experts (skipped {n_skipped} with < {config.min_tokens_per_expert} "
        "train positives)"
    )

    payload = build_ngram_artifact(
        ngram_type=config.ngram_type,
        num_experts=num_experts,
        top_k=config.ngram_artifact_top_k,
        unigram_expert_counts=model.unigram_expert_counts,
        bigram_expert_counts=model.bigram_expert_counts,
        layer=layer,
        hookpoint=hookpoint,
        model_name=model_name,
        vocab_size=vocab_size,
    )
    artifact_path = output_dir / NGRAM_DIR / f"{config.ngram_type}_layer_{layer}.npz"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(artifact_path, **payload)
    logger.info(f"Saved n-gram lookup artifact -> {artifact_path}")
    return rows


# ---------------------------------------------------------------------------
# Stage entrypoint
# ---------------------------------------------------------------------------


def _run_basis(
    basis: str,
    *,
    hookpoint: str,
    layer: int,
    sample: ProbeSample,
    labels: np.ndarray,
    top_k: int,
    num_experts: int,
    config: core.ProbeConfig,
    model_name: str,
    output_dir: Path,
    keep_top_s: int | None,
) -> list[dict[str, Any]]:
    if basis == "ngram":
        return _run_ngram_basis(
            hookpoint=hookpoint,
            layer=layer,
            sample=sample,
            labels=labels,
            top_k=top_k,
            num_experts=num_experts,
            config=config,
            model_name=model_name,
            output_dir=output_dir,
        )

    if basis == "sae":
        X = _load_latents(config.input_dir, hookpoint, sample)
    elif basis in ("neuron", "pca"):
        X = _load_residuals(config.input_dir, hookpoint, sample)
    else:
        msg = f"Unknown basis {basis!r}"
        raise ValueError(msg)

    sel = sample.selected
    train_sel = sample.train_mask_T[sel]
    Y_sel = labels
    X_train, X_test = X[train_sel], X[~train_sel]
    Y_train, Y_test = Y_sel[train_sel], Y_sel[~train_sel]
    if basis == "pca":
        X_train, X_test, effective_dim = _fit_pca_on_train(
            np.asarray(X_train), np.asarray(X_test), config.pca_dim
        )
        logger.info(f"PCA {hookpoint}: {effective_dim} components (fit on train)")
    # SAE latents are already CSR top-k from collect. Neuron top-s is on the
    # residuals; PCA top-s is on the coordinates after the basis change.
    if basis in ("neuron", "pca") and keep_top_s is not None:
        logger.info(
            f"{basis}/{hookpoint}: keeping top-{keep_top_s} activations per token"
        )
        X_train = keep_topk_features(np.asarray(X_train), keep_top_s)
        X_test = keep_topk_features(np.asarray(X_test), keep_top_s)
    return _run_linear_basis(
        basis=basis,
        hookpoint=hookpoint,
        X_train=X_train,
        X_test=X_test,
        Y_train=Y_train,
        Y_test=Y_test,
        config=config,
    )


def run_probes(config: core.ProbeConfig) -> Path:
    """Fit every configured input and write the probe outputs.

    The run is skipped when the output directory already has a manifest for this config
    and the collect run it reads.
    """
    input_dir = Path(config.input_dir)
    output_dir = Path(config.output_dir)
    collect_manifest = core.read_manifest(input_dir)
    parents = {"collect": collect_manifest.config_hash}
    if core.manifest_matches(output_dir, config, parents):
        logger.info(f"probe outputs up-to-date in {output_dir}; skipping")
        return output_dir

    model_cfg = collect_manifest.resolved_config.get("model") or {}
    model_name = str(model_cfg.get("name") or "unknown")
    keep_top_s = _resolve_keep_top_s(config, collect_manifest.resolved_config)
    if any(basis in ("neuron", "pca") for basis in config.bases):
        if keep_top_s is None:
            logger.info(
                "Neuron/PCA features left dense: keep_top_s unset and parent "
                "collect config has no sae.k"
            )
        elif config.keep_top_s is not None:
            logger.info(
                f"Neuron/PCA matched sparsity: top-{keep_top_s} activations per token"
            )
        else:
            logger.info(
                f"Neuron/PCA matched sparsity: top-{keep_top_s} activations "
                "per token (parent collect sae.k)"
            )
    sample = _probe_sample(load_corpus(input_dir, collect_manifest), config)
    logger.info(
        f"Corpus: {sample.tokens.shape[0]:,} docs x {sample.seq_len} tokens; "
        f"using {sample.selected.shape[0]:,} tokens "
        f"({int(sample.train_mask_T[sample.selected].sum()):,} train)"
    )

    rows: list[dict[str, Any]] = []
    for hookpoint in config.hookpoints:
        # Labels come from the paired router layer (resid_post at layer N
        # predicts router N+1).
        layer = core.router_layer_for_hookpoint(hookpoint)
        labels, top_k, num_experts = _load_labels(input_dir, layer, sample)
        for basis in config.bases:
            logger.info(f"Probe basis={basis} hookpoint={hookpoint}")
            rows.extend(
                _run_basis(
                    basis,
                    hookpoint=hookpoint,
                    layer=layer,
                    sample=sample,
                    labels=labels,
                    top_k=top_k,
                    num_experts=num_experts,
                    config=config,
                    model_name=model_name,
                    output_dir=output_dir,
                    keep_top_s=keep_top_s,
                )
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = core.write_parquet(
        output_dir / PROBE_RESULTS_FILENAME,
        pa.Table.from_pylist(rows, schema=PROBE_RESULT_SCHEMA),
    )
    manifest = core.build_manifest(
        stage="probe",
        run_name=f"probe-{core.canonical_config_hash(config)[:8]}",
        config=config,
        parents=parents,
        dataset_revision=collect_manifest.dataset_revision,
        model_revision=collect_manifest.model_revision,
        sae_revision=collect_manifest.sae_revision,
    )
    core.write_manifest(output_dir, manifest)
    logger.info(f"Probe stage complete: {len(rows)} rows -> {results_path}")
    return output_dir
