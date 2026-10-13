"""Deterministic window sampling for explanation examples and held-out eval."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from routerinterp.artifacts import Corpus, ExpertWrites, SparseActivations
from routerinterp.explain.schema import (
    EvalWindow,
    Example,
    ExampleType,
    PipelineContext,
)
from routerinterp.explain.utils import decode_tokens
from routerinterp.select import FeatureScore

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

N_QUANTILES = 10  # Delphi quantile sampler bucket count


@dataclass(frozen=True, slots=True)
class QuantileSample:
    """Indices drawn across activation quantiles, and the quantile of each index."""

    indices: list[int]
    quantile_ids: list[int]


@dataclass(frozen=True, slots=True)
class WindowPool:
    """Windows and the per-position activations inside each one."""

    window_ids: np.ndarray
    activations: np.ndarray


@dataclass(frozen=True, slots=True)
class SampledExamples:
    """Train examples for one target, and the strongest activation among them."""

    examples: list[Example]
    max_activation: float


def entity_seed(seed: int, *parts: object) -> int:
    """Return a per-entity random seed that does not depend on processing order."""
    digest = hashlib.sha256(
        ":".join([str(seed), *(str(p) for p in parts)]).encode()
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def quantile_sample_indices(
    n_windows: int,
    n_quantiles: int,
    n_samples: int,
    seed: int,
) -> QuantileSample:
    """Pick indices spread across a list sorted from highest activation to lowest, and
    which quantile each index came from.

    Extra slots go to the earliest quantiles.
    If the list is shorter than the number of quantiles, the indices are drawn uniformly
    instead.
    """
    if n_windows < n_quantiles:
        rng = random.Random(seed)
        chosen = rng.sample(range(n_windows), min(n_samples, n_windows))
        return QuantileSample(chosen, [0] * len(chosen))
    rng = random.Random(seed)
    quantile_size = n_windows // n_quantiles
    base = n_samples // n_quantiles
    extra = n_samples % n_quantiles
    indices: list[int] = []
    quantile_ids: list[int] = []
    for quantile_id in range(n_quantiles):
        start = quantile_id * quantile_size
        bucket = list(range(start, start + quantile_size))
        samples_per_quantile = base + (1 if quantile_id < extra else 0)
        if len(bucket) < samples_per_quantile:
            chosen = bucket
        else:
            chosen = rng.sample(bucket, samples_per_quantile)
        indices.extend(chosen)
        quantile_ids.extend([quantile_id] * len(chosen))
    return QuantileSample(indices, quantile_ids)


def sample_from_top_pool(n_windows: int, n_sample: int, seed: int) -> list[int]:
    """Sample ``n_sample`` indices from the highest-activation part of the pool.

    That part is three times ``n_sample`` long, or the whole pool if the pool is
    shorter.
    """
    if n_windows <= 0 or n_sample <= 0:
        return []
    if n_sample >= n_windows:
        return list(range(n_windows))
    rng = random.Random(seed)
    n_top = max(n_sample, min(n_windows, n_sample * 3))
    return sorted(rng.sample(range(n_top), n_sample))


def window_budgets(
    *,
    n_features: int,
    train_positives: int,
    train_negatives: int,
    min_pos: int = 3,
    min_neg: int = 2,
) -> tuple[int, int]:
    """Return how many positive and negative windows each feature gets.

    The expert's totals are divided by the number of features.
    The result is at least 3 positive windows and, when any negatives were requested, 2
    negative windows.
    """
    if n_features <= 0:
        msg = f"n_features must be positive, got {n_features}"
        raise ValueError(msg)
    if train_positives <= 0:
        msg = f"train_positives must be > 0, got {train_positives}"
        raise ValueError(msg)
    if train_negatives < 0:
        msg = f"train_negatives must be >= 0, got {train_negatives}"
        raise ValueError(msg)
    pos = max(min_pos, train_positives // n_features)
    neg = max(min_neg, train_negatives // n_features) if train_negatives > 0 else 0
    return pos, neg


def pool_windows(
    indices: np.ndarray,
    values: np.ndarray,
    *,
    ctx_len: int,
) -> WindowPool:
    """Turn per-token activations into windows of length ``ctx_len``.

    ``indices`` must be corpus positions in increasing order.
    Windows come back with the highest peak first.
    """
    if indices.size == 0:
        return WindowPool(
            np.empty(0, dtype=np.int64),
            np.empty((0, ctx_len), dtype=np.float32),
        )
    window_of_token = indices // ctx_len
    pos_in_window = indices % ctx_len
    unique_windows, inverse = np.unique(window_of_token, return_inverse=True)
    acts = np.zeros((len(unique_windows), ctx_len), dtype=np.float32)
    acts[inverse, pos_in_window] = values
    max_per_window = acts.max(axis=1)
    order = np.argsort(-max_per_window, kind="stable")
    return WindowPool(unique_windows[order].astype(np.int64), acts[order])


def inactive_window_ids(
    corpus: Corpus,
    active_window_ids: np.ndarray,
    *,
    ctx_len: int,
    split: str,
) -> np.ndarray:
    """Return windows in ``split`` where this target never activates."""
    n_windows = corpus.n_windows(ctx_len)
    active_mask = np.zeros(n_windows, dtype=bool)
    if active_window_ids.size:
        active_mask[active_window_ids] = True
    all_windows = np.arange(n_windows, dtype=np.int64)
    train_mask = corpus.windows_are_train(all_windows, ctx_len)
    split_mask = train_mask if split == "train" else ~train_mask
    return all_windows[~active_mask & split_mask]


def sample_random_negative_windows(
    candidate_window_ids: np.ndarray,
    count: int,
    seed: int,
) -> list[int]:
    """Sample negative windows from a fixed seed, without replacement."""
    if count <= 0 or candidate_window_ids.size == 0:
        if count > 0:
            logger.warning("No candidate windows for random negatives")
        return []
    rng = random.Random(seed)
    n = min(count, int(candidate_window_ids.size))
    if n < count:
        logger.warning(
            f"Requested {count} random negatives but only {n} candidates available"
        )
    chosen = rng.sample(range(int(candidate_window_ids.size)), n)
    return [int(candidate_window_ids[i]) for i in chosen]


def _make_window(
    corpus: Corpus,
    window_id: int,
    acts: np.ndarray,
    *,
    example_type: ExampleType,
    ctx_len: int,
    split: str = "train",
    group_id: int | None = None,
    quantile: int | None = None,
) -> Example:
    tokens = corpus.window_tokens(int(window_id), ctx_len)
    return Example(
        tokens=[int(t) for t in tokens],
        activations=[float(a) for a in acts],
        window_id=int(window_id),
        example_type=example_type,
        split=split,
        group_id=group_id,
        quantile=quantile,
    )


def sample_feature_examples(
    ctx: PipelineContext,
    hookpoint: str,
    feature_id: int,
    indices: np.ndarray,
    values: np.ndarray,
) -> SampledExamples | None:
    """Sample the most active train windows for one SAE feature."""
    cfg = ctx.config
    pooled = pool_windows(indices, values, ctx_len=cfg.ctx_len)
    pool_ids = pooled.window_ids
    pool_acts = pooled.activations
    train_mask = ctx.corpus.windows_are_train(pool_ids, cfg.ctx_len)
    pool_ids, pool_acts = pool_ids[train_mask], pool_acts[train_mask]
    if pool_ids.size == 0:
        logger.warning(
            f"No train-split activations for feature {hookpoint}/{feature_id}"
        )
        return None
    max_activation = float(pool_acts[0].max())

    examples: list[Example] = []
    n_pos = min(cfg.train_positives, len(pool_ids))
    for i in range(n_pos):
        examples.append(
            _make_window(
                ctx.corpus,
                pool_ids[i],
                pool_acts[i],
                example_type="positive",
                ctx_len=cfg.ctx_len,
            )
        )
    return SampledExamples(examples, max_activation)


def sample_expert_examples(
    ctx: PipelineContext,
    hookpoint: str,
    expert_id: int,
    indices: np.ndarray,
    values: np.ndarray,
) -> SampledExamples | None:
    """Sample train windows where the expert is routed, spread across the activation range, plus
    random train windows where it never activates.

    The random windows are saved with the examples and are not put in the prompt.
    """
    cfg = ctx.config
    pooled = pool_windows(indices, values, ctx_len=cfg.ctx_len)
    pool_ids = pooled.window_ids
    pool_acts = pooled.activations
    train_mask = ctx.corpus.windows_are_train(pool_ids, cfg.ctx_len)
    pool_ids, pool_acts = pool_ids[train_mask], pool_acts[train_mask]
    if pool_ids.size == 0:
        logger.warning(f"No train-split activations for expert {hookpoint}/{expert_id}")
        return None
    max_activation = float(pool_acts[0].max())
    seed = entity_seed(cfg.example_seed, "expert", "train", hookpoint, expert_id)

    examples: list[Example] = []
    picked = quantile_sample_indices(
        len(pool_ids), N_QUANTILES, cfg.train_positives, seed
    )
    pos_idx = picked.indices
    quantile_ids = picked.quantile_ids
    for i, quantile_id in zip(pos_idx, quantile_ids, strict=True):
        examples.append(
            _make_window(
                ctx.corpus,
                pool_ids[i],
                pool_acts[i],
                example_type="positive",
                ctx_len=cfg.ctx_len,
                quantile=quantile_id,
            )
        )

    if cfg.train_negatives > 0:
        candidates = inactive_window_ids(
            ctx.corpus, pool_ids, ctx_len=cfg.ctx_len, split="train"
        )
        zero = np.zeros(cfg.ctx_len, dtype=np.float32)
        for window_id in sample_random_negative_windows(
            candidates, cfg.train_negatives, seed
        ):
            examples.append(
                _make_window(
                    ctx.corpus,
                    window_id,
                    zero,
                    example_type="random_negative",
                    ctx_len=cfg.ctx_len,
                )
            )
    return SampledExamples(examples, max_activation)


def sample_routerinterp_examples(
    ctx: PipelineContext,
    hookpoint: str,
    expert_id: int,
    selected: list[FeatureScore],
    feature_fires: dict[int, SparseActivations],
    route_mask: np.ndarray,
) -> list[Example]:
    """Sample examples for one expert, grouped by selected feature.

    A positive window comes from tokens where that feature activates and the expert is
    routed.
    A negative window comes from tokens where the same feature activates and the expert
    is not.
    """
    cfg = ctx.config
    pos_budget, neg_budget = window_budgets(
        n_features=len(selected),
        train_positives=cfg.train_positives,
        train_negatives=cfg.train_negatives,
    )
    base_seed = entity_seed(cfg.example_seed, "expert", "train", hookpoint, expert_id)
    is_train = ctx.corpus.is_train
    seq_len = ctx.corpus.seq_len
    examples: list[Example] = []
    for pick in selected:
        feature_id = pick.feature_id
        fires = feature_fires.get(
            feature_id,
            SparseActivations(
                np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
            ),
        )
        indices = fires.indices
        values = fires.values
        if indices.size == 0:
            logger.warning(
                f"routerinterp {hookpoint}/{expert_id}: feature {feature_id} "
                "has no SAE activations; skipping group"
            )
            continue
        train_mask = is_train[indices // seq_len]
        idx_tr, val_tr = indices[train_mask], values[train_mask]
        routed = route_mask[idx_tr]
        positive_pool = pool_windows(
            idx_tr[routed], val_tr[routed], ctx_len=cfg.ctx_len
        )
        negative_pool = pool_windows(
            idx_tr[~routed], val_tr[~routed], ctx_len=cfg.ctx_len
        )
        pos_ids = positive_pool.window_ids
        pos_acts = positive_pool.activations
        neg_ids = negative_pool.window_ids
        neg_acts = negative_pool.activations
        feature_seed = base_seed + feature_id
        pos_sel = sample_from_top_pool(len(pos_ids), pos_budget, feature_seed)
        neg_sel = sample_from_top_pool(len(neg_ids), neg_budget, feature_seed + 1)
        if not pos_sel or (neg_budget > 0 and not neg_sel):
            logger.warning(
                f"routerinterp {hookpoint}/{expert_id}: feature {feature_id} "
                f"has pos={len(pos_sel)} neg={len(neg_sel)}; skipping group"
            )
            continue
        positives = [
            _make_window(
                ctx.corpus,
                pos_ids[i],
                pos_acts[i],
                example_type="positive",
                ctx_len=cfg.ctx_len,
                group_id=feature_id,
            )
            for i in pos_sel
        ]
        negatives = [
            _make_window(
                ctx.corpus,
                neg_ids[i],
                neg_acts[i],
                example_type="hard_negative",
                ctx_len=cfg.ctx_len,
                group_id=feature_id,
            )
            for i in neg_sel
        ]
        examples.extend(positives)
        examples.extend(negatives)
    return examples


def sample_esb_examples(
    ctx: PipelineContext,
    hookpoint: str,
    expert_id: int,
    writes: ExpertWrites,
) -> SampledExamples | None:
    """Sample windows for the Expert Strikes Back method for one expert.

    Each position where the expert fired also carries the token ids the logit lens
    assigns to that write.
    """
    cfg = ctx.config
    sub = writes.for_expert(expert_id)
    if sub.token_indices.size == 0:
        logger.warning(f"No expert writes for expert {hookpoint}/{expert_id}")
        return None
    is_train = ctx.corpus.is_train
    seq_len = ctx.corpus.seq_len
    train_mask = is_train[sub.token_indices // seq_len]
    idx_tr = sub.token_indices[train_mask]
    scores_tr = (sub.gate_values * sub.write_norms)[train_mask]
    if idx_tr.size == 0:
        logger.warning(f"No train-split expert writes for {hookpoint}/{expert_id}")
        return None
    pooled = pool_windows(idx_tr, scores_tr, ctx_len=cfg.ctx_len)
    pool_ids = pooled.window_ids
    pool_acts = pooled.activations
    max_activation = float(pool_acts[0].max())

    # flat token position -> row in the (train-filtered) write arrays, for
    # promoted-token attachment.
    row_of_flat: dict[int, int] = {int(t): i for i, t in enumerate(idx_tr)}
    promoted_rows = (
        sub.promoted_token_ids[train_mask]
        if sub.promoted_token_ids is not None
        else None
    )

    examples: list[Example] = []
    n_pos = min(cfg.train_positives, len(pool_ids))
    for i in range(n_pos):
        window = _make_window(
            ctx.corpus,
            pool_ids[i],
            pool_acts[i],
            example_type="positive",
            ctx_len=cfg.ctx_len,
        )
        if promoted_rows is not None:
            promoted = np.full(
                (cfg.ctx_len, promoted_rows.shape[1]), -1, dtype=np.int32
            )
            base = window.window_id * cfg.ctx_len
            for pos in range(cfg.ctx_len):
                row = row_of_flat.get(base + pos)
                if row is not None:
                    promoted[pos] = promoted_rows[row]
            window.promoted_token_ids = promoted.tolist()
        examples.append(window)
    return SampledExamples(examples, max_activation)


def sample_negative_highlight_indices(
    positive_counts: list[int],
    ctx_len: int,
    n_negatives: int,
    seed: int,
) -> list[set[int]]:
    """Highlight positions for negative windows, with about as many highlights as the
    positive windows have.

    Each negative window takes a count from the positive windows' numbers of active
    tokens, then that many distinct positions.
    """
    counts = list(positive_counts)
    if not any(count > 0 for count in counts):
        counts = [1]
    rng = random.Random(seed)
    index_sets: list[set[int]] = []
    for _ in range(n_negatives):
        n_highlight = min(int(rng.choice(counts)), ctx_len)
        indices = (
            set(rng.sample(range(ctx_len), n_highlight)) if n_highlight > 0 else set()
        )
        index_sets.append(indices)
    return index_sets


def build_eval_windows(
    corpus: Corpus,
    tokenizer: PreTrainedTokenizerBase,
    indices: np.ndarray,
    values: np.ndarray,
    *,
    ctx_len: int,
    n_positives: int,
    n_negatives: int,
    seed: int,
    highlight_seed: int,
    exclude_window_ids: set[int],
) -> list[EvalWindow]:
    """Build held-out test windows for one target.
    Windows already used to write the explanation are left out.

    Positive windows are spread across the activation range.
    Negative windows are random test windows where the target never activates, and they
    come first.
    """
    pooled = pool_windows(indices, values, ctx_len=ctx_len)
    pool_ids = pooled.window_ids
    pool_acts = pooled.activations
    test_mask = ~corpus.windows_are_train(pool_ids, ctx_len)
    pool_ids, pool_acts = pool_ids[test_mask], pool_acts[test_mask]
    all_test_active = pool_ids  # before exclusion: defines "active" for negatives
    if exclude_window_ids:
        exclude = np.asarray(sorted(exclude_window_ids), dtype=np.int64)
        keep = ~np.isin(pool_ids, exclude)
        pool_ids, pool_acts = pool_ids[keep], pool_acts[keep]

    picked = quantile_sample_indices(len(pool_ids), N_QUANTILES, n_positives, seed)
    pos_idx = picked.indices
    quantile_ids = picked.quantile_ids
    if len(pos_idx) < n_positives:
        logger.warning(
            f"Held-out positive pool too small: {len(pos_idx)}/{n_positives}"
        )

    candidates = inactive_window_ids(
        corpus, all_test_active, ctx_len=ctx_len, split="test"
    )
    if exclude_window_ids:
        candidates = candidates[
            ~np.isin(candidates, np.asarray(sorted(exclude_window_ids), dtype=np.int64))
        ]
    neg_ids = sample_random_negative_windows(candidates, n_negatives, seed)

    # Leakage guard: eval windows must be disjoint from explain-time windows.
    eval_ids = {int(pool_ids[i]) for i in pos_idx} | set(neg_ids)
    overlap = eval_ids & exclude_window_ids
    if overlap:
        msg = (
            f"Eval/explain window overlap detected ({len(overlap)} windows, "
            f"e.g. {sorted(overlap)[:5]}); refusing to score"
        )
        raise AssertionError(msg)

    positives: list[EvalWindow] = []
    for i, quantile_id in zip(pos_idx, quantile_ids, strict=True):
        acts = pool_acts[i]
        token_ids = corpus.window_tokens(int(pool_ids[i]), ctx_len)
        str_tokens = decode_tokens(tokenizer, token_ids)
        highlight = {int(p) for p in np.nonzero(acts > 0)[0]}
        positives.append(
            EvalWindow(
                window_id=int(pool_ids[i]),
                tokens=[int(t) for t in token_ids],
                str_tokens=str_tokens,
                activations=[float(a) for a in acts],
                ground_truth=True,
                highlight=highlight,
                quantile=int(quantile_id),
            )
        )

    highlight_seed_sets = sample_negative_highlight_indices(
        [sum(a > 0 for a in w.activations) for w in positives],
        ctx_len,
        len(neg_ids),
        highlight_seed,
    )
    negatives: list[EvalWindow] = []
    zero = [0.0] * ctx_len
    for window_id, highlight in zip(neg_ids, highlight_seed_sets, strict=True):
        token_ids = corpus.window_tokens(window_id, ctx_len)
        str_tokens = decode_tokens(tokenizer, token_ids)
        negatives.append(
            EvalWindow(
                window_id=window_id,
                tokens=[int(t) for t in token_ids],
                str_tokens=str_tokens,
                activations=list(zero),
                ground_truth=False,
                highlight=highlight,
            )
        )
    return negatives + positives
