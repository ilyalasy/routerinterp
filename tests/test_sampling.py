"""Window sampling, including random and hard negatives."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from routerinterp.artifacts import Corpus, SparseActivations
from routerinterp.core import ExplainConfig
from routerinterp.explain.sampling import (
    build_eval_windows,
    pool_windows,
    quantile_sample_indices,
    sample_expert_examples,
    sample_feature_examples,
    sample_from_top_pool,
    sample_routerinterp_examples,
    window_budgets,
)
from routerinterp.explain.schema import PipelineContext
from routerinterp.select import FeatureScore, Selection


def _corpus(n_seq: int, n_train: int, seq_len: int = 4) -> Corpus:
    tokens = np.arange(n_seq * seq_len, dtype=np.int32).reshape(n_seq, seq_len)
    is_train = np.zeros(n_seq, dtype=bool)
    is_train[:n_train] = True
    return Corpus(tokens=tokens, doc_ids=np.arange(n_seq), is_train=is_train)


class _Tokenizer:
    def batch_decode(
        self, ids: list[int], skip_special_tokens: bool = True
    ) -> list[str]:
        return [f"t{token}" for token in ids]


def _ctx(corpus: Corpus, **overrides: object) -> PipelineContext:
    config = ExplainConfig(
        method="expert_activation",
        input_dir=Path("in"),
        selection_dir=Path("sel"),
        output_dir=Path("out"),
        hookpoints=["layers.0.mlp"],
        explainer_model="toy",
        train_positives=int(overrides.get("train_positives", 3)),
        train_negatives=int(overrides.get("train_negatives", 2)),
        ctx_len=int(overrides.get("ctx_len", corpus.seq_len)),
        example_seed=int(overrides.get("example_seed", 0)),
    )
    return PipelineContext(
        config=config,
        corpus=corpus,
        selection=Selection(method="cosine", n=1, path=Path("sel.parquet"), records=[]),
        tokenizer=_Tokenizer(),  # type: ignore[arg-type]
    )


def test_pool_and_sample_helpers() -> None:
    pooled = pool_windows(
        np.array([1, 4, 5], dtype=np.int64),
        np.array([1.0, 3.0, 9.0], dtype=np.float32),
        ctx_len=4,
    )
    assert pooled.window_ids.tolist() == [1, 0]
    assert pooled.activations[0].tolist() == [3.0, 9.0, 0.0, 0.0]
    assert pooled.activations[1].tolist() == [0.0, 1.0, 0.0, 0.0]
    empty = pool_windows(
        np.array([], dtype=np.int64), np.array([], dtype=np.float32), ctx_len=4
    )
    assert empty.window_ids.shape == (0,)
    assert empty.activations.shape == (0, 4)

    picked = quantile_sample_indices(20, 4, 6, seed=0)
    assert Counter(picked.quantile_ids) == {0: 2, 1: 2, 2: 1, 3: 1}
    for index, quantile in zip(picked.indices, picked.quantile_ids, strict=True):
        assert quantile * 5 <= index < (quantile + 1) * 5
    small = quantile_sample_indices(3, 10, 5, seed=1)
    assert set(small.indices) == {0, 1, 2}
    assert small.quantile_ids == [0, 0, 0]

    assert sample_from_top_pool(0, 3, 0) == []
    assert sample_from_top_pool(5, 0, 0) == []
    assert sample_from_top_pool(4, 10, 0) == [0, 1, 2, 3]
    top = sample_from_top_pool(100, 2, seed=0)
    assert len(top) == 2 and top == sorted(top)
    assert all(index < 6 for index in top)

    assert window_budgets(n_features=4, train_positives=10, train_negatives=6) == (3, 2)
    assert window_budgets(n_features=2, train_positives=20, train_negatives=0) == (
        10,
        0,
    )
    with pytest.raises(ValueError, match="n_features"):
        window_budgets(n_features=0, train_positives=4, train_negatives=1)


def test_feature_examples_ignore_the_test_split() -> None:
    ctx = _ctx(_corpus(8, 4), train_positives=2, train_negatives=0)
    sampled = sample_feature_examples(
        ctx,
        "layers.0.mlp",
        7,
        np.array([0, 9, 16], dtype=np.int64),
        np.array([1.0, 4.0, 100.0], dtype=np.float32),
    )
    assert sampled is not None
    examples = sampled.examples
    assert sampled.max_activation == 4.0
    assert [(ex.window_id, ex.activations) for ex in examples] == [
        (2, [0.0, 4.0, 0.0, 0.0]),
        (0, [1.0, 0.0, 0.0, 0.0]),
    ]
    assert all(ex.example_type == "positive" and ex.split == "train" for ex in examples)

    assert (
        sample_feature_examples(
            ctx,
            "layers.0.mlp",
            7,
            np.array([16], dtype=np.int64),
            np.array([100.0], dtype=np.float32),
        )
        is None
    )


def test_expert_examples_draw_random_negatives_from_inactive_train_windows() -> None:
    ctx = _ctx(_corpus(8, 4), train_positives=3, train_negatives=4)
    sampled = sample_expert_examples(
        ctx,
        "layers.0.mlp",
        1,
        np.array([0, 5, 10, 16], dtype=np.int64),
        np.array([1.0, 3.0, 5.0, 100.0], dtype=np.float32),
    )
    assert sampled is not None
    examples = sampled.examples
    assert sampled.max_activation == 5.0
    positives = [ex for ex in examples if ex.example_type == "positive"]
    negatives = [ex for ex in examples if ex.example_type == "random_negative"]
    assert {ex.window_id for ex in positives} == {0, 1, 2}
    assert all(ex.quantile == 0 for ex in positives)
    assert [(ex.window_id, ex.activations) for ex in negatives] == [
        (3, [0.0, 0.0, 0.0, 0.0])
    ]
    assert negatives[0].split == "train"

    quiet = _ctx(_corpus(8, 4), train_positives=3, train_negatives=0)
    quiet_sampled = sample_expert_examples(
        quiet,
        "layers.0.mlp",
        1,
        np.array([0, 5, 10], dtype=np.int64),
        np.array([1.0, 3.0, 5.0], dtype=np.float32),
    )
    assert quiet_sampled is not None
    no_negatives = quiet_sampled.examples
    assert all(ex.example_type == "positive" for ex in no_negatives)


def test_routerinterp_hard_negatives_are_the_same_feature_unrouted() -> None:
    corpus = _corpus(6, 4)
    ctx = _ctx(corpus, train_positives=30, train_negatives=30)
    route = np.zeros(corpus.n_tokens, dtype=bool)
    route[[1, 8, 20]] = True
    examples = sample_routerinterp_examples(
        ctx,
        "layers.0.mlp",
        3,
        [FeatureScore(7, 1.0), FeatureScore(8, 0.2), FeatureScore(9, 0.1)],
        {
            7: SparseActivations(
                np.array([1, 6, 8, 9, 20], dtype=np.int64),
                np.array([5.0, 4.0, 8.0, 1.0, 9.0], dtype=np.float32),
            ),
            8: SparseActivations(
                np.array([2], dtype=np.int64), np.array([3.0], dtype=np.float32)
            ),
            9: SparseActivations(
                np.array([22], dtype=np.int64), np.array([9.0], dtype=np.float32)
            ),
        },
        route,
    )
    # Feature 8 has no unrouted fires and feature 9 fires only on test, so both drop.
    assert [(ex.example_type, ex.window_id, ex.activations) for ex in examples] == [
        ("positive", 2, [8.0, 0.0, 0.0, 0.0]),
        ("positive", 0, [0.0, 5.0, 0.0, 0.0]),
        ("hard_negative", 1, [0.0, 0.0, 4.0, 0.0]),
        ("hard_negative", 2, [0.0, 1.0, 0.0, 0.0]),
    ]
    assert {ex.group_id for ex in examples} == {7}


def test_eval_windows_are_held_out_and_negatives_first() -> None:
    corpus = _corpus(8, 4)
    windows = build_eval_windows(
        corpus,
        _Tokenizer(),  # type: ignore[arg-type]
        np.array([0, 16, 21], dtype=np.int64),
        np.array([5.0, 9.0, 2.0], dtype=np.float32),
        ctx_len=4,
        n_positives=5,
        n_negatives=10,
        seed=0,
        highlight_seed=1,
        exclude_window_ids={4, 6},
    )
    assert [window.ground_truth for window in windows] == [False, True]
    assert [window.window_id for window in windows] == [7, 5]
    assert windows[0].activations == [0.0, 0.0, 0.0, 0.0]
    assert windows[0].str_tokens == ["t28", "t29", "t30", "t31"]
    assert windows[0].highlight <= set(range(4))
    assert windows[1].activations == [0.0, 2.0, 0.0, 0.0]
    assert windows[1].highlight == {1}
    assert windows[1].tokens == [20, 21, 22, 23]
