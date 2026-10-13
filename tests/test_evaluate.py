"""Ordinal F1 and n-gram lookup.
No model, no scorer client.
"""

from __future__ import annotations

from routerinterp.explain.evaluate import lookup_predict
from routerinterp.explain.utils import (
    binarise_scores,
    compute_prf1,
    parse_ordinal_scores,
)


def test_ordinal_scores_binarise_into_prf1() -> None:
    fenced = """prefix\n```json\n{"0": 4, "1": 0, "2": 3, "3": 1}\n```\n"""
    scores = parse_ordinal_scores(fenced, 4)
    assert scores == [4, 0, 3, 1]
    predictions = binarise_scores(scores, threshold=2)
    assert predictions == [True, False, True, False]
    mixed = compute_prf1([True, True, False, False], predictions)
    assert (mixed.tp, mixed.fp, mixed.fn, mixed.tn) == (1, 1, 1, 1)
    assert mixed.precision == mixed.recall == mixed.f1 == 0.5

    # Scores stay below the threshold, so every prediction is negative.
    below = binarise_scores(
        parse_ordinal_scores('{"example_0": 0, "example_1": 1}', 2),
        threshold=2,
    )
    assert below == [False, False]
    all_negative = compute_prf1([False, False], below)
    assert (all_negative.precision, all_negative.recall, all_negative.f1) == (
        0.0,
        0.0,
        0.0,
    )
    assert all_negative.tn == 2 and all_negative.tp == 0

    all_positive = compute_prf1([True, True], below)
    assert (all_positive.precision, all_positive.recall, all_positive.f1) == (
        0.0,
        0.0,
        0.0,
    )
    assert all_positive.fn == 2 and all_positive.fp == 0


def test_lookup_predict_hits_bigrams_and_falls_back() -> None:
    tokens = [10, 20, 30, 40]
    # Highlighted token 30 is in the unigram table.
    assert lookup_predict(tokens, {2}, [[30], [99]], None, bigram=False)
    # Pair (20, 30) is in the bigram table.
    assert lookup_predict(tokens, {2}, [[20, 30]], None, bigram=True)
    # The pair misses; the current token is in the expert unigrams.
    assert lookup_predict(tokens, {2}, [[8, 9]], [[30]], bigram=True)
    assert not lookup_predict(tokens, {2}, [[99]], None, bigram=False)
    assert not lookup_predict(tokens, {2}, [[8, 9]], [[11]], bigram=True)
