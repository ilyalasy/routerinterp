"""Toy-tensor checks for margins and top-n feature selection."""

from __future__ import annotations

import pytest
import torch

from routerinterp.select import (
    FeatureScore,
    MarginAggState,
    accumulate_contributions,
    membership_margins,
    select_top_features,
)


def test_membership_margins_sign_and_ties() -> None:
    logits = torch.tensor(
        [
            [
                [3.0, 1.0, 0.0, -2.0],  # top-2 are experts 0 and 1
                [0.0, 4.0, 4.0, 1.0],  # tie for the top-2
            ]
        ]
    )
    margins = membership_margins(logits, k=2)
    torch.testing.assert_close(
        margins,
        torch.tensor(
            [
                [
                    [3.0, 1.0, -1.0, -3.0],
                    [-4.0, 3.0, 3.0, -3.0],
                ]
            ]
        ),
    )
    with pytest.raises(ValueError, match="k must"):
        membership_margins(logits, k=0)
    with pytest.raises(ValueError, match="k must"):
        membership_margins(logits, k=4)


def test_membership_margin_jacobian_matches_finite_difference() -> None:
    # Strictly ordered logits, so the top-k boundary is a single linear piece.
    # k=2: tau is the 3rd-largest logit inside the top-2, and the 2nd outside.
    logits = torch.tensor([5.0, 1.0, -0.5, -3.0], dtype=torch.float64)

    def margins(row: torch.Tensor) -> torch.Tensor:
        return membership_margins(row.view(1, 1, -1), k=2).reshape(-1)

    jacobian = torch.autograd.functional.jacobian(margins, logits)
    expected = torch.tensor(
        [
            [1.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, -1.0, 0.0],
            [0.0, -1.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(jacobian, expected)

    eps = 1e-4
    finite = torch.empty_like(jacobian)
    for index in range(logits.numel()):
        plus = logits.clone()
        minus = logits.clone()
        plus[index] += eps
        minus[index] -= eps
        finite[:, index] = (margins(plus) - margins(minus)) / (2 * eps)
    torch.testing.assert_close(jacobian, finite)


def test_negative_contributions_and_inactive_tokens_stay_out_of_support() -> None:
    state = MarginAggState.create(1, 3)
    accumulate_contributions(
        state,
        expert_id=0,
        y_active_T=torch.tensor([True, False, False]),
        feature_ids_list=[
            torch.tensor([0, 1]),
            torch.tensor([0]),
            torch.tensor([2]),
        ],
        scores_list=[
            torch.tensor([2.0, -4.0]),  # active: negative score is not support
            torch.tensor([5.0]),  # inactive: counts toward L, not A
            torch.tensor([-3.0]),  # inactive and negative: dropped from L
        ],
    )
    scores, coverage, effect = state.finalize()
    torch.testing.assert_close(scores[0], torch.tensor([-0.5, 0.0, 0.0]))
    torch.testing.assert_close(coverage[0], torch.tensor([1.0, 1.0, 0.0]))
    torch.testing.assert_close(effect[0], torch.tensor([2.0, -4.0, 0.0]))


def _assert_picks(
    got: dict[int, list[FeatureScore]],
    expected: dict[int, list[FeatureScore]],
) -> None:
    assert set(got) == set(expected)
    for expert, pairs in expected.items():
        assert [pick.feature_id for pick in got[expert]] == [
            pick.feature_id for pick in pairs
        ]
        for got_pick, want in zip(got[expert], pairs, strict=True):
            assert got_pick.score == pytest.approx(want.score)


def test_select_top_features_overlap_exclusive_and_rejection() -> None:
    scores = torch.tensor(
        [
            [1.0, 0.5, 0.25, -1.0],
            [0.75, 0.375, 0.125, -0.5],
        ]
    )
    shared = select_top_features(scores, n=2, exclusive=False)
    _assert_picks(
        shared,
        {
            0: [FeatureScore(0, 1.0), FeatureScore(1, 0.5)],
            1: [FeatureScore(0, 0.75), FeatureScore(1, 0.375)],
        },
    )
    assert shared[0][0].feature_id == shared[1][0].feature_id

    exclusive = select_top_features(scores, n=2, exclusive=True)
    _assert_picks(
        exclusive,
        {
            0: [FeatureScore(0, 1.0), FeatureScore(1, 0.5)],
            1: [FeatureScore(2, 0.125), FeatureScore(3, -0.5)],
        },
    )
    picked = [pick.feature_id for pairs in exclusive.values() for pick in pairs]
    assert len(picked) == len(set(picked))
    # The stolen features leave expert 1 with a negative score rather than nothing.
    assert exclusive[1][-1].score < 0
    assert all(pick.score > 0 for pick in shared[1])

    starved = select_top_features(torch.tensor([[5.0], [4.0]]), n=1, exclusive=True)
    assert starved[0] == [FeatureScore(0, 5.0)]
    assert starved[1] == []

    everything = select_top_features(scores, n=10, exclusive=False)
    assert [pick.feature_id for pick in everything[0]] == [0, 1, 2, 3]
    assert everything[0][-1].score < 0
    with pytest.raises(ValueError, match="n must"):
        select_top_features(scores, n=0, exclusive=False)
