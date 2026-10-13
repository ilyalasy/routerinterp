"""Train-only PCA and linear probes: a test spike does not move the fit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from routerinterp import probes
from routerinterp.core import ProbeConfig
from routerinterp.probes import _fit_pca_on_train, _run_linear_basis, keep_topk_features


def test_keep_topk_features_zeros_the_small_magnitudes() -> None:
    features = np.array(
        [
            [0.1, -3.0, 0.5, 2.0],
            [4.0, 0.0, -0.2, 0.1],
        ]
    )
    kept = keep_topk_features(features, k=2)
    np.testing.assert_array_equal(
        kept,
        np.array(
            [
                [0.0, -3.0, 0.0, 2.0],
                [4.0, 0.0, -0.2, 0.0],
            ]
        ),
    )
    np.testing.assert_array_equal(keep_topk_features(features, k=4), features)
    np.testing.assert_array_equal(
        keep_topk_features(features, k=0), np.zeros_like(features)
    )


def test_pca_train_mean_ignores_a_test_spike(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(1)
    train = rng.normal(size=(24, 5))
    test = rng.normal(size=(6, 5))
    spiked = test.copy()
    spiked[0, 0] += 1.0e6
    means: list[np.ndarray] = []

    class _RecordingPCA(PCA):
        def fit_transform(self, X, y=None):  # type: ignore[no-untyped-def]
            transformed = super().fit_transform(X, y)
            means.append(np.array(self.mean_, copy=True))
            return transformed

    monkeypatch.setattr("sklearn.decomposition.PCA", _RecordingPCA)
    train_clean, test_clean, dim_clean = _fit_pca_on_train(train, test, 2)
    train_spiked, test_spiked, dim_spiked = _fit_pca_on_train(train, spiked, 2)

    assert dim_clean == dim_spiked == 2
    np.testing.assert_array_equal(train_clean, train_spiked)
    np.testing.assert_allclose(means[0], train.mean(axis=0))
    np.testing.assert_allclose(means[1], means[0])
    assert not np.allclose(test_clean, test_spiked)


def test_linear_probe_coefficients_ignore_test_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(16, 3))
    x_test = rng.normal(size=(8, 3))
    spiked = x_test.copy()
    spiked[0, 0] += 1.0e6
    y_train = np.zeros((16, 2), dtype=np.int64)
    y_train[:8, 0] = 1
    y_train[8:, 1] = 1
    y_test = np.zeros((8, 2), dtype=np.int64)
    y_test[:4, 0] = 1
    y_test[4:, 1] = 1
    config = ProbeConfig(
        input_dir=tmp_path,
        output_dir=tmp_path / "probes",
        hookpoints=["layers.0.mlp"],
        min_tokens_per_expert=4,
    )
    recorded: list[np.ndarray] = []

    class _Recording:
        def __init__(self, **kwargs: object) -> None:
            self._clf = LogisticRegression(**kwargs)  # type: ignore[arg-type]

        def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: object = None):
            self._clf.fit(x, y, sample_weight=sample_weight)
            recorded.append(np.array(self._clf.coef_, copy=True))
            return self

        def decision_function(self, x: np.ndarray) -> np.ndarray:
            return self._clf.decision_function(x)

        def predict(self, x: np.ndarray) -> np.ndarray:
            return self._clf.predict(x)

    monkeypatch.setattr(probes, "LogisticRegression", _Recording)
    _run_linear_basis(
        basis="sae",
        hookpoint="layers.0.mlp",
        X_train=x_train,
        X_test=x_test,
        Y_train=y_train,
        Y_test=y_test,
        config=config,
    )
    clean = [row.copy() for row in recorded]
    recorded.clear()
    _run_linear_basis(
        basis="sae",
        hookpoint="layers.0.mlp",
        X_train=x_train,
        X_test=spiked,
        Y_train=y_train,
        Y_test=y_test,
        config=config,
    )

    assert len(clean) == y_train.shape[1] == len(recorded)
    for expert, (got, after_spike) in enumerate(zip(clean, recorded, strict=True)):
        oracle = LogisticRegression(
            solver=config.solver,
            C=config.logreg_c,
            max_iter=config.max_iter,
            class_weight="balanced",
        )
        oracle.fit(x_train, y_train[:, expert])
        np.testing.assert_allclose(got, oracle.coef_)
        np.testing.assert_allclose(after_spike, oracle.coef_)
