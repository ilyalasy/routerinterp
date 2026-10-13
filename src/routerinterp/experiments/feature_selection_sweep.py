"""Linear probe of expert identity from the top-n selected features, scored on the full
test split.

Training tokens are those where at least one selected feature is active.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from routerinterp.artifacts import load_corpus
from routerinterp.core import (
    MANIFEST_FILENAME,
    ProbeConfig,
    build_manifest,
    manifest_matches,
    parse_layer_index,
    read_parquet,
    router_layer_for_hookpoint,
    write_manifest,
)
from routerinterp.experiments import require, write_tidy
from routerinterp.probes import (
    _load_labels,
    _load_latents,
    _probe_sample,
    _run_linear_basis,
)

FIGURE = "feature_selection_sweep"


@dataclass(frozen=True, slots=True)
class SelectionSlice:
    """One hookpoint's features for a selection method and a top-n cutoff."""

    hookpoint: str
    method: str
    n: int


def probe_selections(
    config: ProbeConfig, selections: list[Path], metrics: Path
) -> Path:
    """Probe each selection file and write the sweep parquet.
    Does not plot.
    """
    require([config.input_dir / MANIFEST_FILENAME, *selections])
    grouped: dict[SelectionSlice, list[dict[str, Any]]] = {}
    for path in selections:
        for row in read_parquet(path).to_pylist():
            key = SelectionSlice(
                str(row["hookpoint"]), str(row["method"]), int(row["n"])
            )
            grouped.setdefault(key, []).append(row)
    hookpoints = list(dict.fromkeys(key.hookpoint for key in grouped))
    probe_cfg = config.model_copy(
        update={
            "hookpoints": hookpoints,
            "bases": ["sae"],
            "output_dir": metrics.parent,
        }
    )
    payload = _payload(
        {
            "selections": [str(path) for path in selections],
            "hookpoints": hookpoints,
            "n_tokens": probe_cfg.n_tokens,
            "seed": probe_cfg.seed,
        }
    )
    run_dir = metrics.parent / FIGURE
    if metrics.exists() and manifest_matches(run_dir, payload, {}):
        logger.info(f"feature-selection sweep up to date -> {metrics}")
        return metrics
    corpus = load_corpus(probe_cfg.input_dir)
    sample = _probe_sample(corpus, probe_cfg)
    train_mask = sample.train_mask_T[sample.selected]
    tidy: list[dict[str, Any]] = []
    for hookpoint in hookpoints:
        layer = router_layer_for_hookpoint(hookpoint)
        latents = _load_latents(probe_cfg.input_dir, hookpoint, sample)
        labels, _, _ = _load_labels(probe_cfg.input_dir, layer, sample)
        for key, records in grouped.items():
            if key.hookpoint != hookpoint:
                continue
            scored = _probe_n(
                latents, labels, train_mask, records, probe_cfg, key.method, hookpoint
            )
            if scored is None:
                continue
            f1, effective = scored
            panel = str(parse_layer_index(hookpoint))
            tidy.append(_tidy(panel, key.method, key.n, "f1", f1))
            if effective is not None:
                tidy.append(_tidy(panel, key.method, key.n, "effective_f1", effective))
    write_tidy(metrics, tidy)
    write_manifest(
        run_dir,
        build_manifest(
            stage="probe-selection",
            run_name="feature-selection-sweep",
            config=payload,
            parents={},
        ),
    )
    logger.info(f"Wrote {FIGURE} -> {metrics}")
    return metrics


def _probe_n(
    latents,
    labels: np.ndarray,
    train_mask: np.ndarray,
    records: list[dict[str, Any]],
    probe_cfg: ProbeConfig,
    method: str,
    hookpoint: str,
) -> tuple[float, float | None] | None:
    by_expert: dict[int, list[int]] = {}
    for row in records:
        by_expert.setdefault(int(row["expert_id"]), []).append(int(row["feature_id"]))
    f1s: list[float] = []
    effective_f1s: list[float] = []
    for expert_id, feats in sorted(by_expert.items()):
        cols = np.asarray(feats, dtype=np.int32)
        subset = latents[:, cols]
        # Train where the expert's n features fire; score every test token.
        active = np.asarray(subset.getnnz(axis=1) > 0).ravel()
        train = train_mask & active
        test = ~train_mask
        if int(train.sum()) == 0 or int(test.sum()) == 0:
            continue
        y = np.asarray(labels[:, expert_id])
        fitted = _run_linear_basis(
            basis=method,
            hookpoint=hookpoint,
            X_train=subset[train],
            X_test=subset[test],
            Y_train=y[train, None],
            Y_test=y[test, None],
            config=probe_cfg,
            coverage=np.asarray(active[test]).ravel(),
        )
        f1s.extend(float(row["value"]) for row in fitted if row["metric"] == "f1")
        effective_f1s.extend(
            float(row["value"]) for row in fitted if row["metric"] == "effective_f1"
        )
    if not f1s:
        logger.warning(f"No probe F1 for {method} {hookpoint}")
        return None
    effective = float(np.mean(effective_f1s)) if effective_f1s else None
    return float(np.mean(f1s)), effective


def _tidy(panel: str, method: str, n: int, metric: str, value: float) -> dict[str, Any]:
    return {
        "figure": FIGURE,
        "panel": panel,
        "series": method,
        "x": n,
        "metric": metric,
        "value": value,
        "split": "test",
    }


def _payload(data: dict[str, Any]):
    from pydantic import BaseModel, ConfigDict

    class _Payload(BaseModel):
        model_config = ConfigDict(extra="allow")

    return _Payload.model_validate(data)
