"""For each expert and Pile subset, that subset's share of the expert's routed tokens,
divided by its share of the whole corpus.

A value of 1 means the expert routes the subset at the corpus rate.
The collect run must have stored ``pile_set_name``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from routerinterp.collect import load_collect_info, load_pile_set_names
from routerinterp.core import (
    MANIFEST_FILENAME,
    ROUTER_TOPK_IDS,
    parse_layer_index,
    read_safetensors,
)
from routerinterp.experiments import require, write_tidy

FIGURE = "pile_domains"


def pile_domains(collect_dir: Path, metrics: Path) -> Path:
    """Compute routing preference from one collect run."""
    require(
        [collect_dir / MANIFEST_FILENAME, collect_dir / "tokens" / "pile_set_names"]
    )
    info = load_collect_info(collect_dir)
    names = load_pile_set_names(collect_dir)
    seq_len = int(info["seq_len"])
    per_token = np.repeat(names, seq_len)
    rows: list[dict[str, Any]] = []
    for hookpoint, router_layer in info["router_layers"].items():
        rows.extend(
            _routing_preference(
                collect_dir,
                int(router_layer),
                per_token,
                x_layer=parse_layer_index(str(hookpoint)),
            )
        )
    write_tidy(metrics, rows)
    logger.info(f"Wrote {FIGURE} -> {metrics}")
    return metrics


def _routing_preference(
    collect_dir: Path, layer: int, pile_names: np.ndarray, *, x_layer: int
) -> list[dict[str, Any]]:
    parts: list[np.ndarray] = []
    router_dir = collect_dir / "router" / f"layers_{layer}"
    for path in sorted(router_dir.glob("shard-*.safetensors")):
        parts.append(np.asarray(read_safetensors(path)[ROUTER_TOPK_IDS]))
    if not parts:
        msg = f"No router shards for layer {layer} under {router_dir}"
        raise FileNotFoundError(msg)
    topk = np.concatenate(parts, axis=0)
    n = min(topk.shape[0], pile_names.shape[0])
    if topk.shape[0] != pile_names.shape[0]:
        logger.warning(
            f"Layer {layer}: router tokens {topk.shape[0]} vs pile labels "
            f"{pile_names.shape[0]}; using {n}"
        )
    topk = topk[:n]
    names = pile_names[:n]
    subsets, inverse = np.unique(names.astype(str), return_inverse=True)
    corpus = np.bincount(inverse, minlength=len(subsets)).astype(np.float64)
    corpus /= max(n, 1)
    n_experts = int(topk.max()) + 1 if topk.size else 0
    routed = np.zeros((n, n_experts), dtype=bool)
    np.put_along_axis(routed, topk.astype(np.int64), True, axis=1)
    rows: list[dict[str, Any]] = []
    for expert_id in range(n_experts):
        mask = routed[:, expert_id]
        total = int(mask.sum())
        if total == 0:
            continue
        counts = np.bincount(inverse[mask], minlength=len(subsets)).astype(np.float64)
        for subset_id, subset in enumerate(subsets):
            share = float(counts[subset_id] / total)
            base = float(corpus[subset_id])
            if base <= 0 or share == 0:
                continue
            panel = str(expert_id)
            rows.append(_row(subset, x_layer, "routing_share", share, panel))
            rows.append(_row(subset, x_layer, "lift", share / base, panel))
    logger.info(f"Layer {layer}: {len(subsets)} subsets, {n_experts} experts")
    return rows


def _row(
    subset: str, layer: int, metric: str, value: float, panel: str
) -> dict[str, Any]:
    return {
        "figure": FIGURE,
        "panel": panel,
        "series": str(subset),
        "x": layer,
        "metric": metric,
        "value": value,
        "split": "",
    }
