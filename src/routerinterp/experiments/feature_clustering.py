"""Main SSH vs DSH evidence. Embed each expert's feature explanations, cluster the embeddings, and count how many
clusters that expert's features fall into.

DSH predicts one cluster per expert.
SSH predicts that count grows with the number of features.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from routerinterp.core import (
    MANIFEST_FILENAME,
    build_manifest,
    manifest_matches,
    parse_layer_index,
    read_manifest,
    read_parquet,
    write_manifest,
)
from routerinterp.experiments import require, write_tidy
from routerinterp.explain.explain import load_explanations

FIGURE = "feature_clustering"


@dataclass(frozen=True, slots=True)
class HookFeature:
    """A feature explanation at one hookpoint."""

    hookpoint: str
    feature_id: int


@dataclass(frozen=True, slots=True)
class ExpertFeature:
    """One selected feature and the expert it was assigned to."""

    expert_id: int
    feature_id: int


def cluster(
    explanations: Path,
    selection: Path,
    metrics: Path,
    *,
    n: int,
    k_max: int,
    embedding_model: str,
) -> None:
    """Cluster one model's feature-autointerp run and write its parquet."""
    require([explanations, selection])
    explain_dir = explanations.parent
    parents: dict[str, str] = {}
    manifest = explain_dir / MANIFEST_FILENAME
    if manifest.exists():
        parents = {"explain": read_manifest(explain_dir).config_hash}
    records = read_parquet(selection).to_pylist()
    hookpoints = list(dict.fromkeys(str(row["hookpoint"]) for row in records))
    payload = _as_config(
        {
            "n": n,
            "k_max": k_max,
            "embedding_model": embedding_model,
            "hookpoints": hookpoints,
            "explanations": str(explanations),
            "selection": str(selection),
        }
    )
    run_dir = metrics.parent / FIGURE
    if metrics.exists() and manifest_matches(run_dir, payload, parents):
        logger.info(f"clustering up to date -> {metrics}")
        return
    loaded = load_explanations(explanations)
    texts: dict[HookFeature, str] = {}
    for key, record in loaded.items():
        if key.target_kind == "feature" and record.get("text"):
            texts[HookFeature(key.hookpoint, key.target_id)] = str(record["text"])
    rows: list[dict[str, Any]] = []
    for hookpoint in hookpoints:
        hooked = [
            row
            for row in records
            if row["hookpoint"] == hookpoint and int(row.get("rank", 0)) < n
        ]
        rows.extend(
            _cluster_explanations(
                hookpoint,
                hooked,
                texts,
                latents_per_expert=n,
                k_max=k_max,
                embedding_model=embedding_model,
            )
        )
    write_tidy(metrics, rows)
    write_manifest(
        run_dir,
        build_manifest(
            stage="cluster",
            run_name="cluster",
            config=payload,
            parents=parents,
        ),
    )
    logger.info(f"Wrote {FIGURE} -> {metrics}")


def _as_config(payload: dict[str, Any]):
    """Hash payload via a one-off pydantic model (manifests hash BaseModel)."""
    from pydantic import BaseModel, ConfigDict

    class _Payload(BaseModel):
        model_config = ConfigDict(extra="allow")

    return _Payload.model_validate(payload)


def _cluster_explanations(
    hookpoint: str,
    selection: list[dict[str, Any]],
    explanations: dict[HookFeature, str],
    *,
    latents_per_expert: int,
    k_max: int,
    embedding_model: str,
) -> list[dict[str, Any]]:
    """Cluster one hookpoint's explanation embeddings with k-means and return the rows."""
    order: list[ExpertFeature] = []
    for record in selection:
        order.append(ExpertFeature(int(record["expert_id"]), int(record["feature_id"])))
    kept = [
        pair
        for pair in order
        if explanations.get(HookFeature(hookpoint, pair.feature_id))
    ]
    if len(kept) < 3:
        msg = f"{hookpoint}: need feature explanations to cluster (got {len(kept)})"
        raise FileNotFoundError(msg)
    unique_ids = list(dict.fromkeys(pair.feature_id for pair in kept))
    vectors = _embed(
        [explanations[HookFeature(hookpoint, feature_id)] for feature_id in unique_ids],
        embedding_model,
    )
    by_id = {fid: vectors[i] for i, fid in enumerate(unique_ids)}
    emb = np.stack([by_id[pair.feature_id] for pair in kept])
    labels, k, silhouette = _best_kmeans(emb, k_max)
    expert_of = np.asarray([pair.expert_id for pair in kept], dtype=np.int32)
    layer = parse_layer_index(hookpoint)
    rows: list[dict[str, Any]] = []
    for expert_id in sorted(set(expert_of.tolist())):
        occupied = len(set(labels[expert_of == expert_id].tolist()))
        rows.append(_row("kmeans", layer, "g", occupied, panel=str(expert_id)))
    rows.append(_row("dsh", layer, "g", 1.0))
    rows.append(_row("ssh", layer, "g", float(latents_per_expert)))
    rows.append(_row("random", layer, "g", _expected_random_g(k, latents_per_expert)))
    rows.append(_row("kmeans", layer, "ari", adjusted_rand_score(expert_of, labels)))
    rows.append(
        _row("kmeans", layer, "nmi", normalized_mutual_info_score(expert_of, labels))
    )
    rows.append(_row("kmeans", layer, "silhouette", silhouette))
    logger.info(
        f"{hookpoint}: K={k} silhouette={silhouette:.3f} "
        f"mean G={np.mean([r['value'] for r in rows if r['series'] == 'kmeans' and r['metric'] == 'g']):.2f}"
    )
    return rows


def _row(
    series: str, layer: int, metric: str, value: float, *, panel: str = ""
) -> dict[str, Any]:
    return {
        "figure": FIGURE,
        "panel": panel,
        "series": series,
        "x": layer,
        "metric": metric,
        "value": value,
        "split": "",
    }


def _expected_random_g(k: int, n: int) -> float:
    """Return how many clusters n latents would occupy if they fell uniformly into K
    clusters.
    """
    if k <= 1:
        return 1.0
    return float(k * (1.0 - (1.0 - 1.0 / k) ** n))


def _embed(texts: list[str], embedding_model: str) -> np.ndarray:
    model = SentenceTransformer(embedding_model)
    encoded = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(encoded, dtype=np.float32)


def _best_kmeans(emb: np.ndarray, k_max: int) -> tuple[np.ndarray, int, float]:
    n_rows = emb.shape[0]
    limit = min(k_max, n_rows - 1)
    best_labels = np.zeros(n_rows, dtype=np.int32)
    best_k, best_sil = 1, float("nan")
    for g in range(2, limit + 1):
        # sklearn types n_init from its "auto" default, but an int is valid and
        # keeps ten restarts (n_init="auto" would run once for k-means++).
        labels = KMeans(
            n_clusters=g, n_init=cast(Any, 10), random_state=42
        ).fit_predict(emb)
        if len(set(labels.tolist())) < 2:
            continue
        sil = float(silhouette_score(emb, labels, metric="cosine"))
        if not np.isnan(best_sil) and sil <= best_sil:
            continue
        best_labels, best_k, best_sil = labels.astype(np.int32), g, sil
    return best_labels, best_k, best_sil
