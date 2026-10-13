"""W&B upload of a finished run directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import wandb
from loguru import logger

from routerinterp import core

DEFAULT_ARTIFACT_THRESHOLD_BYTES = 10_000_000

_TAG_KEYS = (
    "method",
    "bases",
    "n",
    "k",
    "expansion_factor",
    "lambda_",
    "ngram_type",
    "solver",
)


def _json_safe(value: Any) -> Any:
    """Convert numpy scalars and nested containers to JSON-safe values."""
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _run_group(resolved_config: dict[str, Any]) -> str:
    """Build the run group ``<model>/L<layers>`` from a manifest."""
    model = resolved_config.get("model") or {}
    name = str(model.get("name") or "unknown")
    name = name.rsplit("/", 1)[-1]
    hookpoints = resolved_config.get("hookpoints") or []
    layers: set[int] = set()
    for hookpoint in hookpoints:
        try:
            layers.add(core.parse_layer_index(str(hookpoint)))
        except ValueError:
            continue
    if layers:
        return f"{name}/L{'-'.join(str(layer) for layer in sorted(layers))}"
    return name


def _run_tags(resolved_config: dict[str, Any], manifest: core.RunManifest) -> list[str]:
    """Build method and sparsity tags, plus a short config-hash tag."""
    tags: list[str] = []
    for key in _TAG_KEYS:
        value = resolved_config.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            value = ",".join(str(v) for v in value)
        tags.append(f"{key}={value}")
    tags.append(f"hash={manifest.config_hash[:8]}")
    return tags


def _artifact_name(manifest: core.RunManifest) -> str:
    base = f"{manifest.stage}-{manifest.run_name}"
    return "".join(c if (c.isalnum() or c in "-_.") else "-" for c in base)[:128]


def _table_dataframe(df: Any) -> Any:
    """Copy ``df`` with list columns stored as JSON text.

    A W&B table fixes a column's shape from its first row, so a list that changes length
    from row to row has to be text.
    """
    out = df.copy()
    for col in out.columns:
        values = out[col].tolist()
        if not any(isinstance(v, (list, tuple, dict, np.ndarray)) for v in values):
            continue
        out[col] = [None if v is None else json.dumps(_json_safe(v)) for v in values]
    return out


def wandb_log(run_dir: Path, *, extra_tags: list[str] | None = None) -> str | None:
    """Upload ``run_dir`` when ``WANDB_PROJECT`` is set.

    Parquet files under the size limit are logged as tables.
    Other files go into one artifact.
    A failure is logged and does not fail the run that already wrote the directory.
    """
    import os

    project = os.environ.get("WANDB_PROJECT")
    if not project:
        return None
    entity = os.environ.get("WANDB_ENTITY") or None
    try:
        return _upload(run_dir, project, entity, extra_tags=extra_tags)
    except Exception:
        logger.exception(
            f"W&B logging failed for {run_dir}; the run directory was still written"
        )
        return None


def _upload(
    run_dir: Path,
    project: str,
    entity: str | None = None,
    *,
    artifact_threshold_bytes: int = DEFAULT_ARTIFACT_THRESHOLD_BYTES,
    extra_tags: list[str] | None = None,
) -> str:
    run_dir = Path(run_dir)
    manifest = core.read_manifest(run_dir)
    resolved = dict(manifest.resolved_config)
    run = wandb.init(
        project=project,
        entity=entity,
        name=manifest.run_name,
        group=_run_group(resolved),
        job_type=manifest.stage,
        tags=_run_tags(resolved, manifest) + list(extra_tags or []),
        config={
            "config_hash": manifest.config_hash,
            "parents": manifest.parents,
            "provenance": manifest.provenance,
            "resolved_config": resolved,
        },
    )

    artifact = wandb.Artifact(name=_artifact_name(manifest), type="pipeline_outputs")
    n_artifact_files = 0
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name == core.MANIFEST_FILENAME:
            continue
        rel = str(path.relative_to(run_dir))
        if path.suffix == ".parquet" and path.stat().st_size < artifact_threshold_bytes:
            df = _table_dataframe(core.read_parquet(path).to_pandas())
            run.log({path.stem: wandb.Table(dataframe=df)})
        else:
            artifact.add_file(str(path), name=rel)
            n_artifact_files += 1
    if n_artifact_files:
        run.log_artifact(artifact)

    run.summary.update(
        {
            "stage": manifest.stage,
            "config_hash": manifest.config_hash,
            "created_at": manifest.created_at,
            "n_artifact_files": n_artifact_files,
        }
    )
    url = run.url
    run.finish()
    if not url:
        msg = f"W&B logging produced no URL for {run_dir}"
        raise RuntimeError(msg)
    logger.info(f"Logged {run_dir} -> {url}")
    return url
