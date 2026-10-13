"""CLI for pipeline stages and the figure jobs that read produced artifacts .

Flags override yaml fields.
A plot command exits when an input is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from dotenv import load_dotenv

from routerinterp.collect import run_collect
from routerinterp.core import (
    CollectConfig,
    EvaluateConfig,
    ExplainConfig,
    ProbeConfig,
    SelectConfig,
    hookpoint_slug,
    load_yaml_config,
    resolve_run_path,
)
from routerinterp.experiments import MissingInputs, require
from routerinterp.experiments.feature_clustering import cluster as run_cluster
from routerinterp.experiments.feature_selection_sweep import probe_selections
from routerinterp.experiments.pile_domains import pile_domains
from routerinterp.explain.evaluate import run_evaluate
from routerinterp.explain.explain import run_explain
from routerinterp.plot import (
    plot_clusters_per_expert,
    plot_explanation_length,
    plot_explanation_scores,
    plot_feature_selection_sweep,
    plot_pile_distribution,
)
from routerinterp.probes import run_probes
from routerinterp.select import run_select
from routerinterp.train_sae import SAETrainConfig, train_sae

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def _cli() -> None:
    load_dotenv()


def _run_path(path: Path) -> Path:
    return resolve_run_path(path)


def _run_paths(paths: list[Path]) -> list[Path]:
    return [resolve_run_path(path) for path in paths]


ConfigArg = Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)]
OutArg = Annotated[Path, typer.Option("--out")]


def _section(path: Path, key: str) -> dict[str, Any]:
    raw = load_yaml_config(path)
    if not isinstance(raw, dict) or key not in raw:
        msg = f"{path} has no {key!r} section"
        raise typer.BadParameter(msg)
    section = raw[key]
    if not isinstance(section, dict):
        msg = f"{path}:{key} must be a mapping"
        raise typer.BadParameter(msg)
    return dict(section)


def _apply(section: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    for key, value in overrides.items():
        if value is not None:
            section[key] = value
    return section


def _guard(fn: Any) -> None:
    try:
        fn()
    except MissingInputs as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


# Pipeline stages.


@app.command()
def collect(
    config: ConfigArg,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    batch_size: Annotated[int | None, typer.Option("--batch-size")] = None,
) -> None:
    """Cache tokens, router outputs, and SAE latents."""
    section = _apply(
        _section(config, "collect"), output_dir=output_dir, batch_size=batch_size
    )
    run_collect(CollectConfig.model_validate(section))


@app.command("train-sae")
def train_sae_cmd(
    config: ConfigArg,
    batch_size: Annotated[int | None, typer.Option("--batch-size")] = None,
) -> None:
    """Train a Top-K SAE.

    The model comes from the collect config when this section does not set one.
    The dataset comes from collect only when this section leaves it out, so a Pile
    collect corpus is not reused by accident.
    """
    raw = load_yaml_config(config)
    section = dict(_section(config, "sae_training"))
    collect_cfg = raw.get("collect") if isinstance(raw, dict) else None
    if isinstance(collect_cfg, dict):
        section.setdefault("model", collect_cfg.get("model"))
        section.setdefault("dataset", collect_cfg.get("dataset"))
    section = _apply(section, batch_size=batch_size)
    train_sae(SAETrainConfig.model_validate(section))


@app.command()
def select(
    config: ConfigArg,
    method: Annotated[str | None, typer.Option("--method")] = None,
    n: Annotated[int | None, typer.Option("--n")] = None,
    input_dir: Annotated[Path | None, typer.Option("--input-dir")] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    batch_size: Annotated[int | None, typer.Option("--batch-size")] = None,
) -> None:
    """Score and select SAE features for each expert.
    One ``n`` per job.
    """
    section = _apply(
        _section(config, "select"),
        method=method,
        n=n,
        input_dir=input_dir,
        output_dir=output_dir,
        batch_size=batch_size,
    )
    run_select(SelectConfig.model_validate(section))


@app.command()
def explain(
    config: ConfigArg,
    method: Annotated[str | None, typer.Option("--method")] = None,
    input_dir: Annotated[Path | None, typer.Option("--input-dir")] = None,
    selection_dir: Annotated[Path | None, typer.Option("--selection-dir")] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
) -> None:
    """Generate explanations for one method."""
    section = _apply(
        _section(config, "explain"),
        method=method,
        input_dir=input_dir,
        selection_dir=selection_dir,
        output_dir=output_dir,
    )
    run_explain(ExplainConfig.model_validate(section))


@app.command()
def evaluate(
    config: ConfigArg,
    explanations_dir: Annotated[Path | None, typer.Option("--explanations-dir")] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
) -> None:
    """Score a finished explain run on held-out windows."""
    section = _apply(
        _section(config, "evaluate"),
        explanations_dir=explanations_dir,
        output_dir=output_dir,
    )
    run_evaluate(EvaluateConfig.model_validate(section))


@app.command()
def probe(
    config: ConfigArg,
    basis: Annotated[list[str] | None, typer.Option("--basis")] = None,
    input_dir: Annotated[Path | None, typer.Option("--input-dir")] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
) -> None:
    """Fit SAE / neuron / PCA / n-gram routing probes on a finished collect."""
    section = _apply(
        _section(config, "probes"), input_dir=input_dir, output_dir=output_dir
    )
    if basis:
        section["bases"] = basis
    cfg = ProbeConfig.model_validate(section)
    needed = [cfg.input_dir / "manifest.json"]
    if any(name in ("neuron", "pca") for name in cfg.bases):
        needed.extend(
            cfg.input_dir / "residuals" / hookpoint_slug(hp) for hp in cfg.hookpoints
        )
    _guard(lambda: require(needed))
    run_probes(cfg)


# Analysis jobs. Each reads finished stage outputs and writes a tidy parquet.


@app.command()
def cluster(
    explanations: Annotated[Path, typer.Option("--explanations")],
    selection: Annotated[Path, typer.Option("--selection")],
    out: OutArg,
    n: Annotated[int, typer.Option("--n")] = 20,
    k_max: Annotated[int, typer.Option("--k-max")] = 50,
    embedding_model: Annotated[
        str, typer.Option("--embedding-model")
    ] = "sentence-transformers/all-MiniLM-L6-v2",
) -> None:
    """Cluster one model's feature explanations."""
    _guard(
        lambda: run_cluster(
            _run_path(explanations),
            _run_path(selection),
            _run_path(out),
            n=n,
            k_max=k_max,
            embedding_model=embedding_model,
        )
    )


@app.command("probe-selection")
def probe_selection_cmd(
    config: ConfigArg,
    selection: Annotated[list[Path], typer.Option("--selection")],
    out: OutArg,
    collect: Annotated[Path | None, typer.Option("--collect")] = None,
) -> None:
    """Probe finished selection files and write the feature-selection-sweep parquet."""
    if not selection:
        msg = "Pass at least one --selection file"
        raise typer.BadParameter(msg)
    out = _run_path(out)
    section = _apply(
        _section(config, "probes"),
        input_dir=None if collect is None else _run_path(collect),
        output_dir=out.parent,
    )
    cfg = ProbeConfig.model_validate(section)
    _guard(lambda: probe_selections(cfg, _run_paths(selection), out))


@app.command("pile-domains")
def pile_domains_cmd(
    collect: Annotated[Path, typer.Option("--collect")],
    out: OutArg,
) -> None:
    """Compute routing preference from one collect run."""
    _guard(lambda: pile_domains(_run_path(collect), _run_path(out)))


# Plot jobs.


@app.command("plot-clusters")
def plot_clusters_cmd(
    label: Annotated[list[str], typer.Option("--label")],
    metrics: Annotated[list[Path], typer.Option("--metrics")],
    out: OutArg,
) -> None:
    """Histogram of clusters spanned per expert, one panel per metrics parquet."""
    if not label or len(label) != len(metrics):
        msg = "Pass the same number of --label and --metrics values"
        raise typer.BadParameter(msg)
    paths = _run_paths(metrics)
    _guard(
        lambda: (
            require(paths),
            plot_clusters_per_expert(
                list(zip(label, paths, strict=True)), _run_path(out)
            ),
        )
    )


@app.command("plot-feature-selection-sweep")
def plot_feature_selection_sweep_cmd(
    metrics: Annotated[Path, typer.Option("--metrics")],
    out: OutArg,
) -> None:
    """Effective F1 versus features per expert, from the sweep parquet."""
    path = _run_path(metrics)
    _guard(
        lambda: (require([path]), plot_feature_selection_sweep(path, _run_path(out)))
    )


@app.command("plot-pile-domains")
def plot_pile_domains_cmd(
    metrics: Annotated[Path, typer.Option("--metrics")],
    out: OutArg,
) -> None:
    """Expert preference for Pile subsets, from the pile-domains parquet."""
    path = _run_path(metrics)
    _guard(lambda: (require([path]), plot_pile_distribution(path, _run_path(out))))


@app.command("plot-explanation-length")
def plot_explanation_length_cmd(
    evaluate: Annotated[list[Path], typer.Option("--evaluate")],
    out: OutArg,
) -> None:
    """Explanation F1 against explanation length, from evaluate directories."""
    paths = _run_paths(evaluate)
    _guard(lambda: (require(paths), plot_explanation_length(paths, _run_path(out))))


@app.command("plot-explanation-scores")
def plot_explanation_scores_cmd(
    label: Annotated[list[str], typer.Option("--label")],
    evaluate: Annotated[list[Path], typer.Option("--evaluate")],
    out: OutArg,
) -> None:
    """Early, mid, and late explanation F1 from each model's evaluate directory."""
    if not label or len(label) != len(evaluate):
        msg = "Pass the same number of --label and --evaluate values"
        raise typer.BadParameter(msg)
    paths = _run_paths(evaluate)
    _guard(
        lambda: (
            require(paths),
            plot_explanation_scores(
                list(zip(label, paths, strict=True)), _run_path(out)
            ),
        )
    )


if __name__ == "__main__":
    app()
