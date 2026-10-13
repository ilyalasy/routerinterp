"""Paper figures that are drawn from experiment results.

Each function writes the file the paper includes:

* clusters per expert
* feature-selection sweep (effective F1)
* Pile-domain expert preference
* explanation F1 against explanation length
* explanation F1 by depth
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter
from matplotlib.transforms import blended_transform_factory
from scipy.stats import gaussian_kde

from routerinterp.core import parse_layer_index, read_jsonl


def _take(frame: pd.DataFrame, mask: Any) -> pd.DataFrame:
    """Rows selected by a boolean mask.

    ``DataFrame.__getitem__`` has no return annotation, so pyright treats the
    result as a frame, a series, or an array.
    """
    return cast(pd.DataFrame, frame.loc[mask])


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    return cast(pd.Series, frame[name])


# ---------------------------------------------------------------------------
# Clusters per expert
# ---------------------------------------------------------------------------

_CLUSTER_PALETTE = ("#457b9d", "#e07b39")
_MEAN_COLOR = "#1a1a2e"
_CLUSTER_X_MIN = 1
_CLUSTER_X_MAX = 10


def load_clusters_per_expert(
    metrics: Path, *, layer: int | None = None
) -> tuple[np.ndarray, int]:
    """Return clusters spanned per expert, and the hookpoint layer they came from.

    Rows are ``metric="g"`` and ``series="kmeans"``, with the expert id in ``panel``
    and the hookpoint layer in ``x``. The deepest layer is used when ``layer`` is omitted.
    """
    frame = pd.read_parquet(metrics)
    rows = frame[
        (frame["series"] == "kmeans")
        & (frame["metric"] == "g")
        & (frame["panel"].astype(str) != "")
    ].copy()
    if rows.empty:
        msg = f"No per-expert cluster counts in {metrics}"
        raise ValueError(msg)
    rows["layer"] = rows["x"].astype(int)
    chosen = int(rows["layer"].max()) if layer is None else layer
    counts = rows.loc[rows["layer"] == chosen, "value"].to_numpy(dtype=float)
    if counts.size == 0:
        msg = f"No cluster counts for layer {chosen} in {metrics}"
        raise ValueError(msg)
    return counts, chosen


def _draw_cluster_histogram(
    ax: Axes,
    counts: np.ndarray,
    label: str,
    color: str,
    *,
    x_max: int,
) -> None:
    bins = [float(edge) for edge in np.arange(_CLUSTER_X_MIN - 0.5, x_max + 1.5)]
    heights, _, _ = ax.hist(
        counts,
        bins=bins,
        color=color,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.85,
        zorder=3,
    )
    bar_heights = np.asarray(heights, dtype=float)
    mean_val = float(counts.mean())
    ax.set_ylim(0, float(bar_heights.max()) * 1.40)
    ax.axvline(mean_val, color=_MEAN_COLOR, linewidth=2.5, linestyle="-", zorder=5)
    ax.annotate(
        f"mean = {mean_val:.1f}",
        xy=(mean_val, float(bar_heights.max())),
        xytext=(mean_val, float(bar_heights.max()) * 1.08),
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=_MEAN_COLOR,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#999999",
            "linewidth": 0.8,
        },
        zorder=6,
    )
    ax.set_xlabel("Similarity clusters spanned per expert", fontsize=10)
    ax.set_ylabel("Number of experts", fontsize=10)
    ax.set_title(label, fontsize=10)
    ax.set_xlim(_CLUSTER_X_MIN - 0.5, x_max + 0.5)
    ax.set_xticks(range(_CLUSTER_X_MIN, x_max + 1, 2))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="x", color="gray", linewidth=0.3, alpha=0.45, zorder=0)
    ax.grid(axis="y", color="gray", linewidth=0.4, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        _CLUSTER_X_MIN,
        -0.18,
        "← DSH",
        transform=trans,
        ha="center",
        va="top",
        fontsize=8,
        fontstyle="italic",
        color="#555555",
    )
    ax.text(
        x_max,
        -0.18,
        "SSH →",
        transform=trans,
        ha="center",
        va="top",
        fontsize=8,
        fontstyle="italic",
        color="#555555",
    )


def plot_clusters_per_expert(
    sources: list[tuple[str, Path]],
    output: Path,
    *,
    layer: int | None = None,
    x_max: int = _CLUSTER_X_MAX,
) -> Path:
    """Draw one histogram per ``(label, metrics parquet)``."""
    if not sources:
        msg = "Pass at least one clustering metrics parquet"
        raise ValueError(msg)
    panels: list[tuple[str, np.ndarray]] = []
    for label, metrics in sources:
        counts, chosen = load_clusters_per_expert(metrics, layer=layer)
        panels.append((f"{label} (layer {chosen + 1})", counts))
        logger.info(
            f"{label}: clusters/expert min={counts.min()} "
            f"max={counts.max()} mean={counts.mean():.1f}"
        )
    observed = max(int(np.max(counts)) for _, counts in panels)
    x_max = max(x_max, observed)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 4.2))
    fig.subplots_adjust(bottom=0.20, wspace=0.35)
    axes_list = [axes] if len(panels) == 1 else list(axes)
    for ax, (label, counts), color in zip(
        axes_list, panels, _CLUSTER_PALETTE, strict=False
    ):
        _draw_cluster_histogram(ax, counts, label, color, x_max=x_max)
    return _save(fig, output)


# ---------------------------------------------------------------------------
# Feature-selection sweep
# ---------------------------------------------------------------------------

_SWEEP_STYLE: dict[str, dict[str, str]] = {
    "rho_useful": {"color": "#d6604d", "marker": "s", "label": "ρ-usefulness"},
    "cosine": {"color": "#4dac26", "marker": "^", "label": "cosine"},
    "routing_margin": {
        "color": "#fdae61",
        "marker": "D",
        "label": "gradient attribution",
    },
}
_SWEEP_METHODS = ("rho_useful", "cosine", "routing_margin")
_SWEEP_TICKS = (1, 10, 20, 30, 40, 50, 100, 200)


def plot_feature_selection_sweep(
    metrics: Path,
    output: Path,
    *,
    methods: list[str] | None = None,
) -> Path:
    """Draw effective F1 against features per expert, one panel per layer plus the mean.

    ``metrics`` is the sweep parquet: ``series`` is the selection method, ``x`` is the
    feature count, ``panel`` is the hookpoint layer, and ``metric`` is ``effective_f1``.
    Panel titles use the paper layer, one past the hookpoint index.
    """
    selected = list(methods or _SWEEP_METHODS)
    unknown = sorted(set(selected) - set(_SWEEP_STYLE))
    if unknown:
        msg = f"Unknown selection method(s): {', '.join(unknown)}"
        raise ValueError(msg)
    raw = pd.read_parquet(metrics)
    if not (raw["metric"] == "effective_f1").any():
        msg = f"No effective_f1 rows in {metrics}"
        raise ValueError(msg)
    frame = _take(
        raw, (raw["metric"] == "effective_f1") & raw["series"].isin(selected)
    ).copy()
    frame["method"] = frame["series"].astype(str)
    frame["feature_count"] = frame["x"].astype(int)
    frame["layer"] = frame["panel"].astype(int) + 1
    frame["effective_f1"] = frame["value"].astype(float)
    frame = frame.dropna(subset=["effective_f1", "feature_count", "layer"])
    if frame.empty:
        msg = f"No sweep rows to plot in {metrics}"
        raise ValueError(msg)
    layers = sorted(frame["layer"].unique())
    n_panels = len(layers) + 1
    ncols = 3
    nrows = max(1, (n_panels + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.2 * nrows), sharey=True)
    fig.subplots_adjust(
        left=0.07, right=0.98, top=0.85, bottom=0.10, wspace=0.08, hspace=0.42
    )

    def _draw(ax: Axes, sub: pd.DataFrame) -> None:
        for method in selected:
            style = _SWEEP_STYLE[method]
            grp = sub[sub["method"] == method]
            if grp.empty:
                continue
            stats = grp.groupby("feature_count")["effective_f1"].mean().sort_index()
            ax.plot(
                stats.index.to_numpy(),
                stats.to_numpy(),
                color=style["color"],
                marker=style["marker"],
                label=style["label"],
                linewidth=1.8,
                markersize=5,
            )

    def _style(ax: Axes, title: str) -> None:
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Features per expert", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xticks(list(_SWEEP_TICKS))
        ax.tick_params(axis="x", labelsize=7)

    flat = np.atleast_1d(axes).ravel()
    for ax, layer in zip(flat, layers, strict=False):
        _draw(ax, _take(frame, frame["layer"] == layer))
        _style(ax, f"Layer {int(layer)}")
    mean_ax = flat[len(layers)]
    _draw(mean_ax, frame)
    _style(mean_ax, "Mean (all layers)")
    for ax in flat[len(layers) + 1 :]:
        ax.axis("off")
    for ax in flat[::ncols]:
        if ax.axison:
            ax.set_ylabel("F1", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=max(1, len(handles)),
        fontsize=9,
        frameon=False,
    )
    return _save(fig, output)


# ---------------------------------------------------------------------------
# Pile domains
# ---------------------------------------------------------------------------

_PILE_COLORS = {
    "ArXiv": "#e63946",
    "DM Mathematics": "#1d3557",
    "Enron Emails": "#2a9d8f",
    "EuroParl": "#e9c46a",
    "FreeLaw": "#f4a261",
    "Github": "#264653",
    "Gutenberg (PG-19)": "#a8dadc",
    "HackerNews": "#e76f51",
    "NIH ExPorter": "#6a994e",
    "PhilPapers": "#457b9d",
    "Pile-CC": "#adb5bd",
    "PubMed Abstracts": "#b5838d",
    "PubMed Central": "#e07a5f",
    "StackExchange": "#f2cc8f",
    "USPTO Backgrounds": "#81b29a",
    "Wikipedia (en)": "#3d405b",
}
_PILE_CAP = 4.0
_MEDIAN_COLOR = "#1d4e89"


def plot_pile_distribution(
    metrics: Path,
    output: Path,
    *,
    layers: list[int] | None = None,
    top_subsets: int = 8,
    top_experts: int = 8,
    seed: int = 42,
) -> Path:
    """Draw expert preference for the largest Pile subsets, plus each layer's distribution.

    ``metrics`` is the pile-domains parquet. ``lift`` is within-expert share divided by
    the subset's share of the corpus. Panel titles use the paper layer, one past the
    hookpoint index. With hookpoint layers 3, 11, 15, and 19 the panels are 4, 12, and 20.
    """
    frame, corpus = _pile_frame(metrics)
    subsets = sorted(corpus, key=lambda name: corpus[name], reverse=True)[:top_subsets]
    chosen = _pile_layers(frame, layers)
    experts = _pile_experts(frame, top_experts, seed)
    fig, axes = plt.subplots(
        len(chosen),
        2,
        figsize=(13.6, 3.15 * len(chosen)),
        sharey=True,
        squeeze=False,
        gridspec_kw={"width_ratios": [2.6, 1.0]},
    )
    for row, layer in enumerate(chosen):
        layer_df = _take(frame, frame["layer"] == layer)
        _draw_pile_bars(axes[row, 0], layer_df, subsets, experts, layer)
        _draw_pile_density(
            axes[row, 1], _column(layer_df, "preference").dropna().to_numpy(), layer
        )
    handles: list[Patch | Line2D] = [
        Patch(facecolor=_PILE_COLORS.get(name, "#888888"), edgecolor="none", label=name)
        for name in subsets
    ]
    labels = [f"{name} ({corpus[name] * 100:.1f}% of Pile)" for name in subsets]
    handles.append(Line2D([0], [0], color=_MEDIAN_COLOR, linestyle="--", linewidth=1.4))
    labels.append("median expert-subset preference")
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1), h_pad=2.8)
    return _save(fig, output, dpi=200)


def _pile_frame(metrics: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    """Return preference rows and each subset's share of the corpus."""
    raw = pd.read_parquet(metrics)
    share = cast(
        pd.DataFrame,
        _take(raw, raw["metric"] == "routing_share")[["panel", "series", "x", "value"]],
    ).rename(columns={"value": "routing_share"})
    lift = cast(
        pd.DataFrame,
        _take(raw, raw["metric"] == "lift")[["panel", "series", "x", "value"]],
    ).rename(columns={"value": "preference"})
    merged = share.merge(lift, on=["panel", "series", "x"], how="inner")
    if merged.empty:
        msg = f"No routing preference rows in {metrics}"
        raise ValueError(msg)
    merged["expert"] = merged["panel"].astype(int)
    merged["pile_subset"] = merged["series"].astype(str)
    merged["layer"] = merged["x"].astype(int) + 1
    base = merged["routing_share"].astype(float) / merged["preference"].astype(float)
    corpus = {
        str(subset): float(values.median())
        for subset, values in base.groupby(merged["pile_subset"])
    }
    return merged, corpus


def _pile_layers(frame: pd.DataFrame, layers: list[int] | None) -> list[int]:
    if layers:
        return layers
    recorded = sorted(int(layer) for layer in frame["layer"].unique())
    if len(recorded) > 3:
        recorded = [recorded[0], recorded[1], recorded[-1]]
    if not recorded:
        msg = "No layers to plot"
        raise ValueError(msg)
    return recorded


def _pile_experts(frame: pd.DataFrame, top_experts: int, seed: int) -> list[int]:
    experts = np.arange(int(frame["expert"].max()) + 1)
    if len(experts) <= top_experts:
        return [int(expert) for expert in experts]
    rng = np.random.default_rng(seed)
    return sorted(
        int(expert) for expert in rng.choice(experts, top_experts, replace=False)
    )


def _draw_pile_bars(
    ax: Axes,
    layer_df: pd.DataFrame,
    subsets: list[str],
    experts: list[int],
    layer: int,
) -> None:
    centers = np.arange(len(experts))
    width = 0.72 / max(len(subsets), 1)
    for index, subset in enumerate(subsets):
        heights = []
        for expert_id in experts:
            row = _take(
                layer_df,
                (layer_df["expert"] == expert_id) & (layer_df["pile_subset"] == subset),
            )
            heights.append(
                float(_column(row, "preference").iloc[0]) if len(row) else 0.0
            )
        offset = (index - (len(subsets) - 1) / 2) * width
        color = _PILE_COLORS.get(subset, "#888888")
        xs = centers + offset
        ax.bar(
            xs,
            [min(height, _PILE_CAP) for height in heights],
            width=width * 0.92,
            color=color,
            linewidth=0,
            zorder=3,
        )
        for x_pos, height in zip(xs, heights, strict=True):
            if height > _PILE_CAP:
                ax.annotate(
                    f"{height:.1f}",
                    xy=(x_pos, _PILE_CAP),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=color,
                    annotation_clip=False,
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "facecolor": "white",
                        "edgecolor": color,
                        "linewidth": 0.6,
                    },
                )
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.1, zorder=4)
    ax.set_ylim(0, _PILE_CAP)
    ax.set_xlim(centers[0] - 0.55, centers[-1] + 0.55)
    ax.set_xticks(centers)
    ax.set_xticklabels([str(expert) for expert in experts], fontsize=8)
    ax.set_title(
        f"Selected experts (layer {layer})", fontsize=11, fontweight="bold", pad=14
    )
    ax.set_xlabel("Expert ID", fontsize=9)
    ax.set_ylabel("Expert preference for subset", fontsize=9)
    ax.grid(axis="y", linestyle="--", color="#cccccc", zorder=0)
    ax.set_axisbelow(True)


def _draw_pile_density(ax: Axes, preferences: np.ndarray, layer: int) -> None:
    shown = preferences[np.isfinite(preferences) & (preferences <= _PILE_CAP)]
    above = float(np.mean(preferences > _PILE_CAP) * 100) if len(preferences) else 0.0
    median = float(np.median(preferences)) if len(preferences) else 0.0
    if len(shown) > 1:
        grid = np.linspace(0, _PILE_CAP, 300)
        density = gaussian_kde(shown)(grid)
        ax.fill_betweenx(grid, 0, density, color="#9aa0a6", alpha=0.55, zorder=2)
        ax.plot(density, grid, color="#6d7378", linewidth=0.8, zorder=3)
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.1, zorder=4)
    ax.axhline(median, color=_MEDIAN_COLOR, linestyle="--", linewidth=1.3, zorder=5)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, _PILE_CAP)
    ax.set_title(
        f"All expert×subset pairs (layer {layer})", fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Density", fontsize=9)
    ax.tick_params(axis="y", labelleft=False)
    ax.grid(axis="y", linestyle="--", color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    ax.annotate(
        f"{above:.1f}% samples above cap",
        xy=(0.04, 0.96),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "#eef1f4",
            "edgecolor": "#c5cdd4",
            "linewidth": 0.6,
        },
    )
    ax.annotate(
        f"{median:.2f}",
        xy=(0, median),
        xytext=(-8, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=9,
        color=_MEDIAN_COLOR,
        fontweight="bold",
        annotation_clip=False,
    )


# ---------------------------------------------------------------------------
# Explanation length
# ---------------------------------------------------------------------------


def _length_rows(evaluations: list[Path]) -> list[dict[str, Any]]:
    """Join each evaluate directory to its explanation text."""
    rows: list[dict[str, Any]] = []
    for evaluate_dir in evaluations:
        name = evaluate_dir.name
        if not name.startswith("n") or not name[1:].isdigit():
            msg = f"Evaluate directory must be named n<count>, got {evaluate_dir}"
            raise ValueError(msg)
        count = int(name[1:])
        scores = pd.read_parquet(evaluate_dir / "scores.parquet")
        explain_path = (
            evaluate_dir.parent.parent / "explain" / name / "explanations.jsonl"
        )
        if not explain_path.is_file():
            msg = f"Missing explanations file: {explain_path}"
            raise FileNotFoundError(msg)
        by_key = {
            (
                str(record["hookpoint"]),
                int(record["target_id"]),
                str(record["target_kind"]),
            ): record
            for record in read_jsonl(explain_path)
        }
        seen: set[tuple[str, int, str]] = set()
        for record in scores.to_dict("records"):
            if str(record["target_kind"]) == "all" or int(record["target_id"]) < 0:
                continue
            key = (
                str(record["hookpoint"]),
                int(record["target_id"]),
                str(record["target_kind"]),
            )
            if key in seen:
                continue
            seen.add(key)
            explanation = by_key.get(key)
            if explanation is None:
                continue
            n_tokens = explanation.get("n_tokens")
            if n_tokens is None:
                n_tokens = max(1, len(str(explanation.get("text") or "").split()))
            rows.append(
                {
                    "f": count,
                    "layer": parse_layer_index(str(record["hookpoint"])),
                    "f1": float(record["f1"]),
                    "n_tokens": int(n_tokens),
                }
            )
    if not rows:
        msg = "No explanation scores joined to explanation text"
        raise ValueError(msg)
    return rows


def plot_explanation_length(evaluations: list[Path], output: Path) -> Path:
    """Draw explanation F1 against explanation length.

    Each path is an evaluate directory named ``n<count>`` with ``scores.parquet``.
    The matching explanations are ``explain/n<count>/explanations.jsonl`` beside it.
    """
    rows = _length_rows(evaluations)
    counts = tuple(sorted({int(row["f"]) for row in rows}))
    layers = tuple(sorted({int(row["layer"]) for row in rows}))
    colors: dict[int, Any] = dict(
        zip(
            counts,
            plt.cm.Blues(np.linspace(0.4, 1.0, len(counts))),
            strict=True,
        )
    )
    background_tokens: list[float] = []
    background_f1: list[float] = []
    averages: list[tuple[int, float, float, float]] = []
    for count in counts:
        pooled = np.array(
            [float(row["f1"]) for row in rows if int(row["f"]) == count],
            dtype=float,
        )
        layer_tokens: list[float] = []
        layer_f1: list[float] = []
        for layer in layers:
            matched = [
                row
                for row in rows
                if int(row["f"]) == count and int(row["layer"]) == layer
            ]
            if not matched:
                continue
            layer_tokens.append(float(np.mean([row["n_tokens"] for row in matched])))
            layer_f1.append(float(np.mean([row["f1"] for row in matched])))
        background_tokens.extend(layer_tokens)
        background_f1.extend(layer_f1)
        if pooled.size == 0 or not layer_tokens:
            continue
        spread = float(pooled.std(ddof=1)) if pooled.size > 1 else 0.0
        averages.append(
            (
                count,
                float(np.mean(layer_tokens)),
                float(np.mean(layer_f1)),
                1.96 * spread / np.sqrt(pooled.size),
            )
        )
    if not averages:
        msg = "No explanation-length rows to plot"
        raise ValueError(msg)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(
        background_tokens,
        background_f1,
        s=22,
        color="0.75",
        alpha=0.8,
        linewidths=0,
        zorder=1,
        label="per-layer mean",
    )
    for count, tokens, mean_f1, ci in averages:
        ax.errorbar(
            [tokens],
            [mean_f1],
            yerr=[[ci], [ci]],
            fmt="none",
            ecolor=colors[count],
            lw=1.2,
            capsize=3,
            zorder=2,
        )
        ax.scatter(
            [tokens],
            [mean_f1],
            s=80,
            facecolor=colors[count],
            edgecolor="white",
            linewidths=1.0,
            zorder=3,
        )
    ax.set_xlabel("Explanation length (tokens)")
    ax.set_ylabel("Explanation score (F1)")
    ax.set_xlim(200, 620)
    ax.set_ylim(0.25, 0.60)
    ax.grid(alpha=0.25)
    feature_legend = ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                lw=0,
                mfc=colors[count],
                mec="white",
                ms=7,
                label=str(count),
            )
            for count in counts
            if count in colors
        ],
        fontsize=8,
        loc="lower right",
        title="# SAE features",
        title_fontsize=8,
        framealpha=0.95,
    )
    ax.add_artist(feature_legend)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
    fig.tight_layout()
    return _save(fig, output, dpi=180)


# ---------------------------------------------------------------------------
# Explanation F1 by depth
# ---------------------------------------------------------------------------

_SCORE_UNIGRAM = "#8A9099"
_SCORE_EXPERT_IMPACT = "#4A90A4"
_SCORE_ROUTERINTERP = "#E07B54"
_SCORE_BASELINE = "#666666"
_ALWAYS_POSITIVE = 0.261

_SCORE_METHODS = (
    ("unigram", "Unigram Lookup", _SCORE_UNIGRAM),
    ("expert_impact", "Expert Impact AutoInterp", _SCORE_EXPERT_IMPACT),
    ("routerinterp", "RouterInterp (ours)", _SCORE_ROUTERINTERP),
)

_SCORE_DIRS = (
    (("unigram_lookup",), "unigram"),
    (("esb",), "expert_impact"),
    (("routerinterp",), "routerinterp"),
)
_DEPTH_LAYERS = {
    "gpt-oss-20b": {"Early": 3, "Mid": 15, "Late": 19},
    "OLMoE-1B-7B": {"Early": 2, "Mid": 10, "Late": 14},
}


def _macro_by_layer(scores_dir: Path) -> dict[int, float]:
    """Mean F1 of the per-hookpoint macro rows."""
    scores = pd.read_parquet(scores_dir / "scores.parquet")
    rows = _take(
        scores,
        (scores["target_kind"].astype(str) == "all")
        & (scores["hookpoint"].astype(str) != ""),
    )
    grouped: dict[int, list[float]] = {}
    for record in rows.to_dict("records"):
        layer = parse_layer_index(str(record["hookpoint"]))
        grouped.setdefault(layer, []).append(float(record["f1"]))
    if not grouped:
        msg = f"No per-layer macro F1 in {scores_dir / 'scores.parquet'}"
        raise ValueError(msg)
    return {layer: float(np.mean(values)) for layer, values in grouped.items()}


def _score_table(
    models: list[tuple[str, Path]],
) -> dict[str, dict[str, dict[int, float]]]:
    table: dict[str, dict[str, dict[int, float]]] = {}
    for label, evaluate_dir in models:
        table[label] = {}
        for dir_names, method in _SCORE_DIRS:
            found = next(
                (
                    evaluate_dir / name
                    for name in dir_names
                    if (evaluate_dir / name / "scores.parquet").is_file()
                ),
                None,
            )
            if found is None:
                msg = f"No {method} scores under {evaluate_dir}"
                raise FileNotFoundError(msg)
            table[label][method] = _macro_by_layer(found)
    return table


def _depth_layer(model: str, depth: str, layers: set[int]) -> int:
    known = _DEPTH_LAYERS.get(model, {})
    if depth in known and known[depth] in layers:
        return int(known[depth])
    ordered = sorted(layers)
    if depth == "Early":
        return ordered[0]
    if depth == "Late":
        return ordered[-1]
    return ordered[len(ordered) // 2]


def _f1_at(
    table: dict[str, dict[str, dict[int, float]]], model: str, method: str, layer: int
) -> float:
    return float(table[model][method][layer])


def plot_explanation_scores(models: list[tuple[str, Path]], output: Path) -> Path:
    """Draw early, mid, and late explanation F1 for each model.

    ``models`` is ``(label, evaluate_dir)``. ``evaluate_dir`` contains one subdirectory
    per explanation method, each with ``scores.parquet``.
    """
    table = _score_table(models)
    model_names = tuple(label for label, _ in models)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.linewidth": 1.0,
        }
    )
    depths = ("Early", "Mid", "Late")
    fig, ax = plt.subplots(figsize=(7.8, 3.5))
    bar_w = 0.15
    model_group_w = len(_SCORE_METHODS) * bar_w
    model_gap = 0.11
    n_models = len(model_names)
    cluster_w = n_models * model_group_w + max(n_models - 1, 0) * model_gap
    depth_gap = 0.42
    depth_centers = np.arange(len(depths)) * (cluster_w + depth_gap)
    for depth_index, depth in enumerate(depths):
        cluster_left = depth_centers[depth_index] - cluster_w / 2
        for model_index, model in enumerate(model_names):
            group_left = cluster_left + model_index * (model_group_w + model_gap)
            present = set().union(
                *(table[model][method] for method, _, _ in _SCORE_METHODS)
            )
            layer = _depth_layer(model, depth, present)
            for method_index, (method, _name, color) in enumerate(_SCORE_METHODS):
                if layer not in table[model][method]:
                    continue
                x = group_left + method_index * bar_w + bar_w / 2
                value = _f1_at(table, model, method, layer)
                ax.bar(
                    x,
                    value,
                    bar_w,
                    color=color,
                    edgecolor="black",
                    linewidth=0.65,
                    zorder=3,
                )
                ax.text(
                    x,
                    value + 0.01,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color="#333333",
                )
            ax.text(
                group_left + model_group_w / 2,
                -0.045,
                model,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                color="#333333",
            )
    ax.set_ylim(0, 0.78)
    ax.set_yticks(np.arange(0, 0.79, 0.1))
    ax.grid(True, axis="y", linestyle="--", alpha=0.35, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(
        _ALWAYS_POSITIVE,
        color=_SCORE_BASELINE,
        linestyle=":",
        linewidth=1.15,
        alpha=0.8,
        zorder=1,
    )
    ax.set_xticks(depth_centers)
    ax.set_xticklabels(list(depths), fontweight="bold")
    ax.tick_params(axis="x", pad=18)
    ax.set_ylabel("Explanation F1", fontweight="bold")
    ax.set_xlim(
        depth_centers[0] - cluster_w / 2 - 0.1,
        depth_centers[-1] + cluster_w / 2 + 0.1,
    )
    handles: list[Patch | Line2D] = [
        Patch(facecolor=color, edgecolor="black", label=name)
        for _method, name, color in _SCORE_METHODS
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color=_SCORE_BASELINE,
            linestyle=":",
            linewidth=1.15,
            label=f"Always-positive ({_ALWAYS_POSITIVE})",
        )
    )
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        borderaxespad=0.0,
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
        framealpha=0.95,
        borderpad=0.4,
        handlelength=1.2,
        ncol=2,
        columnspacing=1.0,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, output, dpi=300)


def _save(fig: Figure, output: Path, *, dpi: int = 150) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Wrote {output}")
    return output
