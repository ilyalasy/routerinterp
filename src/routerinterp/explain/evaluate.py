"""Score explanations on held-out windows that were not shown when the explanation was
written.

Finished targets are appended to ``eval_progress.jsonl``.
A rerun skips those and does not call the scorer again.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from loguru import logger

from routerinterp.artifacts import (
    Corpus,
    SparseActivations,
    collect_expert_activations,
    collect_feature_activations,
    load_corpus,
    load_tokenizer,
)
from routerinterp.core import (
    EvaluateConfig,
    ExplainConfig,
    append_jsonl,
    build_manifest,
    read_manifest,
    read_parquet,
    write_manifest,
    write_parquet,
)
from routerinterp.explain.explain import (
    EXPLANATIONS_FILENAME,
    LOOKUP_METHODS,
    explain_window_ids,
    load_explanations,
)
from routerinterp.explain.openrouter import OpenRouterChatClient
from routerinterp.explain.prompts import (
    build_scoring_messages,
    ordinal_parse_retry_message,
)
from routerinterp.explain.sampling import build_eval_windows, entity_seed
from routerinterp.explain.schema import (
    PRF1,
    EvalWindow,
    ExplanationKey,
    ShownWindowIds,
    TargetKey,
)
from routerinterp.explain.utils import (
    binarise_scores,
    compute_prf1,
    parse_ordinal_scores,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# Scorer temperature and parse retries are implementation defaults.
# Paper knobs (eval counts, threshold, scorer model) live in EvaluateConfig.
SCORER_TEMPERATURE = 0.2
PARSE_RETRIES = 2

# ``lookup`` matches an n-gram table. ``llm0-4`` is the graded context-match
# scorer (0-4, binarised at ``threshold``). The model name is ``scorer_model``.
ScorerType = Literal["lookup", "llm0-4"]

SCORES_FILENAME = "scores.parquet"
PREDICTIONS_FILENAME = "predictions.parquet"
EVAL_PROGRESS_FILENAME = "eval_progress.jsonl"


@dataclass(frozen=True, slots=True)
class BatchGrades:
    """Grades and request ids from one scoring call."""

    grades: list[int]
    request_ids: list[str]


@dataclass(frozen=True, slots=True)
class WindowGrades:
    """Yes/no predictions, the raw 0-4 grades, and every scoring request id."""

    predictions: list[bool]
    grades: list[int]
    request_ids: list[str]


@dataclass(frozen=True, slots=True)
class TargetScore:
    """One scorer's output for a single target."""

    scorer: ScorerType
    predictions: list[bool]
    raw_scores: list[int]
    threshold: int
    scorer_model: str
    request_ids: list[str]


@dataclass(frozen=True, slots=True)
class EvalRows:
    """Score rows and per-window prediction rows for one target."""

    scores: list[dict[str, Any]]
    predictions: list[dict[str, Any]]


@dataclass(frozen=True, slots=True, order=True)
class ScoreGroup:
    """Rows averaged together for one method, hookpoint, and scorer."""

    method: str
    hookpoint: str
    scorer: str


@dataclass(frozen=True, slots=True, order=True)
class MethodScorer:
    """Rows averaged across hookpoints for one method and scorer."""

    method: str
    scorer: str


type Fires = dict[TargetKey, SparseActivations]
type ProgressByTarget = dict[ExplanationKey, dict[str, Any]]


def lookup_predict(
    tokens: list[int],
    positions: set[int],
    ngrams: list[list[int]],
    unigrams: list[list[int]] | None,
    *,
    bigram: bool,
) -> bool:
    """Return whether any highlighted token is in the expert's n-gram table.

    For unigrams, the highlighted token itself must be listed.
    For bigrams, the previous token and the highlighted token are checked as a pair, and
    a miss falls back to the highlighted token alone.
    """
    table = {tuple(ngram) for ngram in ngrams}
    uni = {tuple(ngram) for ngram in unigrams} if unigrams else set()
    for i in positions:
        if not bigram:
            if (tokens[i],) in table:
                return True
            continue
        if i > 0 and (tokens[i - 1], tokens[i]) in table:
            return True
        if uni and (tokens[i],) in uni:
            return True
    return False


# ---------------------------------------------------------------------------
# LLM 0-4 graded context-match scorer
# ---------------------------------------------------------------------------


async def score_windows_llm(
    client: OpenRouterChatClient,
    explanation: str,
    windows: list[EvalWindow],
    *,
    batch_size: int,
    threshold: int,
    semaphore: asyncio.Semaphore,
) -> WindowGrades:
    """Ask the scorer to grade each window from 0 to 4.

    Returns the yes/no predictions, the raw grades, and the id of every scoring call,
    including retries after a bad parse.
    An empty explanation is treated as all negative, and the scorer is not called.
    """
    n = len(windows)
    if not explanation.strip():
        return WindowGrades([False] * n, [-1] * n, [])
    batches = [windows[i : i + batch_size] for i in range(0, n, batch_size)]

    async def _score_batch(batch: list[EvalWindow]) -> BatchGrades:
        async with semaphore:
            messages = build_scoring_messages(explanation, batch)
            last_error: BaseException | None = None
            request_ids: list[str] = []
            for attempt in range(PARSE_RETRIES + 1):
                attempt_messages = messages
                if attempt > 0:
                    attempt_messages = [
                        *messages,
                        ordinal_parse_retry_message(len(batch)),
                    ]
                try:
                    result = await client.generate(
                        attempt_messages, temperature=SCORER_TEMPERATURE
                    )
                    if result.request_id:
                        request_ids.append(result.request_id)
                    return BatchGrades(
                        parse_ordinal_scores(result.text, len(batch)), request_ids
                    )
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        f"Scoring batch failed (attempt {attempt + 1}/"
                        f"{PARSE_RETRIES + 1}): {exc!r}"
                    )
            msg = f"Could not score batch of {len(batch)} after retries"
            raise RuntimeError(msg) from last_error

    batch_results = await asyncio.gather(*[_score_batch(b) for b in batches])
    scores = [score for batch in batch_results for score in batch.grades]
    request_ids = [rid for batch in batch_results for rid in batch.request_ids]
    return WindowGrades(binarise_scores(scores, threshold), scores, request_ids)


def score_row(
    *,
    method: str,
    hookpoint: str,
    target_kind: str,
    target_id: int,
    scorer: ScorerType,
    threshold: int,
    metrics: PRF1,
    n_pos: int,
    n_neg: int,
    scorer_model: str,
    request_ids: list[str],
) -> dict[str, Any]:
    """Build one row of the scores table."""
    return {
        "method": method,
        "hookpoint": hookpoint,
        "target_kind": target_kind,
        "target_id": target_id,
        "scorer": scorer,
        "threshold": threshold,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "scorer_model": scorer_model,
        "request_ids": request_ids,
    }


def _progress_key(row: dict[str, Any]) -> ExplanationKey:
    return ExplanationKey.from_record(row)


def _read_progress_records(path: Path) -> list[dict[str, Any]]:
    """Read checkpoint lines.
    A truncated final line is ignored.
    """
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            logger.warning(f"Ignoring truncated checkpoint line in {path}")
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def load_eval_progress(path: Path) -> ProgressByTarget:
    """Return the latest checkpoint record for each target.
    Missing file is an empty run.
    """
    latest: ProgressByTarget = {}
    for record in _read_progress_records(path):
        scores = record.get("scores") or []
        if not scores:
            continue
        latest[_progress_key(scores[0])] = record
    return latest


def _seed_progress_from_parquet(output_dir: Path) -> None:
    """Backfill the checkpoint from a finished parquet written before resume."""
    progress_path = output_dir / EVAL_PROGRESS_FILENAME
    scores_path = output_dir / SCORES_FILENAME
    if progress_path.exists() or not scores_path.exists():
        return
    predictions: dict[ExplanationKey, list[dict[str, Any]]] = {}
    preds_path = output_dir / PREDICTIONS_FILENAME
    if preds_path.exists():
        for row in read_parquet(preds_path).to_pylist():
            if str(row.get("target_kind")) == "all":
                continue
            predictions.setdefault(_progress_key(row), []).append(row)
    grouped: dict[ExplanationKey, dict[str, Any]] = {}
    for row in read_parquet(scores_path).to_pylist():
        if str(row.get("target_kind")) == "all" or int(row.get("target_id", -1)) == -1:
            continue
        key = _progress_key(row)
        bucket = grouped.setdefault(
            key, {"scores": [], "predictions": predictions.get(key, [])}
        )
        bucket["scores"].append(row)
    if grouped:
        append_jsonl(progress_path, list(grouped.values()))


def macro_average_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean precision, recall, and F1 for each hookpoint, and once more across hookpoints.

    Those rows use ``target_kind="all"`` and ``target_id=-1``.
    The row across hookpoints has an empty hookpoint.
    ``n_pos`` and ``n_neg`` are sums rather than means.
    """
    macro: list[dict[str, Any]] = []
    groups: dict[ScoreGroup, list[dict[str, Any]]] = {}
    for row in rows:
        key = ScoreGroup(str(row["method"]), str(row["hookpoint"]), str(row["scorer"]))
        groups.setdefault(key, []).append(row)

    def _mean(group: list[dict[str, Any]], field: str) -> float:
        return float(np.mean([float(r[field]) for r in group]))

    def _macro_row(group: list[dict[str, Any]], *, hookpoint: str) -> dict[str, Any]:
        first = group[0]
        return {
            "method": first["method"],
            "hookpoint": hookpoint,
            "target_kind": "all",
            "target_id": -1,
            "scorer": first["scorer"],
            "threshold": first["threshold"],
            "precision": _mean(group, "precision"),
            "recall": _mean(group, "recall"),
            "f1": _mean(group, "f1"),
            "n_pos": sum(int(r["n_pos"]) for r in group),
            "n_neg": sum(int(r["n_neg"]) for r in group),
            "scorer_model": first["scorer_model"],
            "request_ids": [],
        }

    for group_key, group in sorted(groups.items()):
        macro.append(_macro_row(group, hookpoint=group_key.hookpoint))
    global_groups: dict[MethodScorer, list[dict[str, Any]]] = {}
    for row in rows:
        global_groups.setdefault(
            MethodScorer(str(row["method"]), str(row["scorer"])), []
        ).append(row)
    for _group_key, group in sorted(global_groups.items()):
        macro.append(_macro_row(group, hookpoint=""))
    return macro


_ACTIVATION_COLLECTORS = {
    "feature": collect_feature_activations,
    "expert": collect_expert_activations,
}


def _collect_fires(
    input_dir: Path,
    pending: dict[ExplanationKey, dict[str, Any]],
    *,
    expected_tokens: int,
) -> Fires:
    """Load firing positions only for targets that are still unscored.

    Feature targets need the SAE latent files.
    Expert targets need the router top-k files.
    The other set is not read.
    """
    ids_by_kind: dict[str, dict[str, list[int]]] = {}
    for key in pending:
        if key.target_kind not in _ACTIVATION_COLLECTORS:
            msg = f"Unsupported target_kind {key.target_kind!r} in explanations"
            raise ValueError(msg)
        ids_by_kind.setdefault(key.target_kind, {}).setdefault(
            key.hookpoint, []
        ).append(key.target_id)
    fires: Fires = {}
    for target_kind, by_hookpoint in ids_by_kind.items():
        collect = _ACTIVATION_COLLECTORS[target_kind]
        for hookpoint, target_ids in by_hookpoint.items():
            collected = collect(
                input_dir,
                hookpoint,
                target_ids,
                expected_tokens=expected_tokens,
            )
            for target_id, idx_val in collected.items():
                fires[TargetKey(hookpoint, target_kind, target_id)] = idx_val
    return fires


@dataclass
class _EvalContext:
    """Shared state for scoring the pending targets of one explain run."""

    config: EvaluateConfig
    explain_cfg: ExplainConfig
    method: str
    is_lookup: bool
    fires: Fires
    used_ids: ShownWindowIds
    corpus: Corpus
    tokenizer: PreTrainedTokenizerBase
    client: OpenRouterChatClient | None
    semaphore: asyncio.Semaphore
    save_lock: asyncio.Lock
    progress_path: Path
    stats: dict[str, int]


async def _eval_target(
    ctx: _EvalContext,
    key: ExplanationKey,
    record: dict[str, Any],
) -> EvalRows:
    """Score one target and append its checkpoint line.
    Failures return empty.
    """
    hookpoint = key.hookpoint
    target_kind = key.target_kind
    target_id = key.target_id
    text = str(record.get("text") or "")
    acts = ctx.fires.get(key.target)
    if acts is None:
        logger.error(f"No activations collected for {key}; skipping")
        ctx.stats["failed"] += 1
        return EvalRows([], [])
    indices = acts.indices
    values = acts.values
    seed = entity_seed(
        ctx.config.scoring_seed, target_kind, "evaluation", hookpoint, target_id
    )
    highlight_seed = entity_seed(
        ctx.config.scoring_seed, target_kind, "highlight", hookpoint, target_id
    )
    exclude = ctx.used_ids.get(key.target, set())
    try:
        windows = build_eval_windows(
            ctx.corpus,
            ctx.tokenizer,
            indices,
            values,
            ctx_len=ctx.explain_cfg.ctx_len,
            n_positives=ctx.config.eval_positives,
            n_negatives=ctx.config.eval_negatives,
            seed=seed,
            highlight_seed=highlight_seed,
            exclude_window_ids=exclude,
        )
    except Exception as exc:
        logger.error(f"Eval window sampling failed for {key}: {exc!r}")
        ctx.stats["failed"] += 1
        return EvalRows([], [])
    n_pos = sum(1 for w in windows if w.ground_truth)
    n_neg = len(windows) - n_pos
    if n_pos == 0 or n_neg == 0:
        logger.warning(
            f"Skipping {key}: held-out windows degenerate (pos={n_pos} neg={n_neg})"
        )
        ctx.stats["skipped"] += 1
        return EvalRows([], [])

    ground_truths = [w.ground_truth for w in windows]
    scored: list[TargetScore]
    if ctx.is_lookup:
        ngrams = record.get("lookup_ngrams") or []
        unigrams = record.get("lookup_unigrams")
        preds = [
            lookup_predict(
                w.tokens,
                w.highlight,
                ngrams,
                unigrams,
                bigram=ctx.method == "bigram_lookup",
            )
            for w in windows
        ]
        scored = [
            TargetScore("lookup", preds, [-1] * len(windows), -1, "", []),
        ]
    else:
        assert ctx.client is not None
        try:
            grades = await score_windows_llm(
                ctx.client,
                text,
                windows,
                batch_size=ctx.config.scorer_batch_size,
                threshold=ctx.config.threshold,
                semaphore=ctx.semaphore,
            )
        except Exception as exc:
            logger.error(f"LLM scoring failed for {key}: {exc!r}")
            ctx.stats["failed"] += 1
            return EvalRows([], [])
        scored = [
            TargetScore(
                "llm0-4",
                grades.predictions,
                grades.grades,
                ctx.config.threshold,
                ctx.config.scorer_model,
                grades.request_ids,
            ),
        ]

    rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    for item in scored:
        metrics = compute_prf1(ground_truths, item.predictions)
        rows.append(
            score_row(
                method=ctx.method,
                hookpoint=hookpoint,
                target_kind=target_kind,
                target_id=target_id,
                scorer=item.scorer,
                threshold=item.threshold,
                metrics=metrics,
                n_pos=n_pos,
                n_neg=n_neg,
                scorer_model=item.scorer_model,
                request_ids=item.request_ids,
            )
        )
        pred_rows.extend(
            {
                "method": ctx.method,
                "hookpoint": hookpoint,
                "target_kind": target_kind,
                "target_id": target_id,
                "scorer": item.scorer,
                "window_id": w.window_id,
                "ground_truth": w.ground_truth,
                "prediction": bool(pred),
                "raw_score": int(raw),
            }
            for w, pred, raw in zip(
                windows, item.predictions, item.raw_scores, strict=True
            )
        )
    async with ctx.save_lock:
        await asyncio.to_thread(
            append_jsonl,
            ctx.progress_path,
            [{"scores": rows, "predictions": pred_rows}],
        )
    ctx.stats["scored"] += 1
    return EvalRows(rows, pred_rows)


async def run_evaluate_async(config: EvaluateConfig) -> Path:
    """Score every explanation in the explain run.

    Targets already recorded in ``eval_progress.jsonl`` are skipped.
    """
    if not 1 <= config.threshold <= 4:
        msg = f"threshold must be between 1 and 4, got {config.threshold}"
        raise ValueError(msg)
    explain_manifest = read_manifest(config.explanations_dir)
    if explain_manifest.stage != "explain":
        msg = (
            f"{config.explanations_dir} is a {explain_manifest.stage!r} run, "
            "expected an explain run"
        )
        raise ValueError(msg)
    explain_cfg = ExplainConfig.model_validate(explain_manifest.resolved_config)
    method = explain_cfg.method
    if config.scoring_seed == explain_cfg.example_seed:
        logger.warning(
            "scoring_seed equals the explain example_seed; disjointness is still "
            "enforced via the test split + explicit window-id exclusion"
        )

    explanations = load_explanations(config.explanations_dir / EXPLANATIONS_FILENAME)
    targets = {key: rec for key, rec in explanations.items() if key.method == method}
    if not targets:
        msg = (
            f"No explanations for method {method!r} in "
            f"{config.explanations_dir / EXPLANATIONS_FILENAME}"
        )
        raise FileNotFoundError(msg)
    used_ids = explain_window_ids(config.explanations_dir)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = config.output_dir / EVAL_PROGRESS_FILENAME
    _seed_progress_from_parquet(config.output_dir)
    done = load_eval_progress(progress_path)
    pending = {key: record for key, record in targets.items() if key not in done}
    if done:
        logger.info(f"Resuming: {len(done)} evaluations already present")

    is_lookup = method in LOOKUP_METHODS
    client = (
        None
        if is_lookup or not pending
        else OpenRouterChatClient(config.scorer_model, temperature=SCORER_TEMPERATURE)
    )
    sem = asyncio.Semaphore(config.max_concurrent)
    save_lock = asyncio.Lock()
    stats = {"scored": 0, "skipped": 0, "failed": 0, "resumed": len(done)}

    ctx: _EvalContext | None = None
    if pending:
        collect_manifest = read_manifest(explain_cfg.input_dir)
        corpus = load_corpus(explain_cfg.input_dir, collect_manifest)
        if corpus.seq_len % explain_cfg.ctx_len != 0:
            msg = (
                f"seq_len {corpus.seq_len} is not a multiple of ctx_len "
                f"{explain_cfg.ctx_len}; windows would cross document boundaries"
            )
            raise ValueError(msg)
        ctx = _EvalContext(
            config=config,
            explain_cfg=explain_cfg,
            method=method,
            is_lookup=is_lookup,
            fires=_collect_fires(
                explain_cfg.input_dir,
                pending,
                expected_tokens=corpus.n_tokens,
            ),
            used_ids=used_ids,
            corpus=corpus,
            tokenizer=load_tokenizer(collect_manifest),
            client=client,
            semaphore=sem,
            save_lock=save_lock,
            progress_path=progress_path,
            stats=stats,
        )

    try:
        from tqdm.asyncio import tqdm as atqdm

        if ctx is not None:
            await atqdm.gather(
                *[
                    _eval_target(ctx, key, record)
                    for key, record in sorted(pending.items())
                ],
                desc=f"Evaluate {method}",
            )
        finished = load_eval_progress(progress_path)
        target_rows = [row for item in finished.values() for row in item["scores"]]
        prediction_rows = [
            row for item in finished.values() for row in item["predictions"]
        ]
        logger.info(
            f"Evaluate stage done: scored={stats['scored']} "
            f"resumed={stats['resumed']} skipped={stats['skipped']} "
            f"failed={stats['failed']}"
        )

        if stats["failed"]:
            msg = f"Evaluate stage failed for {stats['failed']} targets"
            raise RuntimeError(msg)

        all_score_rows = target_rows + macro_average_rows(target_rows)
        if all_score_rows:
            write_parquet(config.output_dir / SCORES_FILENAME, all_score_rows)
        if prediction_rows:
            write_parquet(config.output_dir / PREDICTIONS_FILENAME, prediction_rows)
        logger.info(
            f"Wrote {len(all_score_rows)} score rows "
            f"({len(target_rows)} targets + macro) and "
            f"{len(prediction_rows)} predictions to {config.output_dir}"
        )

        manifest = build_manifest(
            stage="evaluate",
            run_name=f"evaluate-{method}",
            config=config,
            parents={"explain": explain_manifest.config_hash},
        )
        write_manifest(config.output_dir, manifest)
        logger.info(f"Wrote evaluate manifest to {config.output_dir}")
        return config.output_dir
    finally:
        if client is not None:
            await client.aclose()


def run_evaluate(config: EvaluateConfig) -> Path:
    """Run evaluation synchronously for the CLI and the experiment jobs."""
    return asyncio.run(run_evaluate_async(config))
