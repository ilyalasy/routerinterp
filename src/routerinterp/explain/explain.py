"""Expert/feature explainers: methods, task builders, and the run entrypoint."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from routerinterp.artifacts import (
    collect_expert_activations,
    collect_feature_activations,
    load_corpus,
    load_expert_writes,
    load_tokenizer,
)
from routerinterp.core import (
    ExplainConfig,
    NgramType,
    append_jsonl,
    build_manifest,
    hookpoint_slug,
    read_jsonl,
    read_manifest,
    read_parquet,
    stable_hash,
    write_manifest,
    write_parquet,
)
from routerinterp.explain.openrouter import OpenRouterChatClient
from routerinterp.explain.prompts import (
    NGRAM_AUTOINTERP_SHOWN,
    NGRAM_LOOKUP_K,
    build_esb_messages,
    build_expert_activation_messages,
    build_feature_messages,
    build_ngram_messages,
    build_routerinterp_messages,
)
from routerinterp.explain.sampling import (
    sample_esb_examples,
    sample_expert_examples,
    sample_feature_examples,
    sample_routerinterp_examples,
)
from routerinterp.explain.schema import (
    CachedExplanation,
    Example,
    ExplainerBuilder,
    ExplanationKey,
    Explanations,
    ExplanationTask,
    NgramEntry,
    PipelineContext,
    ShownWindowIds,
    TargetKey,
)
from routerinterp.explain.utils import decode_tokens, split_examples
from routerinterp.select import Selection

EXPLANATIONS_FILENAME = "explanations.jsonl"
EXAMPLES_DIRNAME = "examples"


@dataclass(frozen=True, slots=True)
class ExpertNgrams:
    """Bigrams and unigrams counted where one expert was routed."""

    bigrams: list[NgramEntry]
    unigrams: list[NgramEntry]


def build_feature_autointerp_tasks(ctx: PipelineContext) -> list[ExplanationTask]:
    """Build feature AutoInterp tasks from each feature's most active windows."""
    cfg = ctx.config
    tasks: list[ExplanationTask] = []
    for hookpoint in cfg.hookpoints:
        feature_ids = ctx.selection.feature_union(hookpoint)
        if not feature_ids:
            logger.warning(f"No selected features for {hookpoint}; skipping")
            continue
        fires = collect_feature_activations(
            cfg.input_dir, hookpoint, feature_ids, expected_tokens=ctx.corpus.n_tokens
        )
        for feature_id in feature_ids:
            acts = fires[feature_id]
            sampled = sample_feature_examples(
                ctx, hookpoint, feature_id, acts.indices, acts.values
            )
            if sampled is None:
                continue
            examples = sampled.examples
            max_activation = sampled.max_activation
            split = split_examples(examples)
            positives = split.positives
            negatives = split.negatives
            messages = build_feature_messages(
                positives,
                negatives,
                tokenizer=ctx.tokenizer,
                max_activation=max_activation,
            )
            tasks.append(
                ExplanationTask(
                    method=cfg.method,
                    hookpoint=hookpoint,
                    target_kind="feature",
                    target_id=feature_id,
                    messages=messages,
                    examples=examples,
                )
            )
    return tasks


def build_routerinterp_tasks(ctx: PipelineContext) -> list[ExplanationTask]:
    """Build one explanation task per expert from the features selected for that expert.

    The examples are the windows sampled for those features: ones where the feature
    activates and the expert is routed, and ones where it activates and the expert is
    not.
    """
    cfg = ctx.config
    tasks: list[ExplanationTask] = []
    for hookpoint in cfg.hookpoints:
        expert_ids = ctx.selection.experts(hookpoint)
        if not expert_ids:
            logger.warning(f"No selected experts for {hookpoint}; skipping")
            continue
        feature_ids = ctx.selection.feature_union(hookpoint)
        feature_fires = collect_feature_activations(
            cfg.input_dir, hookpoint, feature_ids, expected_tokens=ctx.corpus.n_tokens
        )
        expert_fires = collect_expert_activations(
            cfg.input_dir, hookpoint, expert_ids, expected_tokens=ctx.corpus.n_tokens
        )
        for expert_id in expert_ids:
            selected = ctx.selection.features_for_expert(hookpoint, expert_id)
            if not selected:
                logger.warning(f"No selected features for {hookpoint}/{expert_id}")
                continue
            indices = expert_fires[expert_id].indices
            route_mask = np.zeros(ctx.corpus.n_tokens, dtype=bool)
            route_mask[indices] = True
            examples = sample_routerinterp_examples(
                ctx, hookpoint, expert_id, selected, feature_fires, route_mask
            )
            if not examples:
                logger.warning(
                    f"No usable feature-expert examples for {hookpoint}/{expert_id}"
                )
                continue
            messages = build_routerinterp_messages(examples, tokenizer=ctx.tokenizer)
            tasks.append(
                ExplanationTask(
                    method=cfg.method,
                    hookpoint=hookpoint,
                    target_kind="expert",
                    target_id=expert_id,
                    messages=messages,
                    examples=examples,
                )
            )
    return tasks


def count_expert_ngrams(
    fire_indices: np.ndarray,
    flat_tokens: np.ndarray,
    *,
    seq_len: int,
    top_k: int,
) -> ExpertNgrams:
    """Count how often each token and token pair occurs where the router selects the
    expert.

    A pair is skipped when the expert is selected on the first token of a sequence,
    because there is no previous token in that sequence.
    """
    if fire_indices.size == 0:
        return ExpertNgrams([], [])
    vocab = int(flat_tokens.max()) + 1
    cur = flat_tokens[fire_indices].astype(np.int64)

    uni_counts = np.bincount(cur, minlength=vocab)
    uni_order = np.argsort(-uni_counts, kind="stable")
    unigrams = [
        NgramEntry((int(tok),), int(uni_counts[tok]))
        for tok in uni_order[:top_k]
        if uni_counts[tok] > 0
    ]

    not_doc_start = fire_indices % seq_len != 0
    prev = flat_tokens[fire_indices[not_doc_start] - 1].astype(np.int64)
    cur_bi = flat_tokens[fire_indices[not_doc_start]].astype(np.int64)
    bigrams: list[NgramEntry] = []
    if prev.size:
        keys = prev * vocab + cur_bi
        uniq, counts = np.unique(keys, return_counts=True)
        order = np.argsort(-counts, kind="stable")
        for pos in order[:top_k]:
            if counts[pos] <= 0:
                continue
            key = int(uniq[pos])
            bigrams.append(NgramEntry((key // vocab, key % vocab), int(counts[pos])))
    return ExpertNgrams(bigrams, unigrams)


def _format_lookup(entries: list[NgramEntry], tokenizer: Any) -> str:
    """Format the n-gram list stored as the lookup method's text."""
    parts = [
        f"{''.join(decode_tokens(tokenizer, entry.token_ids))!r} (n={entry.count})"
        for entry in entries
    ]
    if not parts:
        return ""
    return "Activates on ngrams: " + ", ".join(parts)


def _ngram_tasks(
    ctx: PipelineContext, *, order: NgramType, lookup: bool
) -> list[ExplanationTask]:
    """Build one expert task per hookpoint, for single tokens or for token pairs.

    Lookup stores the 100 most frequent items and does not call a model.
    The description methods prompt on the 20 most frequent.
    """
    cfg = ctx.config
    top_k = NGRAM_LOOKUP_K if lookup else NGRAM_AUTOINTERP_SHOWN
    tasks: list[ExplanationTask] = []
    flat_tokens = ctx.corpus.flat_tokens
    is_train = ctx.corpus.is_train
    seq_len = ctx.corpus.seq_len
    for hookpoint in cfg.hookpoints:
        expert_ids = ctx.selection.experts(hookpoint)
        if not expert_ids:
            logger.warning(f"No selected experts for {hookpoint}; skipping")
            continue
        fires = collect_expert_activations(
            cfg.input_dir, hookpoint, expert_ids, expected_tokens=ctx.corpus.n_tokens
        )
        for expert_id in expert_ids:
            indices = fires[expert_id].indices
            train_indices = indices[is_train[indices // seq_len]]
            counted = count_expert_ngrams(
                train_indices, flat_tokens, seq_len=seq_len, top_k=top_k
            )
            primary = counted.bigrams if order == "bigram" else counted.unigrams
            if not primary:
                logger.warning(f"No {order}s for expert {hookpoint}/{expert_id}")
                continue
            shown = primary[:top_k]
            ngram_examples = [
                Example(
                    tokens=[int(t) for t in entry.token_ids],
                    activations=[float(entry.count)],
                    window_id=-1,
                    example_type="positive",
                )
                for entry in shown
            ]
            if lookup:
                tasks.append(
                    ExplanationTask(
                        method=cfg.method,
                        hookpoint=hookpoint,
                        target_kind="expert",
                        target_id=expert_id,
                        messages=[],
                        examples=ngram_examples,
                        text=_format_lookup(shown, ctx.tokenizer),
                        lookup_ngrams=[list(entry.token_ids) for entry in shown],
                        lookup_unigrams=(
                            [
                                list(entry.token_ids)
                                for entry in counted.unigrams[:top_k]
                            ]
                            if order == "bigram"
                            else None
                        ),
                    )
                )
            else:
                tasks.append(
                    ExplanationTask(
                        method=cfg.method,
                        hookpoint=hookpoint,
                        target_kind="expert",
                        target_id=expert_id,
                        messages=build_ngram_messages(shown, tokenizer=ctx.tokenizer),
                        examples=ngram_examples,
                    )
                )
    return tasks


def build_unigram_autointerp_tasks(ctx: PipelineContext) -> list[ExplanationTask]:
    """Build unigram AutoInterp tasks that ask for a description of the top unigrams."""
    return _ngram_tasks(ctx, order="unigram", lookup=False)


def build_bigram_autointerp_tasks(ctx: PipelineContext) -> list[ExplanationTask]:
    """Build bigram AutoInterp tasks that ask for a description of the top bigrams."""
    return _ngram_tasks(ctx, order="bigram", lookup=False)


def build_unigram_lookup_tasks(ctx: PipelineContext) -> list[ExplanationTask]:
    """Build unigram lookup tasks from the 100 most common tokens."""
    return _ngram_tasks(ctx, order="unigram", lookup=True)


def build_bigram_lookup_tasks(ctx: PipelineContext) -> list[ExplanationTask]:
    """Build bigram lookup tasks from the 100 most common token pairs."""
    return _ngram_tasks(ctx, order="bigram", lookup=True)


def build_expert_activation_tasks(ctx: PipelineContext) -> list[ExplanationTask]:
    """Build expert-activation tasks from windows where the expert is routed."""
    cfg = ctx.config
    tasks: list[ExplanationTask] = []
    for hookpoint in cfg.hookpoints:
        expert_ids = ctx.selection.experts(hookpoint)
        if not expert_ids:
            logger.warning(f"No selected experts for {hookpoint}; skipping")
            continue
        fires = collect_expert_activations(
            cfg.input_dir, hookpoint, expert_ids, expected_tokens=ctx.corpus.n_tokens
        )
        for expert_id in expert_ids:
            acts = fires[expert_id]
            sampled = sample_expert_examples(
                ctx, hookpoint, expert_id, acts.indices, acts.values
            )
            if sampled is None:
                continue
            examples = sampled.examples
            positives = split_examples(examples).positives
            messages = build_expert_activation_messages(
                positives, tokenizer=ctx.tokenizer
            )
            tasks.append(
                ExplanationTask(
                    method=cfg.method,
                    hookpoint=hookpoint,
                    target_kind="expert",
                    target_id=expert_id,
                    messages=messages,
                    examples=examples,
                )
            )
    return tasks


def build_esb_tasks(ctx: PipelineContext) -> list[ExplanationTask]:
    """Build Expert Strikes Back tasks from the strongest expert writes."""
    cfg = ctx.config
    tasks: list[ExplanationTask] = []
    for hookpoint in cfg.hookpoints:
        expert_ids = ctx.selection.experts(hookpoint)
        if not expert_ids:
            logger.warning(f"No selected experts for {hookpoint}; skipping")
            continue
        writes = load_expert_writes(
            cfg.input_dir, hookpoint, expected_tokens=ctx.corpus.n_tokens
        )
        for expert_id in expert_ids:
            sampled = sample_esb_examples(ctx, hookpoint, expert_id, writes)
            if sampled is None:
                continue
            examples = sampled.examples
            messages = build_esb_messages(
                expert_id,
                examples,
                tokenizer=ctx.tokenizer,
                ctx_len=cfg.ctx_len,
            )
            tasks.append(
                ExplanationTask(
                    method=cfg.method,
                    hookpoint=hookpoint,
                    target_kind="expert",
                    target_id=expert_id,
                    messages=messages,
                    examples=examples,
                )
            )
    return tasks


#: The method registry: ``ExplainConfig.method`` -> task builder.
EXPLAINERS: dict[str, ExplainerBuilder] = {
    "feature_autointerp": build_feature_autointerp_tasks,
    "routerinterp": build_routerinterp_tasks,
    "unigram_autointerp": build_unigram_autointerp_tasks,
    "bigram_autointerp": build_bigram_autointerp_tasks,
    "unigram_lookup": build_unigram_lookup_tasks,
    "bigram_lookup": build_bigram_lookup_tasks,
    "expert_activation": build_expert_activation_tasks,
    "esb": build_esb_tasks,
}

LOOKUP_METHODS = frozenset({"unigram_lookup", "bigram_lookup"})


def explainer_for_method(method: str) -> ExplainerBuilder:
    """Look up the task builder registered for ``method``."""
    try:
        return EXPLAINERS[method]
    except KeyError:
        msg = f"Unknown explain method {method!r}; available: {sorted(EXPLAINERS)}"
        raise ValueError(msg) from None


def prompt_hash(model: str, messages: list[dict[str, str]]) -> str:
    """Hash one prompt from the model name, the system text, and the other messages.

    A rerun skips a target already stored under this hash.
    """
    system = "\n".join(
        m.get("content", "") for m in messages if m.get("role") == "system"
    )
    prompt = json.dumps(
        [m for m in messages if m.get("role") != "system"],
        sort_keys=True,
        separators=(",", ":"),
    )
    return stable_hash(model + system + prompt)


def example_row(
    window: Example,
    task: ExplanationTask,
    *,
    example_seed: int,
) -> dict[str, Any]:
    """Build one row of an examples parquet."""
    return {
        "method": task.method,
        "hookpoint": task.hookpoint,
        "target_kind": task.target_kind,
        "target_id": task.target_id,
        "example_type": window.example_type,
        "tokens": window.tokens,
        "activations": window.activations,
        "split": window.split,
        "example_seed": example_seed,
        "window_id": window.window_id,
        "group_id": window.group_id,
        "quantile": window.quantile,
    }


def write_example_parquets(
    output_dir: Path,
    tasks: list[ExplanationTask],
    *,
    example_seed: int,
) -> list[str]:
    """Write one parquet of examples per hookpoint, and return where each task's rows sit.

    A task's rows are one contiguous range.
    The reference looks like ``examples/<hookpoint>.parquet#start-end``.
    """
    refs = [""] * len(tasks)
    by_hookpoint: dict[str, list[int]] = {}
    for i, task in enumerate(tasks):
        by_hookpoint.setdefault(task.hookpoint, []).append(i)
    for hookpoint, task_indices in by_hookpoint.items():
        rows: list[dict[str, Any]] = []
        for i in task_indices:
            task = tasks[i]
            start = len(rows)
            rows.extend(
                example_row(w, task, example_seed=example_seed) for w in task.examples
            )
            end = len(rows)
            slug = hookpoint_slug(hookpoint)
            refs[i] = f"{EXAMPLES_DIRNAME}/{slug}.parquet#{start}-{end}"
        if rows:
            path = (
                output_dir / EXAMPLES_DIRNAME / f"{hookpoint_slug(hookpoint)}.parquet"
            )
            write_parquet(path, rows)
            logger.info(f"Wrote {len(rows)} example rows to {path}")
    return refs


def load_explanations(path: Path) -> Explanations:
    """Read ``explanations.jsonl``; latest record wins per target key."""
    if not path.exists():
        return {}
    latest: Explanations = {}
    for record in read_jsonl(path):
        latest[ExplanationKey.from_record(record)] = record
    return latest


def explain_window_ids(explanations_dir: Path) -> ShownWindowIds:
    """Return the windows shown when each explanation was written."""
    examples_dir = explanations_dir / EXAMPLES_DIRNAME
    used: ShownWindowIds = {}
    if not examples_dir.is_dir():
        return used
    for path in sorted(examples_dir.glob("*.parquet")):
        for row in read_parquet(path).to_pylist():
            window_id = int(row["window_id"])
            if window_id < 0:  # n-gram rows carry no window
                continue
            used.setdefault(TargetKey.from_record(row), set()).add(window_id)
    return used


async def run_explain_async(config: ExplainConfig) -> Path:
    """Generate an explanation for every target of ``config.method``.

    The example parquets are written before any model call.
    A target already stored for this same prompt is skipped.
    """
    if not config.hookpoints:
        msg = "ExplainConfig.hookpoints must be non-empty"
        raise ValueError(msg)
    select_manifest = read_manifest(config.selection_dir)
    collect_manifest = read_manifest(config.input_dir)
    corpus = load_corpus(config.input_dir, collect_manifest)
    if corpus.seq_len % config.ctx_len != 0:
        msg = (
            f"seq_len {corpus.seq_len} is not a multiple of ctx_len "
            f"{config.ctx_len}; windows would cross document boundaries"
        )
        raise ValueError(msg)
    selection = Selection.load(config.selection_dir)
    tokenizer = load_tokenizer(collect_manifest)
    ctx = PipelineContext(
        config=config, corpus=corpus, selection=selection, tokenizer=tokenizer
    )

    builder = explainer_for_method(config.method)
    tasks = builder(ctx)
    logger.info(f"Built {len(tasks)} explanation tasks for method={config.method}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    refs = write_example_parquets(
        config.output_dir, tasks, example_seed=config.example_seed
    )

    explanations_path = config.output_dir / EXPLANATIONS_FILENAME
    done = {
        CachedExplanation.from_record(key, record)
        for key, record in load_explanations(explanations_path).items()
    }
    if done:
        logger.info(f"Resuming: {len(done)} explanations already present")

    needs_llm = config.method not in LOOKUP_METHODS
    client = OpenRouterChatClient(config.explainer_model) if needs_llm else None
    sem = asyncio.Semaphore(config.max_concurrent)
    save_lock = asyncio.Lock()  # one writer for the shared explanations JSONL
    stats = {"skipped": 0, "written": 0, "failed": 0}

    async def _explain_one(task: ExplanationTask, example_ref: str) -> None:
        async with sem:
            prompt = task.messages or [
                {"role": "user", "content": task.text or ""},
            ]
            hashed = prompt_hash(config.explainer_model, prompt)
            done_key = CachedExplanation(task.key, hashed)
            if done_key in done:
                stats["skipped"] += 1
                return
            request_ids: list[str] = []
            if task.text is not None:
                text = task.text
            else:
                assert client is not None
                try:
                    result = await client.generate(task.messages)
                    text = result.text.strip()
                    if not text:
                        msg = "Explanation model returned no text"
                        raise ValueError(msg)
                    if result.request_id:
                        request_ids.append(result.request_id)
                except Exception as exc:
                    logger.error(f"Explain failed for {task.key}: {exc!r}")
                    stats["failed"] += 1
                    return
            record = {
                "method": task.method,
                "hookpoint": task.hookpoint,
                "target_kind": task.target_kind,
                "target_id": task.target_id,
                "text": text,
                "n_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
                "explainer_model": ""
                if task.text is not None
                else config.explainer_model,
                "prompt_hash": hashed,
                "response_hash": stable_hash(text),
                "request_ids": request_ids,
                "example_ref": example_ref,
                "example_seed": config.example_seed,
                "lookup_ngrams": task.lookup_ngrams,
                "lookup_unigrams": task.lookup_unigrams,
            }
            async with save_lock:
                await asyncio.to_thread(append_jsonl, explanations_path, [record])
            done.add(done_key)
            stats["written"] += 1

    try:
        from tqdm.asyncio import tqdm as atqdm

        await atqdm.gather(
            *[_explain_one(task, ref) for task, ref in zip(tasks, refs, strict=True)],
            desc=f"Explain {config.method}",
        )
        logger.info(
            f"Explain stage done: written={stats['written']} "
            f"skipped={stats['skipped']} failed={stats['failed']}"
        )

        if stats["failed"]:
            msg = f"Explain stage failed for {stats['failed']} targets"
            raise RuntimeError(msg)

        manifest = build_manifest(
            stage="explain",
            run_name=f"explain-{config.method}",
            config=config,
            parents={
                "collect": collect_manifest.config_hash,
                "select": select_manifest.config_hash,
            },
        )
        write_manifest(config.output_dir, manifest)
        logger.info(f"Wrote explain manifest to {config.output_dir}")
        return config.output_dir
    finally:
        if client is not None:
            await client.aclose()


def run_explain(config: ExplainConfig) -> Path:
    """Run explanation synchronously for the CLI and the experiment jobs."""
    return asyncio.run(run_explain_async(config))
