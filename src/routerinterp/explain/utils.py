"""Token highlighting, ordinal score parsing, and precision/recall."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from routerinterp.explain.schema import PRF1, Example

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

HIGHLIGHT_THRESHOLD = 0.6  # fraction of the window max


def decode_tokens(
    tokenizer: PreTrainedTokenizerBase, token_ids: Sequence[int] | np.ndarray
) -> list[str]:
    """Decode one string per token id."""
    return [
        str(t)
        for t in tokenizer.batch_decode(list(token_ids), skip_special_tokens=True)
    ]


def highlight(
    str_tokens: Sequence[str],
    activations: Sequence[float],
    *,
    threshold_ratio: float = HIGHLIGHT_THRESHOLD,
    markers: tuple[str, str] = ("<<", ">>"),
) -> str:
    """Wrap tokens whose activation is above ``max * threshold_ratio`` in markers.

    Adjacent marked tokens share one pair of markers.
    A window of all zeros is left unmarked.
    """
    if not str_tokens:
        return ""
    threshold = max(activations) * threshold_ratio
    parts: list[str] = []
    i = 0
    n = len(str_tokens)
    while i < n:
        if activations[i] > threshold:
            parts.append(markers[0])
            while i < n and activations[i] > threshold:
                parts.append(str_tokens[i])
                i += 1
            parts.append(markers[1])
        else:
            parts.append(str_tokens[i])
            i += 1
    return "".join(parts)


def highlight_spans(
    str_tokens: Sequence[str],
    indices: set[int],
    *,
    markers: tuple[str, str] = ("<<", ">>"),
) -> str:
    """Highlight an explicit set of token indices."""
    return highlight(
        str_tokens,
        [1.0 if i in indices else 0.0 for i in range(len(str_tokens))],
        threshold_ratio=0.5,
        markers=markers,
    )


def join_activations(
    str_tokens: Sequence[str],
    activations: Sequence[float],
    normalized_activations: Sequence[float],
    *,
    threshold_ratio: float = HIGHLIGHT_THRESHOLD,
    label: str = "Activations",
) -> str:
    """Format a line of the tokens above the relative threshold, with their 0-10 display
    values.

    Only the first 11 such tokens are included.
    """
    acts = ""
    activation_count = 0
    max_act = max(activations) if len(activations) > 0 else 0.0
    for str_tok, token_activation, normalized_activation in zip(
        str_tokens, activations, normalized_activations, strict=False
    ):
        if token_activation > max_act * threshold_ratio:
            if activation_count > 10:
                break
            acts += f'("{str_tok}" : {int(normalized_activation)}), '
            activation_count += 1
    return f"{label}: " + acts


def normalize_activations(
    activations: Sequence[float], max_activation: float, eps: float = 1e-6
) -> list[int]:
    """Scale activations onto the 0-10 display range with ``ceil(act * 10 / max)``."""
    max_activation = max(max_activation, eps)
    return [
        int(min(10, max(0, int(np.ceil(a * 10 / max_activation))))) for a in activations
    ]


# ---------------------------------------------------------------------------
# Ordinal response parsing
# ---------------------------------------------------------------------------

_OBJECT_BLOCK_RE = re.compile(r"\{[\s\S]*\}")
_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+\-]+)?\s*([\s\S]*?)```", re.IGNORECASE)


def _extract_object_block(text: str) -> str:
    """Strip markdown fences/preamble; return first ``{...}`` block if present."""
    stripped = text.strip()
    fence = _FENCE_RE.search(stripped)
    if fence:
        stripped = fence.group(1).strip()
    match = _OBJECT_BLOCK_RE.search(stripped)
    if match:
        return match.group(0)
    return stripped


def _strip_json_fence(text: str) -> str:
    """Strip a markdown JSON fence from the start and end of the text."""
    stripped = text.strip()
    return stripped.replace("```json", "").replace("```", "").strip()


def _coerce_ordinal_dict(parsed: object, batch_size: int) -> list[int] | None:
    """Read ordered scores from 0 to 4 out of an example-id mapping."""
    if not isinstance(parsed, dict):
        return None
    out: list[int] = []
    for i in range(batch_size):
        found = False
        value: object = None
        for key in (i, str(i), f"example_{i}"):
            if key in parsed:
                value = parsed[key]
                found = True
                break
        if (
            not found
            or isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 4
        ):
            return None
        out.append(value)
    return out


def parse_ordinal_scores(text: str, batch_size: int) -> list[int]:
    """Read a mapping from example id to an integer score from 0 to 4.

    JSON in a markdown fence is accepted, and a key may be ``0``, ``"0"``, or
    ``"example_0"``.
    Raises unless the text yields exactly ``batch_size`` scores.
    """
    stripped = _strip_json_fence(text)
    candidates = [stripped, _extract_object_block(text)]
    seen: set[str] = set()
    for block in candidates:
        if not block or block in seen:
            continue
        seen.add(block)
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(block)
            except (json.JSONDecodeError, SyntaxError, ValueError, TypeError):
                continue
            scores = _coerce_ordinal_dict(parsed, batch_size)
            if scores is not None:
                return scores
    msg = f"Could not parse ordinal outputs of length {batch_size}"
    raise ValueError(msg)


def binarise_scores(scores: list[int], threshold: int) -> list[bool]:
    """Turn grades from 0 to 4 into yes/no predictions at ``threshold``."""
    return [score >= threshold for score in scores]


# ---------------------------------------------------------------------------
# Precision / recall / F1 (pure; sklearn binary zero_division=0 semantics)
# ---------------------------------------------------------------------------


def confusion_counts(
    ground_truths: list[bool], predictions: list[bool]
) -> tuple[int, int, int, int]:
    """Return ``(tp, fp, tn, fn)``."""
    if len(ground_truths) != len(predictions):
        msg = (
            f"ground_truths ({len(ground_truths)}) and predictions "
            f"({len(predictions)}) must have equal length"
        )
        raise ValueError(msg)
    tp = fp = tn = fn = 0
    for truth, pred in zip(ground_truths, predictions, strict=True):
        if truth and pred:
            tp += 1
        elif not truth and pred:
            fp += 1
        elif not truth and not pred:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def prf1_from_counts(tp: int, fp: int, tn: int, fn: int) -> PRF1:
    """Compute precision, recall, and F1 from confusion counts.
    Each is 0 when its denominator is 0.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom = 2 * tp + fp + fn
    f1 = 2 * tp / denom if denom > 0 else 0.0
    return PRF1(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, tn=tn, fn=fn)


def compute_prf1(ground_truths: list[bool], predictions: list[bool]) -> PRF1:
    """Compute confusion counts, precision, recall, and F1 from two boolean lists."""
    tp, fp, tn, fn = confusion_counts(ground_truths, predictions)
    return prf1_from_counts(tp, fp, tn, fn)


@dataclass(frozen=True, slots=True)
class ExampleSplit:
    """Positives and the other examples that share a group."""

    positives: list[Example]
    negatives: list[Example]


def split_examples(examples: Sequence[Example]) -> ExampleSplit:
    """Split examples into positives and negatives by ``example_type``."""
    positives = [w for w in examples if w.example_type == "positive"]
    negatives = [w for w in examples if w.example_type != "positive"]
    return ExampleSplit(positives, negatives)


def group_examples(examples: Sequence[Example]) -> list[ExampleSplit]:
    """Group examples by ``group_id``, positives and negatives, in the order groups first
    appear.
    """
    order: list[int] = []
    buckets: dict[int, list[Example]] = {}
    for window in examples:
        if window.group_id is None:
            msg = "Example is missing group_id"
            raise ValueError(msg)
        if window.group_id not in buckets:
            order.append(window.group_id)
            buckets[window.group_id] = []
        buckets[window.group_id].append(window)
    return [split_examples(buckets[group_id]) for group_id in order]
