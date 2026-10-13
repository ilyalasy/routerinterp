"""Assemble prompts for explainer and scorer LLMs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from routerinterp.explain.schema import EvalWindow, Example, NgramEntry
from routerinterp.explain.utils import (
    decode_tokens,
    group_examples,
    highlight,
    highlight_spans,
    join_activations,
    normalize_activations,
)
from routerinterp.prompts import render_messages

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# AutoInterp shows this many n-grams of one order (unigram or bigram).
NGRAM_AUTOINTERP_SHOWN = 20
# Lookup baselines keep the top-100 co-occurring n-grams (paper Table 1).
NGRAM_LOOKUP_K = 100
# Expert Impact: logit-lens tokens and top activations listed per snippet.
ESB_NUM_LOGITS = 3
ESB_TOP_ACTS = 5

ROUTED_NEG_HEADER = "Hard negatives (feature fires, expert not selected):"


def moe_neg_header(*, routed: bool) -> str:
    """Return the negative-example label for ``prompts/moe_expert.jinja``."""
    return ROUTED_NEG_HEADER if routed else ""


def build_moe_explainer_prompt(
    examples: str,
    *,
    grouped: bool = False,
    contrastive: bool = False,
    routed: bool = False,
) -> list[dict[str, str]]:
    """Build a mixture-of-experts prompt with one few-shot example and highlighted tokens."""
    return render_messages(
        "moe_expert",
        examples=examples,
        grouped=grouped,
        contrastive=contrastive,
        routed=routed,
        neg_header=moe_neg_header(routed=routed),
    )


def build_feature_messages(
    positives: list[Example],
    negatives: list[Example],
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_activation: float,
) -> list[dict[str, str]]:
    """Build a feature prompt from activating examples, and from non-activating examples
    when any are present.
    """
    pos_lines: list[str] = []
    for i, window in enumerate(positives, start=1):
        str_tokens = decode_tokens(tokenizer, window.tokens)
        normalized = normalize_activations(window.activations, max_activation)
        if negatives:
            highlighted = highlight(str_tokens, window.activations)
            pos_lines.append(
                f"Example {i}:  {highlighted.strip().replace(chr(10), '')}"
            )
        else:
            highlighted = highlight(str_tokens, window.activations)
            pos_lines.append(
                f"Example {i}:  {highlighted.strip().replace(chr(10), ' ')}"
            )
        pos_lines.append(join_activations(str_tokens, window.activations, normalized))

    if not negatives:
        user = f"\n{'\n'.join(pos_lines)}\n"
        contrastive = False
    else:
        lines: list[str] = ["ACTIVATING EXAMPLES:", *pos_lines]
        lines.append("\nNON-ACTIVATING EXAMPLES:")
        for i, window in enumerate(negatives, start=1):
            str_tokens = decode_tokens(tokenizer, window.tokens)
            highlighted = highlight(str_tokens, window.activations)
            lines.append(f"Example {i}:  {highlighted.strip().replace(chr(10), '')}")
        user = "\n".join(lines)
        contrastive = True
    return render_messages("feature_autointerp", contrastive=contrastive, user=user)


def build_expert_activation_messages(
    positives: list[Example],
    *,
    tokenizer: PreTrainedTokenizerBase,
) -> list[dict[str, str]]:
    """Build an expert-activation prompt from highlighted windows where the expert is
    routed.
    """
    lines: list[str] = []
    for i, window in enumerate(positives, start=1):
        str_tokens = decode_tokens(tokenizer, window.tokens)
        lines.append(f"Example {i}:  {highlight(str_tokens, window.activations)}")
    return build_moe_explainer_prompt("\n".join(lines))


def build_routerinterp_messages(
    examples: list[Example],
    *,
    tokenizer: PreTrainedTokenizerBase,
) -> list[dict[str, str]]:
    """Build one prompt block per feature, positives then negatives.

    Uses the contrastive few-shot when any feature has hard negatives.
    """
    grouped = group_examples(examples)
    contrastive = any(group.negatives for group in grouped)
    neg_header = moe_neg_header(routed=contrastive)
    group_blocks: list[str] = []
    for g, group in enumerate(grouped, start=1):
        positives = group.positives
        negatives = group.negatives
        lines = [f"Group {g}:"]
        for i, window in enumerate(positives, start=1):
            str_tokens = decode_tokens(tokenizer, window.tokens)
            lines.append(f"Example {i}: {highlight(str_tokens, window.activations)}")
        if negatives:
            lines.append(neg_header)
            for i, window in enumerate(negatives, start=1):
                str_tokens = decode_tokens(tokenizer, window.tokens)
                lines.append(
                    f"Example {i}: {highlight(str_tokens, window.activations)}"
                )
        group_blocks.append("\n".join(lines))
    return build_moe_explainer_prompt(
        "\n\n".join(group_blocks),
        grouped=True,
        contrastive=contrastive,
        routed=contrastive,
    )


def _decode_promoted_token(tokenizer: PreTrainedTokenizerBase, token_id: int) -> str:
    if token_id < 0:
        return ""
    text = tokenizer.decode([token_id], skip_special_tokens=True)
    return text.replace("Ġ", "").replace("▁", "").strip()


def build_esb_messages(
    expert_id: int,
    positives: list[Example],
    *,
    tokenizer: PreTrainedTokenizerBase,
    ctx_len: int,
    num_logits: int = ESB_NUM_LOGITS,
    max_top_acts: int = ESB_TOP_ACTS,
) -> list[dict[str, str]]:
    """Build the Expert Strikes Back prompt from Herbst et al.
    2026.
    """
    examples: list[dict[str, Any]] = []
    for example_id, window in enumerate(positives, start=1):
        str_tokens = decode_tokens(tokenizer, window.tokens)
        scores = window.activations
        context = highlight(
            str_tokens, scores, threshold_ratio=0.0, markers=("**", "**")
        )
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = [i for i in ranked[:max_top_acts] if scores[i] > 0.0]
        top: list[dict[str, str]] = []
        for index in top_indices:
            promoted = ""
            if window.promoted_token_ids is not None:
                ids = window.promoted_token_ids[index][:num_logits]
                promoted = ", ".join(
                    decoded
                    for token_id in ids
                    if (decoded := _decode_promoted_token(tokenizer, int(token_id)))
                )
            top.append(
                {
                    "token": str_tokens[index].replace("Ġ", "").replace("▁", ""),
                    "score": f"{scores[index]:.2f}",
                    "promoted": promoted,
                }
            )
        examples.append({"id": example_id, "context": context, "top": top})
    return render_messages(
        "esb",
        expert_id=expert_id,
        seq_len=ctx_len,
        num_logits=num_logits,
        num_show=max_top_acts,
        examples=examples,
    )


def build_ngram_messages(
    entries: list[NgramEntry],
    *,
    tokenizer: PreTrainedTokenizerBase,
) -> list[dict[str, str]]:
    """Build a prompt that asks for a description of the listed unigrams or bigrams."""
    words = ["".join(decode_tokens(tokenizer, entry.token_ids)) for entry in entries]
    return render_messages("ngram", words=words)


def ordinal_parse_retry_message(batch_size: int) -> dict[str, str]:
    """Build the follow-up message sent when ordinal JSON parsing fails."""
    last_key = batch_size - 1
    return {
        "role": "user",
        "content": (
            "\n\nYour previous response could not be parsed. "
            "Return ONLY one valid JSON object mapping example id to score, "
            f'with string keys "0" through "{last_key}" and integer values '
            "0 through 4 (no null, strings, or markdown). "
            'Example: {"0": 4, "1": 0}. Do not include any other text.'
        ),
    }


def build_scoring_messages(
    explanation: str, batch: list[EvalWindow]
) -> list[dict[str, str]]:
    """Build one scoring prompt that grades a batch of windows."""
    examples = "\n\n".join(
        f"<example_{i}>\n{highlight_spans(w.str_tokens, w.highlight)}\n</example_{i}>"
        for i, w in enumerate(batch)
    )
    return render_messages("graded_scoring", explanation=explanation, examples=examples)
