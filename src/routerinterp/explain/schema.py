"""Types shared by sampling, prompts, and the explain/evaluate runners."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from routerinterp.artifacts import Corpus
from routerinterp.core import ExplainConfig
from routerinterp.select import Selection

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

TargetKind = Literal["feature", "expert"]
ExampleType = Literal["positive", "hard_negative", "random_negative"]


@dataclass(frozen=True, slots=True, order=True)
class TargetKey:
    """A feature or expert at one hookpoint."""

    hookpoint: str
    target_kind: str
    target_id: int

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> TargetKey:
        return cls(
            str(record["hookpoint"]),
            str(record["target_kind"]),
            int(record["target_id"]),
        )


@dataclass(frozen=True, slots=True, order=True)
class ExplanationKey:
    """The method that wrote an explanation, and the target it describes."""

    method: str
    hookpoint: str
    target_kind: str
    target_id: int

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> ExplanationKey:
        return cls(
            str(record["method"]),
            str(record["hookpoint"]),
            str(record["target_kind"]),
            int(record["target_id"]),
        )

    @property
    def target(self) -> TargetKey:
        return TargetKey(self.hookpoint, self.target_kind, self.target_id)


@dataclass(frozen=True, slots=True)
class CachedExplanation:
    """An explanation already stored for one prompt."""

    key: ExplanationKey
    prompt_hash: str | None

    @classmethod
    def from_record(
        cls, key: ExplanationKey, record: dict[str, Any]
    ) -> CachedExplanation:
        prompt_hash = record.get("prompt_hash")
        return cls(key, prompt_hash if isinstance(prompt_hash, str) else None)


@dataclass(frozen=True, slots=True)
class NgramEntry:
    """Token ids of one n-gram and how often it occurred."""

    token_ids: tuple[int, ...]
    count: int


type Explanations = dict[ExplanationKey, dict[str, Any]]
type ShownWindowIds = dict[TargetKey, set[int]]


@dataclass
class Example:
    """One example for a target (Parquet row + prompt input)."""

    tokens: list[int]
    activations: list[float]
    window_id: int  # corpus grid row; -1 for n-gram rows
    example_type: ExampleType
    split: str = "train"
    group_id: int | None = None  # feature id when examples are grouped by feature
    quantile: int | None = None
    # In-memory only (never written to Parquet): per-position promoted token
    # ids [ctx_len, P] for esb prompts.
    promoted_token_ids: list[list[int]] | None = None


@dataclass
class ExplanationTask:
    """One target to explain: examples + assembled LLM messages."""

    method: str
    hookpoint: str
    target_kind: TargetKind
    target_id: int
    messages: list[dict[str, str]]
    examples: list[Example]
    # Set for lookup methods: the description is the n-gram table, not an LLM call.
    text: str | None = None
    lookup_ngrams: list[list[int]] | None = None
    lookup_unigrams: list[list[int]] | None = None

    @property
    def key(self) -> ExplanationKey:
        return ExplanationKey(
            self.method, self.hookpoint, self.target_kind, self.target_id
        )


@dataclass
class PipelineContext:
    """Everything method builders need: config + loaded upstream artifacts."""

    config: ExplainConfig
    corpus: Corpus
    selection: Selection
    tokenizer: PreTrainedTokenizerBase


ExplainerBuilder = Callable[[PipelineContext], list[ExplanationTask]]


@dataclass
class EvalWindow:
    """One held-out scoring window."""

    window_id: int
    tokens: list[int]
    str_tokens: list[str]
    activations: list[float]
    ground_truth: bool
    highlight: set[int]
    quantile: int | None = None


@dataclass(frozen=True)
class PRF1:
    """Binary classification metrics with confusion counts."""

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int


@dataclass(frozen=True)
class ChatResponse:
    """One chat completion.
    ``request_id`` is the OpenRouter generation id.
    """

    text: str
    request_id: str | None = None
