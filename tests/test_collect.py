"""Train/test membership and sequence batching for the collect stage."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from routerinterp import collect
from routerinterp.collect import iter_sequence_batches
from routerinterp.core import (
    CollectConfig,
    DatasetConfig,
    ModelConfig,
    doc_is_train,
    split_doc_ids,
)


def test_doc_split_is_deterministic_and_exhaustive() -> None:
    kwargs = {"seed": 1, "train_frac": 0.8, "fingerprint": "pile@abc"}
    train, test = split_doc_ids(40, **kwargs)
    again, _ = split_doc_ids(40, **kwargs)
    np.testing.assert_array_equal(train, again)
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(range(40))
    assert list(train) == sorted(train)
    assert train.size and test.size
    for doc_id in train:
        assert doc_is_train(int(doc_id), **kwargs)
    for doc_id in test:
        assert not doc_is_train(int(doc_id), **kwargs)

    # Anchors the hash itself, not only agreement between the two helpers.
    assert doc_is_train(0, seed=0, train_frac=0.5, fingerprint="pile@abc") is False
    assert doc_is_train(1, seed=0, train_frac=0.5, fingerprint="pile@abc") is True
    other, _ = split_doc_ids(40, seed=1, train_frac=0.8, fingerprint="other@rev")
    assert not np.array_equal(train, other)

    empty_train, empty_test = split_doc_ids(0, seed=0, train_frac=0.5, fingerprint="x")
    assert empty_train.size == 0 and empty_test.size == 0
    all_test, no_test = split_doc_ids(6, seed=0, train_frac=0, fingerprint="x")
    no_train, all_train = split_doc_ids(6, seed=0, train_frac=1, fingerprint="x")
    assert all_test.size == 0 and no_test.size == 6
    assert no_train.size == 6 and all_train.size == 0
    with pytest.raises(ValueError, match="train_frac"):
        doc_is_train(0, seed=0, train_frac=1.5, fingerprint="x")
    with pytest.raises(ValueError, match="n_docs"):
        split_doc_ids(-1, seed=0, train_frac=0.5, fingerprint="x")


class _Tokenizer:
    eos_token = "</s>"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert text == self.eos_token
        return [99]

    def __call__(self, text: str, add_special_tokens: bool = False) -> SimpleNamespace:
        return SimpleNamespace(input_ids=[int(piece) for piece in text.split()])


def _config(
    tmp_path: Path,
    *,
    batch_size: int = 2,
    train_frac: float = 0.5,
    seed: int = 7,
) -> CollectConfig:
    return CollectConfig(
        model=ModelConfig(name="toy"),
        dataset=DatasetConfig(
            name="toy-corpus",
            n_tokens=128,
            seq_len=4,
            seed=3,
            text_column="text",
        ),
        output_dir=tmp_path,
        batch_size=batch_size,
        train_frac=train_frac,
        seed=seed,
    )


def _patch_rows(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, object]]) -> dict:
    seen: dict[str, object] = {}

    class _Stream:
        def shuffle(self, seed: int, buffer_size: int) -> _Stream:
            seen["seed"] = seed
            seen["buffer_size"] = buffer_size
            return self

        def __iter__(self):
            return iter(rows)

    monkeypatch.setattr(collect, "load_dataset", lambda *_a, **_k: _Stream())
    return seen


# eos + body. seq_len 4, so doc0 fills a block, doc1+doc2 share the next two,
# and doc3 is a short tail that must not become its own block.
_ROWS = [
    {"text": "1 2 3", "meta": {"pile_set_name": "ArXiv"}},
    {"text": "4", "pile_set_name": "Wiki"},
    {"text": "5 6 7 8 9", "meta": {"pile_set_name": "GitHub"}},
    {"text": "7"},
]
_BLOCKS = [
    np.array([99, 1, 2, 3]),
    np.array([99, 4, 99, 5]),
    np.array([6, 7, 8, 9]),
]


def test_batches_pack_documents_and_label_the_opening_doc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen = _patch_rows(monkeypatch, _ROWS)
    config = _config(tmp_path)
    batches = list(iter_sequence_batches(config, _Tokenizer(), n_sequences=10))

    assert seen["seed"] == config.dataset.seed == 3
    assert len(batches) == 2
    tokens, doc_ids, _split, piles = batches[0]
    assert tokens.shape == (2, 4)
    np.testing.assert_array_equal(tokens, np.stack([_BLOCKS[0], _BLOCKS[1]]))
    np.testing.assert_array_equal(doc_ids, np.array([0, 1]))
    assert piles == ["ArXiv", "Wiki"]
    tokens, doc_ids, _split, piles = batches[1]
    assert tokens.shape == (1, 4)
    np.testing.assert_array_equal(tokens[0], _BLOCKS[2])
    assert int(doc_ids[0]) == 2
    assert piles == ["GitHub"]
    for batch in batches:
        _tokens, batch_docs, batch_split, _piles = batch
        for doc_id, label in zip(batch_docs, batch_split, strict=True):
            assert bool(label) is doc_is_train(
                int(doc_id),
                seed=config.seed,
                train_frac=config.train_frac,
                fingerprint=config.dataset.fingerprint,
            )

    # Resume skips blocks but keeps the original doc ids.
    skipped = list(
        iter_sequence_batches(config, _Tokenizer(), n_sequences=3, skip_sequences=2)
    )
    assert len(skipped) == 1
    assert int(skipped[0][1][0]) == 2
    assert skipped[0][3] == ["GitHub"]

    capped = list(iter_sequence_batches(config, _Tokenizer(), n_sequences=2))
    assert len(capped) == 1
    np.testing.assert_array_equal(capped[0][1], np.array([0, 1]))


def test_batch_split_boundaries_and_missing_pile_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_rows(monkeypatch, _ROWS)
    tokenizer = _Tokenizer()
    for train_frac, expected in ((0.0, False), (1.0, True)):
        config = _config(tmp_path, train_frac=train_frac, batch_size=8)
        batches = list(iter_sequence_batches(config, tokenizer, n_sequences=10))
        assert batches and np.all(batches[0][2] == expected)

    _patch_rows(monkeypatch, [{"text": "1 2 3"}, {"text": "4 5 6"}])
    plain = _config(tmp_path)
    batches = list(iter_sequence_batches(plain, tokenizer, n_sequences=10))
    assert len(batches) == 1
    assert batches[0][3] is None

    _patch_rows(monkeypatch, [])
    assert list(iter_sequence_batches(plain, tokenizer, n_sequences=10)) == []
    _patch_rows(monkeypatch, _ROWS)
    assert (
        list(iter_sequence_batches(plain, tokenizer, n_sequences=3, skip_sequences=9))
        == []
    )

    broken = _config(tmp_path, train_frac=1.5, batch_size=1)
    _patch_rows(monkeypatch, [{"text": "1 2 3"}])
    with pytest.raises(ValueError, match="train_frac"):
        next(iter_sequence_batches(broken, tokenizer, n_sequences=1))


def test_stratified_row_filter_caps_each_source_then_relaxes() -> None:
    sources = ("ArXiv", "Github")
    rows = [
        {"id": "a1", "meta": {"pile_set_name": "ArXiv"}},
        {"id": "g1", "pile_set_name": "Github"},
        {"id": "a2", "meta": {"pile_set_name": "ArXiv"}},
        {"id": "other", "meta": {"pile_set_name": "Pile-CC"}},
        {"id": "a3", "meta": {"pile_set_name": "ArXiv"}},
        {"id": "bare"},
        {"id": "g2", "pile_set_name": "Github"},
        {"id": "a4", "meta": {"pile_set_name": "ArXiv"}},
        {"id": "g3", "pile_set_name": "Github"},
    ]
    taken = list(
        collect._iter_stratified_pile_rows(
            rows,
            num_examples=5,
            sources=sources,
            stale_threshold=2,
        )
    )
    assert [row["id"] for row in taken] == ["a1", "g1", "a2", "g2", "a4"]
