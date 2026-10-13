"""Train Top-K SAEs with eai-sparsify and export one SafeTensors directory per hookpoint."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from datasets import load_dataset
from loguru import logger

from routerinterp.backends import get_model_backend
from routerinterp.core import (
    DatasetConfig,
    ModelConfig,
    _Config,
    build_manifest,
    set_global_seed,
    write_json,
    write_manifest,
    write_safetensors,
)

if TYPE_CHECKING:
    # sparsify is the `sae` group, imported in train_sae().
    from datasets import Dataset
    from sparsify import Trainer
    from transformers import PreTrainedTokenizerBase


class SAETrainConfig(_Config):
    """Top-K SAE training run; all hyperparameters live here, none in code."""

    model: ModelConfig
    dataset: DatasetConfig
    hookpoints: list[str]
    expansion_factor: int = 32
    k: int = 32
    lr: float | None = None  # None -> sparsify's per-optimizer auto LR
    batch_size: int = 32
    grad_acc_steps: int = 1
    optimizer: Literal["adam", "muon", "signum"] = "signum"
    auxk_alpha: float = 0.0
    checkpoint_dir: Path = Path("./sae_checkpoints")
    save_every: int = 1000
    run_name: str | None = None  # default: "<hookpoints>-<k>-<expansion_factor>"
    output_dir: Path = Path("./sae_export")
    wandb_project: str | None = None  # W&B logging disabled when unset
    wandb_entity: str | None = None
    seed: int = 0


def train_sae(config: SAETrainConfig) -> Path:
    """Train Top-K SAEs on ``config.hookpoints`` and export them, resuming the latest
    checkpoint.
    """
    from sparsify import SparseCoderConfig, TrainConfig, Trainer  # sae group

    set_global_seed(config.seed)
    run_name = config.run_name or "-".join(
        [*config.hookpoints, str(config.k), str(config.expansion_factor)]
    )

    backend = get_model_backend(config.model)
    model = backend.load_model()
    tokenizer = backend.load_tokenizer()
    backend.validate(model, config.hookpoints)

    tokenized = _load_tokenized_dataset(config, tokenizer)
    log_to_wandb = _setup_wandb(config)

    train_config = TrainConfig(
        sae=SparseCoderConfig(k=config.k, expansion_factor=config.expansion_factor),
        hookpoints=list(config.hookpoints),
        batch_size=config.batch_size,
        lr=config.lr,
        grad_acc_steps=config.grad_acc_steps,
        save_every=config.save_every,
        save_dir=str(config.checkpoint_dir),
        run_name=run_name,
        optimizer=config.optimizer,
        auxk_alpha=config.auxk_alpha,
        log_to_wandb=log_to_wandb,
        init_seeds=[config.seed],
    )
    trainer = Trainer(train_config, tokenized, model)

    checkpoint = _latest_checkpoint(config.checkpoint_dir, run_name)
    if checkpoint is not None:
        logger.info(f"Resuming SAE training from {checkpoint}")
        trainer.load_state(str(checkpoint))

    logger.info(
        f"Training SAEs on {config.hookpoints} (k={config.k}, "
        f"expansion_factor={config.expansion_factor}, "
        f"optimizer={config.optimizer}, lr={config.lr or 'auto'})"
    )
    trainer.fit()

    exported = _export_saes(trainer, config)
    write_manifest(
        config.output_dir,
        build_manifest(
            "train_sae",
            run_name,
            config,
            dataset_revision=config.dataset.revision,
            model_revision=getattr(model.config, "_commit_hash", None),
        ),
    )
    logger.info(f"Exported {len(exported)} SAE(s) to {config.output_dir}")
    return config.output_dir


def _load_tokenized_dataset(
    config: SAETrainConfig, tokenizer: PreTrainedTokenizerBase
) -> Dataset:
    """Load a chunked token stream, capped at ``dataset.n_tokens``."""
    from sparsify.data import chunk_and_tokenize  # sae group

    ds_cfg = config.dataset
    logger.info(f"Loading dataset {ds_cfg.name} (split={ds_cfg.split})")
    dataset = load_dataset(ds_cfg.name, revision=ds_cfg.revision, split=ds_cfg.split)
    dataset = dataset.shuffle(seed=ds_cfg.seed)
    tokenized = chunk_and_tokenize(
        dataset,
        tokenizer,
        text_key=ds_cfg.text_column,
        max_seq_len=ds_cfg.seq_len,
        num_proc=ds_cfg.num_proc,
    )
    n_seq = ds_cfg.n_tokens // ds_cfg.seq_len
    if len(tokenized) > n_seq:
        tokenized = tokenized.select(range(n_seq))
    logger.info(f"Training on {len(tokenized)} sequences of {ds_cfg.seq_len} tokens")
    return tokenized


def _setup_wandb(config: SAETrainConfig) -> bool:
    """Enable sparsify's built-in W&B logging when a project is configured."""
    project = config.wandb_project or os.environ.get("WANDB_PROJECT")
    if not project:
        return False
    os.environ.setdefault("WANDB_PROJECT", project)
    entity = config.wandb_entity or os.environ.get("WANDB_ENTITY")
    if entity:
        os.environ.setdefault("WANDB_ENTITY", entity)
    return True


def _latest_checkpoint(checkpoint_dir: Path, run_name: str) -> Path | None:
    """Find the latest trainer checkpoint directory, preferring ``run_name``."""
    preferred = checkpoint_dir / run_name / "state.pt"
    if preferred.is_file():
        return preferred.parent
    candidates = sorted(
        checkpoint_dir.glob("*/state.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1].parent if candidates else None


def _export_saes(trainer: Trainer, config: SAETrainConfig) -> list[Path]:
    """Write each SAE as ``sae.safetensors`` and ``cfg.json`` in the sparsify on-disk
    layout.
    """
    exported: list[Path] = []
    for name, sae in trainer.saes.items():
        out_dir = config.output_dir / name
        tensors = {
            key: value.detach().cpu().float().numpy()
            for key, value in sae.state_dict().items()
        }
        write_safetensors(out_dir / "sae.safetensors", tensors)
        write_json(
            out_dir / "cfg.json",
            {
                **sae.cfg.to_dict(),
                "d_in": sae.d_in,
                "hookpoint": name,
                "model_name": config.model.name,
                "model_revision": config.model.revision,
            },
        )
        exported.append(out_dir)
        logger.info(f"Exported SAE {name!r} -> {out_dir}")
    return exported
