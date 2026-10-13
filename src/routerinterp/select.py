"""Score how strongly each SAE feature predicts each expert, on train tokens only, and keep
the top ``n`` features per expert.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pyarrow as pa
import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor
from tqdm import tqdm

from routerinterp import core
from routerinterp.backends import (
    get_model_backend,
    get_sae_backend,
    install_residual_hooks,
    install_residual_replacement,
)
from routerinterp.collect import TOKENS_SPLIT, load_collect_config, load_collect_info
from routerinterp.core import (
    SelectConfig,
    residual_is_input,
    router_layer_for_hookpoint,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from transformers import PreTrainedModel

    from routerinterp.backends import ModelBackend, SAEBackend
    from routerinterp.core import CollectConfig, SAEConfig

__all__ = [
    "MarginAggState",
    "MarginAttribution",
    "accumulate_attribution",
    "accumulate_contributions",
    "attribute_batch_margins",
    "compute_rho_useful_scores",
    "compute_routing_margin_scores",
    "cosine_scores",
    "load_token_split",
    "margin_diagonal",
    "membership_margins",
    "FeatureScore",
    "Selection",
    "run_select",
    "select_top_features",
]

DEFAULT_EXPERT_CHUNK = 8  # experts per vmap-ed Jacobian chunk


@dataclass(frozen=True, slots=True)
class FeatureScore:
    """One selected feature and the score that ranked it."""

    feature_id: int
    score: float


SELECTION_SCHEMA = pa.schema(
    [
        ("method", pa.string()),
        ("hookpoint", pa.string()),
        ("expert_id", pa.int32()),
        ("n", pa.int32()),
        ("rank", pa.int32()),
        ("feature_id", pa.int32()),
        ("score", pa.float32()),
    ]
)


@dataclass
class Selection:
    """One ``selections/<method>/n<n>.parquet`` table written by ``run_select``."""

    method: str
    n: int
    path: Path
    records: list[dict[str, Any]]

    @classmethod
    def load(cls, selection_dir: Path) -> Selection:
        """Load the feature selection table named in the run manifest.

        If the manifest does not record the method and ``n``, there must be exactly one
        parquet under ``selections/``.
        """
        manifest = core.read_manifest(selection_dir)
        cfg = manifest.resolved_config
        method = cfg.get("method")
        n = cfg.get("n")
        path: Path | None = None
        if method is not None and n is not None:
            candidate = selection_dir / "selections" / str(method) / f"n{n}.parquet"
            if candidate.exists():
                path = candidate
        if path is None:
            found = sorted(selection_dir.glob("selections/*/n*.parquet"))
            if len(found) == 1:
                path = found[0]
            else:
                msg = (
                    f"Cannot resolve selection parquet under {selection_dir} "
                    f"(manifest method={method!r} n={n!r}; glob found "
                    f"{[str(p) for p in found]})"
                )
                raise FileNotFoundError(msg)
        table = core.read_parquet(path)
        missing = set(SELECTION_SCHEMA.names) - set(table.column_names)
        if missing:
            msg = f"Selection parquet {path} is missing columns {sorted(missing)}"
            raise ValueError(msg)
        if method is None:
            method = path.parent.name
        if n is None:
            n = int(path.stem.removeprefix("n"))
        records = table.to_pylist()
        logger.info(
            f"Loaded selection {path} ({len(records)} rows, method={method} n={n})"
        )
        return cls(method=str(method), n=int(n), path=path, records=records)

    def experts(self, hookpoint: str) -> list[int]:
        """Return the sorted expert ids selected at ``hookpoint``."""
        hp = core.normalize_hookpoint(hookpoint)
        return sorted(
            {
                int(r["expert_id"])
                for r in self.records
                if core.normalize_hookpoint(str(r["hookpoint"])) == hp
            }
        )

    def features_for_expert(self, hookpoint: str, expert_id: int) -> list[FeatureScore]:
        """Return the features selected for one expert, in rank order."""
        hp = core.normalize_hookpoint(hookpoint)
        rows = [
            r
            for r in self.records
            if core.normalize_hookpoint(str(r["hookpoint"])) == hp
            and int(r["expert_id"]) == expert_id
        ]
        rows.sort(key=lambda r: int(r["rank"]))
        return [FeatureScore(int(r["feature_id"]), float(r["score"])) for r in rows]

    def feature_union(self, hookpoint: str) -> list[int]:
        """Return the sorted feature ids selected at ``hookpoint``, across experts."""
        hp = core.normalize_hookpoint(hookpoint)
        return sorted(
            {
                int(r["feature_id"])
                for r in self.records
                if core.normalize_hookpoint(str(r["hookpoint"])) == hp
            }
        )


# ---------------------------------------------------------------------------
# Membership margins (Eq. 2)
# ---------------------------------------------------------------------------


def membership_margins(logits_BSE: Tensor, k: int) -> Tensor:
    """Compare each expert's logit with the logit just across the top-k cutoff.

    For an expert already in the top-k, that boundary is the next logit down.
    For an expert outside it, the boundary is the lowest logit still inside.
    """
    n_experts = logits_BSE.shape[-1]
    if not 1 <= k < n_experts:
        msg = f"k must be in [1, E), got k={k}, E={n_experts}"
        raise ValueError(msg)
    top_vals_BSK1 = logits_BSE.topk(k + 1, dim=-1).values
    s_k = top_vals_BSK1[..., k - 1 : k]  # k-th largest
    s_k1 = top_vals_BSK1[..., k : k + 1]  # (k+1)-th largest
    tau_BSE = torch.where(logits_BSE >= s_k, s_k1, s_k)
    return logits_BSE - tau_BSE


# ---------------------------------------------------------------------------
# Aggregation state: S = A - lambda * L with coverage/effect diagnostics
# ---------------------------------------------------------------------------


@dataclass
class MarginAggState:
    """Running sums for causal-support aggregation ``S = A - lambda * L``."""

    sum_pos_EF: Tensor
    sum_neg_EF: Tensor
    count_pos_E: Tensor
    count_neg_E: Tensor
    count_fire_pos_EF: Tensor
    sum_c_fire_pos_EF: Tensor
    lambda_: float = 1.0

    @classmethod
    def create(
        cls,
        n_experts: int,
        n_features: int,
        *,
        lambda_: float = 1.0,
    ) -> MarginAggState:
        return cls(
            sum_pos_EF=torch.zeros(n_experts, n_features, dtype=torch.float32),
            sum_neg_EF=torch.zeros(n_experts, n_features, dtype=torch.float32),
            count_pos_E=torch.zeros(n_experts, dtype=torch.float32),
            count_neg_E=torch.zeros(n_experts, dtype=torch.float32),
            count_fire_pos_EF=torch.zeros(n_experts, n_features, dtype=torch.float32),
            sum_c_fire_pos_EF=torch.zeros(n_experts, n_features, dtype=torch.float32),
            lambda_=lambda_,
        )

    def finalize(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(scores_EF, coverage_EF, effect_EF)``."""
        a_EF = self.sum_pos_EF / self.count_pos_E.unsqueeze(1).clamp(min=1)
        l_EF = self.sum_neg_EF / self.count_neg_E.unsqueeze(1).clamp(min=1)
        scores_EF = a_EF - self.lambda_ * l_EF
        coverage_EF = self.count_fire_pos_EF / self.count_pos_E.unsqueeze(1).clamp(
            min=1
        )
        effect_EF = self.sum_c_fire_pos_EF / self.count_fire_pos_EF.clamp(min=1)
        return scores_EF, coverage_EF, effect_EF


def accumulate_contributions(
    state: MarginAggState,
    *,
    expert_id: int,
    y_active_T: Tensor,
    feature_ids_list: list[Tensor],
    scores_list: list[Tensor],
) -> None:
    """Add one expert's per-token feature contributions into ``state``.

    The score uses only the positive part of each contribution.
    The effect diagnostic keeps the signed value.
    """
    t_count = y_active_T.numel()
    n_pos = int(y_active_T.sum().item())
    state.count_pos_E[expert_id] += n_pos
    state.count_neg_E[expert_id] += t_count - n_pos

    for t in range(t_count):
        ids_A = feature_ids_list[t]
        scores_A = scores_list[t]
        if ids_A.numel() == 0:
            continue
        ids_A = ids_A.long().cpu()
        scores_A = scores_A.float().cpu()
        pos_support_A = scores_A.clamp(min=0)
        if y_active_T[t]:
            state.sum_pos_EF[expert_id].index_add_(0, ids_A, pos_support_A)
            state.count_fire_pos_EF[expert_id].index_add_(
                0, ids_A, torch.ones_like(scores_A)
            )
            state.sum_c_fire_pos_EF[expert_id].index_add_(0, ids_A, scores_A)
        else:
            state.sum_neg_EF[expert_id].index_add_(0, ids_A, pos_support_A)


# ---------------------------------------------------------------------------
# Chunked same-token Jacobian diagonals (Eq. 3 attribution engine)
# ---------------------------------------------------------------------------


class _StopForward(Exception):
    """Raised from a hook to abort the rest of the forward."""


def _store_and_stop(
    _hookpoint: str, hidden: Tensor, *, store: dict[str, Tensor]
) -> None:
    """Stash the residual and abort the rest of the forward."""
    store["h_BSH"] = hidden
    raise _StopForward


def _store_router_logits(logits_TE: Tensor, *, store: dict[str, Tensor]) -> None:
    """Stash in-graph router logits and abort the rest of the forward."""
    store["logits_TE"] = logits_TE
    raise _StopForward


def _capture_residual(
    model: PreTrainedModel,
    *,
    hookpoint: str,
    input_ids_BS: Tensor,
) -> Tensor:
    """Capture the detached residual at ``hookpoint``, without gradients, and stop the
    forward pass.
    """
    store: dict[str, Tensor] = {}
    handles = install_residual_hooks(
        model, [hookpoint], partial(_store_and_stop, store=store)
    )
    try:
        with torch.no_grad():
            try:
                model(input_ids_BS)
            except _StopForward:
                pass
    finally:
        for handle in handles:
            handle.remove()
    h_BSH = store.get("h_BSH")
    if h_BSH is None:
        msg = f"Failed to capture residual at hookpoint {hookpoint!r}"
        raise RuntimeError(msg)
    return h_BSH


def _make_margins_fn(
    model: PreTrainedModel,
    backend: ModelBackend,
    *,
    hookpoint: str,
    router_layer_idx: int,
    input_ids_BS: Tensor,
    top_k: int,
    position: int,
    aux: dict[str, Tensor],
) -> Callable[[Tensor], Tensor]:
    """Build the function whose derivative is the margin.

    Calling it replaces the residual at the hookpoint with its argument, reads the
    router logits, and then stops the forward pass.
    """
    batch_size, seq_len = int(input_ids_BS.shape[0]), int(input_ids_BS.shape[1])
    pos = position % seq_len

    def f(h_BSH: Tensor) -> Tensor:
        handles = [install_residual_replacement(model, hookpoint, h_BSH)]
        store: dict[str, Tensor] = {}
        try:
            handles.append(
                backend.install_router_logit_hook(
                    model,
                    router_layer_idx,
                    partial(_store_router_logits, store=store),
                )
            )
            try:
                model(input_ids_BS)
            except _StopForward:
                pass
        finally:
            for handle in handles:
                handle.remove()
        logits_TE = store.get("logits_TE")
        if logits_TE is None:
            msg = f"Failed to capture router logits at layer {router_layer_idx}"
            raise RuntimeError(msg)
        if logits_TE.shape[0] != batch_size * seq_len:
            msg = (
                f"Router logits {tuple(logits_TE.shape)} do not match "
                f"B*S = {batch_size}*{seq_len}"
            )
            raise RuntimeError(msg)
        logits_BSE = logits_TE.view(batch_size, seq_len, -1)
        aux["logits_BE"] = logits_BSE[:, pos, :].detach()
        return membership_margins(logits_BSE, top_k)[:, pos, :]

    return f


def margin_diagonal(
    model: PreTrainedModel,
    backend: ModelBackend,
    *,
    hookpoint: str,
    router_layer_idx: int,
    input_ids_BS: Tensor,
    top_k: int,
    position: int = -1,
    expert_chunk: int = DEFAULT_EXPERT_CHUNK,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute how each expert's margin changes when the residual at one position changes.

    Returns the residual, the margins, the router logits, and those derivatives for one
    batch.
    """
    h_BSH = _capture_residual(model, hookpoint=hookpoint, input_ids_BS=input_ids_BS)
    n_experts = backend.num_experts(model)
    pos = position % int(input_ids_BS.shape[1])
    aux: dict[str, Tensor] = {}
    f = _make_margins_fn(
        model,
        backend,
        hookpoint=hookpoint,
        router_layer_idx=router_layer_idx,
        input_ids_BS=input_ids_BS,
        top_k=top_k,
        position=position,
        aux=aux,
    )

    # has_aux=False returns (output, vjp_fn); the stub also includes the 3-tuple form.
    margins_BE, vjp_fn, *_ = torch.func.vjp(f, h_BSH)
    batch_size, _ = margins_BE.shape
    grad_chunks: list[Tensor] = []
    for start in range(0, n_experts, expert_chunk):
        stop = min(start + expert_chunk, n_experts)
        chunk = stop - start
        basis_CBE = margins_BE.new_zeros(chunk, batch_size, n_experts)
        rows_C = torch.arange(chunk, device=margins_BE.device)
        cols_C = torch.arange(start, stop, device=margins_BE.device)
        basis_CBE[rows_C, :, cols_C] = 1.0
        (grad_CBSH,) = torch.vmap(vjp_fn)(basis_CBE)
        grad_chunks.append(grad_CBSH[:, :, pos, :])
    grad_diag_EBH = torch.cat(grad_chunks, dim=0)

    return h_BSH.detach(), margins_BE.detach(), aux["logits_BE"], grad_diag_EBH


# ---------------------------------------------------------------------------
# Per-batch attribution -> aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ActiveFeatures:
    """Nonzero feature ids and values, one tensor per token."""

    feature_ids: list[Tensor]
    values: list[Tensor]


def _sparse_active(z_BF: Tensor) -> ActiveFeatures:
    """Return the nonzero feature ids and values at each token."""
    ids_list: list[Tensor] = []
    vals_list: list[Tensor] = []
    for b in range(z_BF.shape[0]):
        nz_A = (z_BF[b] > 0).nonzero(as_tuple=True)[0]
        ids_list.append(nz_A)
        vals_list.append(z_BF[b, nz_A])
    return ActiveFeatures(ids_list, vals_list)


@dataclass
class MarginAttribution:
    """Per-token routing-margin attribution for one batch at ``position``."""

    margins_BE: Tensor  # detached [B, E]
    topk_ids_BK: Tensor  # detached int64 [B, K]
    feature_ids: list[Tensor]  # per-sample active feature ids [A_b]
    feature_vals: list[Tensor]  # per-sample latent values [A_b]
    contributions_E: list[list[Tensor]]  # [E][B] -> per-sample c_f [A_b]


def attribute_batch_margins(
    model: PreTrainedModel,
    backend: ModelBackend,
    sae: SAEBackend,
    *,
    hookpoint: str,
    router_layer_idx: int,
    input_ids_BS: Tensor,
    top_k: int,
    position: int = -1,
    expert_chunk: int = DEFAULT_EXPERT_CHUNK,
) -> MarginAttribution:
    """Credit each expert's margin, at one position, to the SAE features active there.

    For feature ``f`` and expert ``i``, ``c_f = z_f * <d_f, dm_i / dh>``: the
    feature's activation times how far the margin moves along that feature's decoder
    direction.
    """
    h_BSH, margins_BE, logits_BE, grad_diag_EBH = margin_diagonal(
        model,
        backend,
        hookpoint=hookpoint,
        router_layer_idx=router_layer_idx,
        input_ids_BS=input_ids_BS,
        top_k=top_k,
        position=position,
        expert_chunk=expert_chunk,
    )
    device = h_BSH.device
    pos = position % int(input_ids_BS.shape[1])
    with torch.no_grad():
        z_BF = sae.encode(h_BSH[:, pos : pos + 1, :])[:, 0].float().cpu()
    active = _sparse_active(z_BF)
    feature_ids = active.feature_ids
    feature_vals = active.values
    topk_ids_BK = logits_BE.float().topk(top_k, dim=-1).indices.cpu()
    decoder_FH = sae.decoder_weights().to(device=device, dtype=h_BSH.dtype)

    n_experts = margins_BE.shape[-1]
    contributions_E: list[list[Tensor]] = []
    for expert_id in range(n_experts):
        grad_BH = grad_diag_EBH[expert_id]
        per_sample: list[Tensor] = []
        for b in range(input_ids_BS.shape[0]):
            ids_A = feature_ids[b]
            if ids_A.numel() == 0:
                per_sample.append(torch.empty(0, dtype=torch.float32))
                continue
            dirs_AH = decoder_FH[ids_A.to(device)]
            grad_H = grad_BH[b]
            scores_A = feature_vals[b].to(device=device, dtype=grad_H.dtype) * (
                dirs_AH @ grad_H
            )
            per_sample.append(scores_A.detach().float().cpu())
        contributions_E.append(per_sample)

    return MarginAttribution(
        margins_BE=margins_BE,
        topk_ids_BK=topk_ids_BK,
        feature_ids=feature_ids,
        feature_vals=feature_vals,
        contributions_E=contributions_E,
    )


def accumulate_attribution(state: MarginAggState, attr: MarginAttribution) -> None:
    """Fold one batch's :class: `MarginAttribution` into the aggregation state."""
    for expert_id, per_sample in enumerate(attr.contributions_E):
        y_active_B = (attr.topk_ids_BK == expert_id).any(dim=-1)
        accumulate_contributions(
            state,
            expert_id=expert_id,
            y_active_T=y_active_B,
            feature_ids_list=attr.feature_ids,
            scores_list=per_sample,
        )


def compute_routing_margin_scores(
    model: PreTrainedModel,
    backend: ModelBackend,
    sae: SAEBackend,
    *,
    hookpoint: str,
    router_layer_idx: int,
    tokens_NS: Tensor,
    top_k: int,
    n_experts: int,
    lambda_: float = 1.0,
    batch_size: int,
    position: int = -1,
    expert_chunk: int = DEFAULT_EXPERT_CHUNK,
    max_batches: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute routing-margin scores for one hookpoint, using train tokens only.

    Also returns coverage, the fraction of that expert's routed tokens where the feature
    is active, and effect, the mean positive contribution on those tokens.
    """
    hook_layer = core.parse_layer_index(hookpoint)
    upstream = (
        hook_layer <= router_layer_idx
        if residual_is_input(hookpoint)
        else hook_layer < router_layer_idx
    )
    if not upstream:
        msg = (
            f"Residual at {hookpoint!r} is not upstream of router layer "
            f"{router_layer_idx}; margin gradients would be zero"
        )
        raise ValueError(msg)

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    device = next(model.parameters()).device
    state = MarginAggState.create(n_experts, sae.n_features, lambda_=lambda_)
    n_batches = math.ceil(tokens_NS.shape[0] / batch_size)
    for batch_idx in tqdm(
        range(n_batches), desc=f"routing_margin {hookpoint}", unit="batch"
    ):
        if max_batches is not None and batch_idx >= max_batches:
            break
        input_ids_BS = tokens_NS[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        attr = attribute_batch_margins(
            model,
            backend,
            sae,
            hookpoint=hookpoint,
            router_layer_idx=router_layer_idx,
            input_ids_BS=input_ids_BS.to(device),
            top_k=top_k,
            position=position,
            expert_chunk=expert_chunk,
        )
        accumulate_attribution(state, attr)
    return state.finalize()


# ---------------------------------------------------------------------------
# rho_useful: difference of conditional mean activations (train tokens)
# ---------------------------------------------------------------------------


def compute_rho_useful_scores(
    input_dir: Path,
    hookpoint: str,
    *,
    n_features: int | None = None,
    device: str | None = None,
    index_chunk: int = 4_000_000,
) -> Tensor:
    """Compute how much higher each feature's mean activation is when an expert is routed
    than when it is not.

    Means are over train tokens, and a feature that is inactive at a token counts as
    zero.
    The result has shape ``[n_experts, n_features]``.
    """
    info = load_collect_info(input_dir)
    layer_idx = router_layer_for_hookpoint(hookpoint)
    n_experts = int(info["n_experts"])
    if n_features is None:
        sae_info = info.get("sae") or {}
        if hookpoint not in sae_info:
            msg = f"No SAE feature count for {hookpoint!r} in collect info"
            raise ValueError(msg)
        n_features = int(sae_info[hookpoint]["n_features"])
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    slug = core.hookpoint_slug(hookpoint)
    latent_shards = core.csr_shard_paths(input_dir / "latents" / slug)
    router_shards = sorted(
        (input_dir / "router" / f"layers_{layer_idx}").glob("shard-*.safetensors")
    )
    token_shards = sorted((input_dir / "tokens").glob("shard-*.safetensors"))
    if not latent_shards or not (
        len(latent_shards) == len(router_shards) == len(token_shards)
    ):
        msg = (
            f"Shard mismatch for {hookpoint!r}: {len(latent_shards)} latents, "
            f"{len(router_shards)} router, {len(token_shards)} tokens shards"
        )
        raise FileNotFoundError(msg)

    n_experts_f = n_experts * n_features
    sum_active_flat = torch.zeros(n_experts_f, dtype=torch.float32, device=device)
    total_sum_F = torch.zeros(n_features, dtype=torch.float32, device=device)
    count_active_E = torch.zeros(n_experts, dtype=torch.float32, device=device)
    n_train_tokens = 0

    shard_iter = zip(token_shards, router_shards, latent_shards, strict=True)
    for token_path, router_path, latent_path in tqdm(
        list(shard_iter), desc=f"rho_useful {hookpoint}", unit="shard"
    ):
        tokens_t = core.read_safetensors(token_path)
        seq_len = int(tokens_t[core.TOKENS].shape[1])
        train_T = np.repeat(tokens_t[TOKENS_SPLIT].astype(bool), seq_len)
        topk_ids_TK = torch.from_numpy(
            core.read_safetensors(router_path)[core.ROUTER_TOPK_IDS].astype(np.int64)
        ).to(device)
        csr = core.read_csr_shard(latent_path)

        counts_T = np.diff(csr.offsets)
        entry_token_A = np.repeat(np.arange(counts_T.shape[0]), counts_T)
        keep_A = train_T[entry_token_A]
        ids_A = torch.from_numpy(csr.feature_ids[keep_A].astype(np.int64)).to(device)
        vals_A = torch.from_numpy(csr.values[keep_A].astype(np.float32)).to(device)
        etok_A = torch.from_numpy(entry_token_A[keep_A].astype(np.int64)).to(device)

        n_train_tokens += int(train_T.sum())
        train_mask_T = torch.from_numpy(train_T).to(device)
        count_active_E += torch.bincount(
            topk_ids_TK[train_mask_T].reshape(-1), minlength=n_experts
        ).float()
        total_sum_F.index_add_(0, ids_A, vals_A)
        top_k = topk_ids_TK.shape[1]
        for start in range(0, ids_A.shape[0], index_chunk):
            ids_C = ids_A[start : start + index_chunk]
            vals_C = vals_A[start : start + index_chunk]
            experts_CK = topk_ids_TK[etok_A[start : start + index_chunk]]
            flat_CK = experts_CK * n_features + ids_C.unsqueeze(1)
            sum_active_flat.index_add_(
                0, flat_CK.reshape(-1), vals_C.repeat_interleave(top_k)
            )

    sum_active_EF = sum_active_flat.view(n_experts, n_features)
    count_inactive_E = float(n_train_tokens) - count_active_E
    mean_active_EF = sum_active_EF / count_active_E.clamp(min=1).unsqueeze(1)
    mean_inactive_EF = (total_sum_F.unsqueeze(0) - sum_active_EF) / (
        count_inactive_E.clamp(min=1).unsqueeze(1)
    )
    return (mean_active_EF - mean_inactive_EF).cpu()


# ---------------------------------------------------------------------------
# cosine: router weight rows vs SAE decoder directions
# ---------------------------------------------------------------------------


@torch.no_grad()
def cosine_scores(router_weights_EH: Tensor, decoder_FH: Tensor) -> Tensor:
    """Compute the cosine similarity between router rows and decoder directions."""
    # Decoder weights are trainable parameters. Scores are data, not a graph.
    router_norm_EH = F.normalize(router_weights_EH.float(), p=2, dim=1)
    decoder_norm_FH = F.normalize(decoder_FH.float(), p=2, dim=1)
    return router_norm_EH @ decoder_norm_FH.T


# ---------------------------------------------------------------------------
# Top-n selection with optional exclusive assignment
# ---------------------------------------------------------------------------


def select_top_features(
    scores_EF: Tensor,
    n: int,
    *,
    exclusive: bool,
) -> dict[int, list[FeatureScore]]:
    """Select the ``n`` highest-scoring features for each expert.

    With ``exclusive``, scores are taken from strongest to weakest, and a feature
    already given to one expert is not given to another.
    """
    n_experts, n_features = scores_EF.shape
    per_expert = min(n, n_features)
    if per_expert < 1:
        msg = f"n must be >= 1, got {n}"
        raise ValueError(msg)

    if not exclusive:
        result: dict[int, list[FeatureScore]] = {}
        for expert_id in range(n_experts):
            vals_N, idx_N = scores_EF[expert_id].topk(per_expert)
            result[expert_id] = [
                FeatureScore(int(idx), float(val))
                for idx, val in zip(idx_N.tolist(), vals_N.tolist(), strict=True)
            ]
        return result

    target_total = n_experts * per_expert
    order = scores_EF.reshape(-1).argsort(descending=True)
    assigned_F = torch.zeros(n_features, dtype=torch.bool)
    buckets: list[list[FeatureScore]] = [[] for _ in range(n_experts)]
    n_assigned = 0
    for flat_idx in order.tolist():
        if n_assigned >= target_total:
            break
        expert_id, feature_id = divmod(flat_idx, n_features)
        if assigned_F[feature_id] or len(buckets[expert_id]) >= per_expert:
            continue
        buckets[expert_id].append(
            FeatureScore(feature_id, float(scores_EF[expert_id, feature_id]))
        )
        assigned_F[feature_id] = True
        n_assigned += 1
    for expert_id, feats in enumerate(buckets):
        if not feats:
            logger.warning(f"Expert {expert_id}: no features after greedy dedup")
    logger.info(
        f"Exclusive assignment: {n_assigned}/{target_total} cells filled "
        f"(min/expert={min(len(b) for b in buckets)})"
    )
    return dict(enumerate(buckets))


# ---------------------------------------------------------------------------
# Token loading (train/test split from collect shards)
# ---------------------------------------------------------------------------


def load_token_split(
    input_dir: Path,
    *,
    split: Literal["train", "test"] = "train",
    n_tokens_cap: int | None = None,
) -> Tensor:
    """Load the token blocks of one split, concatenated in shard order."""
    token_shards = sorted((input_dir / "tokens").glob("shard-*.safetensors"))
    if not token_shards:
        msg = f"No tokens shards under {input_dir / 'tokens'}"
        raise FileNotFoundError(msg)
    parts: list[np.ndarray] = []
    n_tokens = 0
    for path in token_shards:
        tensors = core.read_safetensors(path)
        split_N = tensors[TOKENS_SPLIT].astype(bool)
        mask_N = split_N if split == "train" else ~split_N
        block_NS = tensors[core.TOKENS][mask_N]
        parts.append(block_NS)
        n_tokens += block_NS.size
        if n_tokens_cap is not None and n_tokens >= n_tokens_cap:
            break
    tokens_NS = np.concatenate(parts)
    if tokens_NS.shape[0] == 0:
        msg = f"No {split!r} tokens found under {input_dir}"
        raise ValueError(msg)
    tokens_NS = torch.from_numpy(tokens_NS.astype(np.int64))
    if n_tokens_cap is not None:
        max_seqs = max(n_tokens_cap // tokens_NS.shape[1], 1)
        tokens_NS = tokens_NS[:max_seqs]
    return tokens_NS


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _require_sae(collect_config: CollectConfig) -> SAEConfig:
    if collect_config.sae is None:
        msg = "The collect run has no SAE config; this method needs SAE weights"
        raise ValueError(msg)
    return collect_config.sae


def run_select(
    config: SelectConfig,
    *,
    run_name: str = "select",
    expert_chunk: int = DEFAULT_EXPERT_CHUNK,
    position: int = -1,
) -> Path:
    """Score features and write the top-n table for ``config.n``.

    ``expert_chunk`` and ``position`` apply only to the routing-margin method.
    """
    collect_manifest = core.read_manifest(config.input_dir)
    if collect_manifest.stage != "collect":
        msg = f"{config.input_dir} is not a collect run"
        raise ValueError(msg)
    parents = {"collect": collect_manifest.config_hash}
    if core.manifest_matches(config.output_dir, config, parents):
        logger.info(f"select up to date -> {config.output_dir}")
        return config.output_dir
    core.set_global_seed(config.seed)
    info = load_collect_info(config.input_dir)

    scores_by_hp: dict[str, Tensor] = {}
    diagnostics_by_hp: dict[str, dict[str, Tensor]] = {}

    if config.method == "rho_useful":
        # calculate from cached activations
        for hp in config.hookpoints:
            scores_by_hp[hp] = compute_rho_useful_scores(config.input_dir, hp)
    elif config.method in ("cosine", "routing_margin"):
        collect_config = load_collect_config(config.input_dir)
        backend = get_model_backend(collect_config.model)
        model = backend.load_model()
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        device = next(model.parameters()).device
        backend.validate(model, config.hookpoints)
        tokens_NS: Tensor | None = None
        if config.method == "routing_margin":
            tokens_NS = load_token_split(
                config.input_dir, split="train", n_tokens_cap=config.n_tokens
            )
        for hp in config.hookpoints:
            sae = get_sae_backend(_require_sae(collect_config), hp).to(str(device))
            layer_idx = router_layer_for_hookpoint(hp)
            if config.method == "cosine":
                scores_by_hp[hp] = cosine_scores(
                    backend.router_weight_matrix(model, layer_idx),
                    sae.decoder_weights(),
                ).cpu()
            else:
                assert tokens_NS is not None
                batch_size = config.batch_size or int(info["batch_size"])
                scores_EF, coverage_EF, effect_EF = compute_routing_margin_scores(
                    model,
                    backend,
                    sae,
                    hookpoint=hp,
                    router_layer_idx=layer_idx,
                    tokens_NS=tokens_NS,
                    top_k=int(info["top_k"]),
                    n_experts=int(info["n_experts"]),
                    lambda_=config.lambda_,
                    batch_size=batch_size,
                    position=position,
                    expert_chunk=expert_chunk,
                )
                scores_by_hp[hp] = scores_EF
                diagnostics_by_hp[hp] = {
                    core.COVERAGE: coverage_EF,
                    core.EFFECT: effect_EF,
                }
    else:
        msg = f"Unknown select method: {config.method!r}"
        raise ValueError(msg)

    for hp, scores_EF in scores_by_hp.items():
        tensors = {core.SCORES: scores_EF.detach().float().cpu().numpy()}
        for key, diag_EF in diagnostics_by_hp.get(hp, {}).items():
            tensors[key] = diag_EF.detach().float().cpu().numpy()
        core.write_safetensors(
            config.output_dir
            / "scores"
            / config.method
            / f"{core.hookpoint_slug(hp)}.safetensors",
            tensors,
        )

    n = config.n
    rows: list[dict[str, object]] = []
    for hp in config.hookpoints:
        selection = select_top_features(scores_by_hp[hp], n, exclusive=config.exclusive)
        for expert_id, features in selection.items():
            for rank, pick in enumerate(features):
                rows.append(
                    {
                        "method": config.method,
                        "hookpoint": hp,
                        "expert_id": expert_id,
                        "n": n,
                        "rank": rank,
                        "feature_id": pick.feature_id,
                        "score": pick.score,
                    }
                )
    core.write_parquet(
        config.output_dir / "selections" / config.method / f"n{n}.parquet",
        pa.Table.from_pylist(rows, schema=SELECTION_SCHEMA),
    )
    logger.info(
        f"Wrote {len(rows)} selection rows for n={n} "
        f"({config.method}, {len(config.hookpoints)} hookpoints)"
    )

    manifest = core.build_manifest(
        stage="select",
        run_name=run_name,
        config=config,
        parents=parents,
        dataset_revision=collect_manifest.dataset_revision,
        model_revision=collect_manifest.model_revision,
        sae_revision=collect_manifest.sae_revision,
        provenance={
            "expert_chunk": expert_chunk,
            "position": position,
            "n": config.n,
        },
    )
    core.write_manifest(config.output_dir, manifest)
    logger.success(f"select ({config.method}) completed -> {config.output_dir}")
    return config.output_dir
