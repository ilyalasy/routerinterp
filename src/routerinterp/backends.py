"""Protocols and adapters for models and SAEs."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, runtime_checkable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from loguru import logger
from torch import Tensor
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from routerinterp.core import (
    ModelConfig,
    SAEConfig,
    normalize_hookpoint,
    parse_layer_index,
    residual_is_input,
    router_layer_for_hookpoint,
    write_json,
    write_safetensors,
)

if TYPE_CHECKING:
    # `sae` dependency group. Imported in the loaders so the CLI can run
    # without those libraries installed.
    from dictionary_learning.dictionary import Dictionary
    from sparsify import SparseCoder

    from routerinterp.core import SAESourceConfig

RoutingStyle = Literal["olmoe", "gpt_oss"]

# layer_idx, router logits [T, E] (pre-softmax, flattened tokens)
RouterCallback = Callable[[int, Tensor], None]
# in-graph pre-softmax router logits [T, E]; may raise to abort the forward
RouterLogitCallback = Callable[[Tensor], None]
# hookpoint, hidden states [B, S, H] (detached)
ActivationCallback = Callable[[str, Tensor], None]
# layer_idx, expert_idx, token_idx [R], expert writes [R, H] (unscaled)
ExpertWriteCallback = Callable[[int, int, Tensor, Tensor], None]
Closer = Callable[[], None]


@runtime_checkable
class ModelBackend(Protocol):
    """HF MoE model access: residuals, router logits, expert writes."""

    config: ModelConfig
    model_type: str
    routing_style: RoutingStyle

    def load_model(self) -> PreTrainedModel: ...
    def load_tokenizer(self) -> PreTrainedTokenizerBase: ...
    def num_experts(self, model: PreTrainedModel) -> int: ...
    def top_k(self, model: PreTrainedModel) -> int: ...
    def get_layers(self, model: PreTrainedModel) -> list[nn.Module]: ...
    def resolve_module(self, model: PreTrainedModel, act_module: str) -> nn.Module: ...

    def router_module(self, model: PreTrainedModel, layer_idx: int) -> nn.Module:
        """Return the router linear of one layer."""
        ...

    def router_weight_matrix(self, model: PreTrainedModel, layer_idx: int) -> Tensor:
        """Return the router weight matrix ``[E, H]`` for one layer."""
        ...

    def install_router_logit_hook(
        self,
        model: PreTrainedModel,
        layer_idx: int,
        on_logits: RouterLogitCallback,
    ) -> torch.utils.hooks.RemovableHandle:
        """Hook one layer's router so ``on_logits`` receives in-graph logits.

        A raise from the callback aborts the rest of the forward.
        """
        ...

    def install_captures(
        self,
        model: PreTrainedModel,
        *,
        activation_hookpoints: Sequence[str],
        router_layers: Sequence[int],
        expert_write_layers: Sequence[int],
        on_activation: ActivationCallback,
        on_router_logits: RouterCallback,
        on_expert_write: ExpertWriteCallback,
    ) -> Closer:
        """Install forward hooks; return a closer that restores the model."""
        ...

    def validate(self, model: PreTrainedModel, hookpoints: Sequence[str]) -> None:
        """Check that layers, hookpoints, dimensions, and revisions line up."""
        ...


@runtime_checkable
class SAEBackend(Protocol):
    """SAE encode/decode access for one hookpoint."""

    hookpoint: str

    @property
    def n_features(self) -> int: ...
    @property
    def revision(self) -> str | None: ...

    def encode(self, x_BSH: Tensor) -> Tensor:
        """Encode residuals [B, S, H] -> latents [B, S, F] (>= 0)."""
        ...

    def decoder_weights(self) -> Tensor:
        """Return decoder directions ``[F, H]``."""
        ...

    def to(self, device: str) -> SAEBackend: ...


def get_model_backend(config: ModelConfig) -> ModelBackend:
    """Resolve the adapter for ``config.name``'s ``model_type``."""
    hf_config = AutoConfig.from_pretrained(
        config.name,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
    )
    model_type = getattr(hf_config, "model_type", None)
    backend_cls = (
        MODEL_BACKEND_REGISTRY.get(model_type) if isinstance(model_type, str) else None
    )
    if backend_cls is None:
        supported = ", ".join(sorted(MODEL_BACKEND_REGISTRY))
        msg = (
            f"Unsupported model_type {model_type!r} for model {config.name!r}. "
            f"Add an entry to MODEL_BACKEND_REGISTRY. Supported: {supported}"
        )
        raise ValueError(msg)
    logger.info(f"Using model backend {backend_cls.__name__} ({model_type})")
    return backend_cls(config)


def get_sae_backend(config: SAEConfig, hookpoint: str) -> SAEBackend:
    """Resolve the SAE backend for one hookpoint."""
    if hookpoint not in config.hookpoints:
        msg = f"Hookpoint {hookpoint!r} not in sae.hookpoints {config.hookpoints}"
        raise ValueError(msg)
    source = config.source
    if source.type == "sparsify_wandb":
        root, revision = _download_wandb_artifact(source, config)
        return SparsifySAE.from_dir(root, hookpoint, revision=revision)
    if source.type == "hf_dictionary_learning":
        return DictionaryLearningSAE.from_hf(source, hookpoint)
    # local: directory with SafeTensors + config.json; sniff the format.
    if source.path is None:
        msg = f"source.path is required for SAE source type {source.type!r}"
        raise ValueError(msg)
    if _resolve_hookpoint_dir(source.path, hookpoint, _is_sparsify_dir) is not None:
        return SparsifySAE.from_dir(source.path, hookpoint)
    if (
        _resolve_hookpoint_dir(source.path, hookpoint, _is_dictionary_learning_dir)
        is not None
    ):
        return DictionaryLearningSAE.from_dir(
            source.path, hookpoint, trust_pickle=source.trust_pickle
        )
    msg = (
        f"No SAE checkpoint for hookpoint {hookpoint!r} under {source.path} "
        "(expected sparsify cfg.json + sae.safetensors, or dictionary_learning "
        "config.json + ae.safetensors)"
    )
    raise FileNotFoundError(msg)


# ===========================================================================
# Model backends
# ===========================================================================
# normalize_hookpoint, residual_is_input, and router_layer_for_hookpoint
# are imported from routerinterp.core.


class _OlmoeSparseMoe(Protocol):
    experts: Sequence[nn.Module]
    gate: Callable[[Tensor], Tensor]
    top_k: int
    num_experts: int


class _GptOssMlp(Protocol):
    experts: nn.Module


class _GptOssExperts(Protocol):
    hidden_size: int
    num_experts: int
    gate_up_proj: Tensor
    gate_up_proj_bias: Tensor
    down_proj: Tensor
    down_proj_bias: Tensor
    limit: float
    alpha: float


def _unpack_hidden(value: object) -> Tensor:
    hidden = value[0] if isinstance(value, tuple) else value
    if not isinstance(hidden, Tensor):
        msg = f"Expected Tensor hidden state, got {type(hidden)}"
        raise TypeError(msg)
    return hidden


def _layer_mlp(layer: nn.Module) -> nn.Module:
    mlp = getattr(layer, "mlp", None)
    if not isinstance(mlp, nn.Module):
        msg = f"Layer {type(layer)} has no mlp module"
        raise TypeError(msg)
    return mlp


def _require_attrs(module: nn.Module, *names: str, label: str) -> None:
    missing = [name for name in names if not hasattr(module, name)]
    if missing:
        msg = f"{label} missing attributes: {', '.join(missing)}"
        raise TypeError(msg)


def _get_layers(model: PreTrainedModel) -> list[nn.Module]:
    """Locate the transformer layer list on a Hugging Face MoE model."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return list(cast(Sequence[nn.Module], inner.layers))
    layers = getattr(model, "layers", None)
    if layers is not None:
        return list(cast(Sequence[nn.Module], layers))
    msg = f"Could not find layers in model {type(model)}"
    raise ValueError(msg)


def _resolve_module(model: PreTrainedModel, act_module: str) -> nn.Module:
    """Locate ``act_module`` under the HF backbone (``base_model`` / ``model`` )."""
    roots: list[nn.Module] = []
    base_model = getattr(model, "base_model", None)
    if isinstance(base_model, nn.Module):
        roots.append(base_model)
    inner = getattr(model, "model", None)
    if isinstance(inner, nn.Module) and inner not in roots:
        roots.append(inner)
    roots.append(cast(nn.Module, model))

    for root in roots:
        for name, module in root.named_modules():
            if name == act_module:
                return module

    msg = f"Activation module {act_module!r} not found in model"
    raise ValueError(msg)


def _residual_pre_hook(
    _module: nn.Module,
    args: tuple[object, ...],
    *,
    hookpoint: str,
    on_activation: ActivationCallback,
) -> None:
    hidden = _unpack_hidden(args[0] if len(args) == 1 else args)
    on_activation(hookpoint, hidden.detach())


def _residual_post_hook(
    _module: nn.Module,
    _args: tuple[object, ...],
    output: Tensor | tuple[object, ...],
    *,
    hookpoint: str,
    on_activation: ActivationCallback,
) -> None:
    hidden = _unpack_hidden(output)
    on_activation(hookpoint, hidden.detach())


def install_residual_hooks(
    model: PreTrainedModel,
    hookpoints: Sequence[str],
    on_activation: ActivationCallback,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Call back with the residual at each hookpoint.

    ``layers.N.mlp`` is hooked on the module input; any other hookpoint is hooked on the
    module output.
    If the callback raises, the rest of the forward pass stops.
    """
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for hp in hookpoints:
        act_module = normalize_hookpoint(hp)
        module = _resolve_module(model, act_module)
        if residual_is_input(act_module):
            hook = partial(
                _residual_pre_hook,
                hookpoint=act_module,
                on_activation=on_activation,
            )
            handles.append(module.register_forward_pre_hook(hook))
        else:
            hook = partial(
                _residual_post_hook,
                hookpoint=act_module,
                on_activation=on_activation,
            )
            handles.append(module.register_forward_hook(hook))
    return handles


def _replace_residual_input(
    _module: nn.Module,
    args: tuple[object, ...],
    *,
    hidden_BSH: Tensor,
) -> tuple[object, ...]:
    return (hidden_BSH, *args[1:])


def _replace_residual_output(
    _module: nn.Module,
    _args: tuple[object, ...],
    output: object,
    *,
    hidden_BSH: Tensor,
) -> object:
    if isinstance(output, tuple):
        return (hidden_BSH, *output[1:])
    return hidden_BSH


def install_residual_replacement(
    model: PreTrainedModel,
    hookpoint: str,
    hidden_BSH: Tensor,
) -> torch.utils.hooks.RemovableHandle:
    """Replace the residual at ``hookpoint`` with ``hidden_BSH`` for one forward pass.

    ``layers.N.mlp`` is replaced at the module input; any other hookpoint is replaced at
    the module output.
    The replacement stays in the autograd graph.
    """
    act_module = normalize_hookpoint(hookpoint)
    module = _resolve_module(model, act_module)
    if residual_is_input(act_module):
        hook = partial(_replace_residual_input, hidden_BSH=hidden_BSH)
        return module.register_forward_pre_hook(hook)
    hook = partial(_replace_residual_output, hidden_BSH=hidden_BSH)
    return module.register_forward_hook(hook)


class _HFModelBackend:
    """Shared Hugging Face plumbing for the concrete MoE backends."""

    config: ModelConfig
    model_type: str
    routing_style: RoutingStyle
    router_attr: str  # child of ``layer.mlp`` that is the router linear

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def load_model(self) -> PreTrainedModel:
        dtype = getattr(torch, self.config.dtype)
        device_map: str | torch.device = (
            "auto" if self.config.device == "auto" else torch.device(self.config.device)
        )
        logger.info(
            f"Loading model {self.config.name} (revision={self.config.revision}, "
            f"dtype={self.config.dtype}, device_map={device_map})"
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.config.name,
            revision=self.config.revision,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=self.config.trust_remote_code,
        )
        model.eval()
        return model

    def load_tokenizer(self) -> PreTrainedTokenizerBase:
        return AutoTokenizer.from_pretrained(
            self.config.name,
            revision=self.config.revision,
            trust_remote_code=self.config.trust_remote_code,
        )

    def num_experts(self, model: PreTrainedModel) -> int:
        config = model.config
        n_experts = getattr(config, "num_local_experts", None) or getattr(
            config, "num_experts", None
        )
        if n_experts is None:
            msg = (
                f"Model config {type(config).__name__} has neither "
                "num_local_experts nor num_experts"
            )
            raise ValueError(msg)
        return int(n_experts)

    def top_k(self, model: PreTrainedModel) -> int:
        k = getattr(model.config, "num_experts_per_tok", None)
        if k is None:
            msg = (
                f"Model config {type(model.config).__name__} has no num_experts_per_tok"
            )
            raise ValueError(msg)
        return int(k)

    def get_layers(self, model: PreTrainedModel) -> list[nn.Module]:
        return _get_layers(model)

    def resolve_module(self, model: PreTrainedModel, act_module: str) -> nn.Module:
        return _resolve_module(model, normalize_hookpoint(act_module))

    def router_module(self, model: PreTrainedModel, layer_idx: int) -> nn.Module:
        layers = self.get_layers(model)
        if not 0 <= layer_idx < len(layers):
            msg = f"Layer index {layer_idx} out of bounds (num_layers={len(layers)})"
            raise ValueError(msg)
        mlp = _layer_mlp(layers[layer_idx])
        router = getattr(mlp, self.router_attr, None)
        if not isinstance(router, nn.Module) or not hasattr(router, "weight"):
            msg = f"Layer {layer_idx} mlp has no {self.router_attr!r} with a weight"
            raise ValueError(msg)
        return router

    def router_weight_matrix(self, model: PreTrainedModel, layer_idx: int) -> Tensor:
        weight = getattr(self.router_module(model, layer_idx), "weight", None)
        if not isinstance(weight, Tensor):
            msg = f"Layer {layer_idx} {self.router_attr!r} weight is not a Tensor"
            raise TypeError(msg)
        return weight

    def install_router_logit_hook(
        self,
        model: PreTrainedModel,
        layer_idx: int,
        on_logits: RouterLogitCallback,
    ) -> torch.utils.hooks.RemovableHandle:
        raise NotImplementedError

    def install_captures(
        self,
        model: PreTrainedModel,
        *,
        activation_hookpoints: Sequence[str],
        router_layers: Sequence[int],
        expert_write_layers: Sequence[int],
        on_activation: ActivationCallback,
        on_router_logits: RouterCallback,
        on_expert_write: ExpertWriteCallback,
    ) -> Closer:
        raise NotImplementedError

    def validate(self, model: PreTrainedModel, hookpoints: Sequence[str]) -> None:
        model_type = getattr(model.config, "model_type", None)
        if model_type != self.model_type:
            msg = (
                f"{type(self).__name__} expects model_type {self.model_type!r}, "
                f"got {model_type!r}"
            )
            raise ValueError(msg)
        n_experts = self.num_experts(model)
        k = self.top_k(model)
        if not 0 < k <= n_experts:
            msg = f"Invalid routing config: top_k={k}, num_experts={n_experts}"
            raise ValueError(msg)
        layers = self.get_layers(model)
        for hookpoint in hookpoints:
            module = self.resolve_module(model, hookpoint)
            router_layer = router_layer_for_hookpoint(hookpoint)
            if router_layer >= len(layers):
                msg = (
                    f"Router layer {router_layer} for hookpoint {hookpoint!r} out "
                    f"of bounds (num_layers={len(layers)})"
                )
                raise ValueError(msg)
            _layer_mlp(layers[router_layer])
            logger.debug(f"Hookpoint {hookpoint!r} -> {type(module).__name__}")
        logger.info(
            f"Validated {len(hookpoints)} hookpoint(s) for {self.model_type}: "
            f"num_layers={len(layers)}, num_experts={n_experts}, top_k={k}"
        )


@dataclass
class _OlmoeRoutingCtx:
    expert_mask_EKT: Tensor  # (E, K, T)
    batch_size: int
    seq_len: int


def _olmoe_router_logits_hook(
    _module: nn.Module,
    _args: tuple[object, ...],
    output: Tensor | tuple[object, ...],
    *,
    on_logits: RouterLogitCallback,
) -> None:
    """Read OLMoE gate logits before the softmax."""
    on_logits(_unpack_hidden(output))


class OlmoeBackend(_HFModelBackend):
    """OLMoE: HF ``output_router_logits`` + optional per-expert write hooks."""

    model_type = "olmoe"
    routing_style: RoutingStyle = "olmoe"
    router_attr = "gate"

    def install_router_logit_hook(
        self,
        model: PreTrainedModel,
        layer_idx: int,
        on_logits: RouterLogitCallback,
    ) -> torch.utils.hooks.RemovableHandle:
        gate = self.router_module(model, layer_idx)
        hook = partial(_olmoe_router_logits_hook, on_logits=on_logits)
        return gate.register_forward_hook(hook)

    def install_captures(
        self,
        model: PreTrainedModel,
        *,
        activation_hookpoints: Sequence[str],
        router_layers: Sequence[int],
        expert_write_layers: Sequence[int],
        on_activation: ActivationCallback,
        on_router_logits: RouterCallback,
        on_expert_write: ExpertWriteCallback,
    ) -> Closer:
        handles = install_residual_hooks(model, activation_hookpoints, on_activation)
        router_set = set(router_layers)
        expert_set = set(expert_write_layers)
        routing_ctx: dict[int, _OlmoeRoutingCtx] = {}
        original_forward = model.forward if router_set else None

        if router_set:

            def forward_with_router(*args: object, **kwargs: object) -> object:
                kwargs["output_router_logits"] = True
                assert original_forward is not None
                outputs = original_forward(*args, **kwargs)
                router_logits = getattr(outputs, "router_logits", None)
                if router_logits is None:
                    msg = "OLMoE forward did not return router_logits"
                    raise ValueError(msg)
                for layer_idx, logits_TE in enumerate(router_logits):
                    if logits_TE is None or layer_idx not in router_set:
                        continue
                    on_router_logits(layer_idx, logits_TE)
                return outputs

            model.forward = forward_with_router  # type: ignore[method-assign]

        if expert_set:
            layers = self.get_layers(model)
            for layer_idx in expert_set:
                if layer_idx >= len(layers):
                    msg = f"Layer index {layer_idx} out of bounds"
                    raise ValueError(msg)
                moe_module = _layer_mlp(layers[layer_idx])
                _require_attrs(
                    moe_module,
                    "experts",
                    "gate",
                    "top_k",
                    "num_experts",
                    label=f"Layer {layer_idx} mlp",
                )
                olmoe_moe = cast(_OlmoeSparseMoe, moe_module)

                def _make_pre(idx: int = layer_idx):
                    def pre_hook(module: nn.Module, args: tuple[object, ...]) -> None:
                        hidden_BSH = cast(Tensor, args[0])
                        batch_size, seq_len, hidden_dim = hidden_BSH.shape
                        flat_TH = hidden_BSH.view(-1, hidden_dim)
                        block = cast(_OlmoeSparseMoe, module)
                        k = int(block.top_k)
                        n_exp = int(block.num_experts)
                        router_logits_TE = block.gate(flat_TH)
                        routing_weights_TE = F.softmax(
                            router_logits_TE, dim=1, dtype=torch.float
                        )
                        _, selected_TK = torch.topk(routing_weights_TE, k, dim=-1)
                        expert_mask_EKT = F.one_hot(
                            selected_TK, num_classes=n_exp
                        ).permute(2, 1, 0)
                        routing_ctx[idx] = _OlmoeRoutingCtx(
                            expert_mask_EKT=expert_mask_EKT,
                            batch_size=batch_size,
                            seq_len=seq_len,
                        )

                    return pre_hook

                handles.append(moe_module.register_forward_pre_hook(_make_pre()))
                for expert_idx, expert in enumerate(olmoe_moe.experts):

                    def _make_expert(lidx: int = layer_idx, eidx: int = expert_idx):
                        def hook(
                            _module: nn.Module,
                            _args: tuple[object, ...],
                            output_RH: Tensor,
                        ) -> None:
                            ctx = routing_ctx.get(lidx)
                            if ctx is None:
                                return
                            _slot_R, token_idx_R = torch.where(
                                ctx.expert_mask_EKT[eidx]
                            )
                            if token_idx_R.numel() == 0 or output_RH.numel() == 0:
                                return
                            on_expert_write(
                                lidx, eidx, token_idx_R.detach(), output_RH.detach()
                            )

                        return hook

                    handles.append(expert.register_forward_hook(_make_expert()))

        def closer() -> None:
            for handle in handles:
                handle.remove()
            if original_forward is not None:
                model.forward = original_forward  # type: ignore[method-assign]
            routing_ctx.clear()

        return closer


def _gpt_oss_router_logits_hook(
    module: nn.Module,
    args: tuple[object, ...],
    *,
    on_logits: RouterLogitCallback,
) -> None:
    """Read GPT-OSS router logits from the linear that runs before the top-k."""
    hidden = _unpack_hidden(args[0] if len(args) == 1 else args)
    flat_TH = hidden.reshape(-1, hidden.shape[-1])
    weight = getattr(module, "weight", None)
    bias = getattr(module, "bias", None)
    if not isinstance(weight, Tensor):
        msg = f"GPT-OSS router weight is not a Tensor, got {type(weight)}"
        raise TypeError(msg)
    if not isinstance(bias, Tensor):
        msg = f"GPT-OSS router bias is not a Tensor, got {type(bias)}"
        raise TypeError(msg)
    on_logits(F.linear(flat_TH, weight, bias))


def _forward_router_logits(
    logits_TE: Tensor, *, layer_idx: int, on_router_logits: RouterCallback
) -> None:
    on_router_logits(layer_idx, logits_TE)


class GptOssBackend(_HFModelBackend):
    """GPT-OSS: pre-topk router hook + fused-expert write capture wrapper."""

    model_type = "gpt_oss"
    routing_style: RoutingStyle = "gpt_oss"
    router_attr = "router"

    def install_router_logit_hook(
        self,
        model: PreTrainedModel,
        layer_idx: int,
        on_logits: RouterLogitCallback,
    ) -> torch.utils.hooks.RemovableHandle:
        router = self.router_module(model, layer_idx)
        hook = partial(_gpt_oss_router_logits_hook, on_logits=on_logits)
        return router.register_forward_pre_hook(hook)

    def install_captures(
        self,
        model: PreTrainedModel,
        *,
        activation_hookpoints: Sequence[str],
        router_layers: Sequence[int],
        expert_write_layers: Sequence[int],
        on_activation: ActivationCallback,
        on_router_logits: RouterCallback,
        on_expert_write: ExpertWriteCallback,
    ) -> Closer:
        handles = install_residual_hooks(model, activation_hookpoints, on_activation)
        layers = self.get_layers(model)
        original_forwards: dict[int, Callable[..., Tensor]] = {}
        wrapped_experts: dict[int, nn.Module] = {}

        for layer_idx in router_layers:
            handles.append(
                self.install_router_logit_hook(
                    model,
                    layer_idx,
                    partial(
                        _forward_router_logits,
                        layer_idx=layer_idx,
                        on_router_logits=on_router_logits,
                    ),
                )
            )

        for layer_idx in expert_write_layers:
            if layer_idx >= len(layers):
                msg = f"Layer index {layer_idx} out of bounds"
                raise ValueError(msg)
            mlp_module = _layer_mlp(layers[layer_idx])
            _require_attrs(
                mlp_module, "experts", "router", label=f"Layer {layer_idx} mlp"
            )
            experts = cast(_GptOssMlp, mlp_module).experts
            wrapped_experts[layer_idx] = experts
            original_forwards[layer_idx] = experts.forward
            experts.forward = self._make_experts_forward(  # type: ignore[method-assign]
                layer_idx, experts, original_forwards[layer_idx], on_expert_write
            )

        def closer() -> None:
            for handle in handles:
                handle.remove()
            for layer_idx, experts in wrapped_experts.items():
                original = original_forwards.get(layer_idx)
                if original is not None:
                    experts.forward = original  # type: ignore[method-assign]
            wrapped_experts.clear()
            original_forwards.clear()

        return closer

    def _make_experts_forward(
        self,
        layer_idx: int,
        experts: nn.Module,
        original_forward: Callable[..., Tensor],
        on_expert_write: ExpertWriteCallback,
    ) -> Callable[..., Tensor]:
        def wrapped(
            hidden_states: Tensor,
            router_indices: Tensor | None = None,
            routing_weights: Tensor | None = None,
        ) -> Tensor:
            if router_indices is None or routing_weights is None:
                return original_forward(
                    hidden_states,
                    router_indices=router_indices,
                    routing_weights=routing_weights,
                )
            with torch.no_grad():
                self._capture_gpt_oss_writes(
                    layer_idx,
                    cast(_GptOssExperts, experts),
                    hidden_states,
                    router_indices,
                    on_expert_write,
                )
            return original_forward(
                hidden_states,
                router_indices=router_indices,
                routing_weights=routing_weights,
            )

        return wrapped

    def _capture_gpt_oss_writes(
        self,
        layer_idx: int,
        experts: _GptOssExperts,
        hidden_states: Tensor,
        router_indices: Tensor,
        on_expert_write: ExpertWriteCallback,
    ) -> None:
        """Capture unscaled expert writes for each routed token and expert."""
        flat_TH = hidden_states.reshape(-1, experts.hidden_size)
        n_exp = int(experts.num_experts)
        expert_mask_EKT = F.one_hot(router_indices, num_classes=n_exp)
        expert_mask_EKT = expert_mask_EKT.permute(2, 1, 0)  # (E, K, T)
        expert_hit = torch.greater(expert_mask_EKT.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx_t in expert_hit:
            expert_idx = int(expert_idx_t[0])
            _slot_R, token_idx_R = torch.where(expert_mask_EKT[expert_idx])
            if token_idx_R.numel() == 0:
                continue
            current_state_RH = flat_TH[token_idx_R]
            gate_up_RI2 = (
                current_state_RH @ experts.gate_up_proj[expert_idx]
                + experts.gate_up_proj_bias[expert_idx]
            )
            gate_RI, up_RI = gate_up_RI2[..., ::2], gate_up_RI2[..., 1::2]
            gate_RI = gate_RI.clamp(min=None, max=experts.limit)
            up_RI = up_RI.clamp(min=-experts.limit, max=experts.limit)
            glu_RI = gate_RI * torch.sigmoid(gate_RI * experts.alpha)
            gated_output_RI = (up_RI + 1) * glu_RI
            out_RH = (
                gated_output_RI @ experts.down_proj[expert_idx]
                + experts.down_proj_bias[expert_idx]
            )
            on_expert_write(
                layer_idx, expert_idx, token_idx_R.detach(), out_RH.detach()
            )


MODEL_BACKEND_REGISTRY: dict[str, type[_HFModelBackend]] = {
    "olmoe": OlmoeBackend,
    "gpt_oss": GptOssBackend,
}


# ===========================================================================
# SAE backends
# ===========================================================================


def _resolve_sae_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _is_sparsify_dir(path: Path) -> bool:
    return (path / "cfg.json").is_file() and (path / "sae.safetensors").is_file()


def _is_dictionary_learning_dir(path: Path) -> bool:
    return (path / "config.json").is_file() and (
        (path / "ae.safetensors").is_file() or (path / "ae.pt").is_file()
    )


def _resolve_hookpoint_dir(
    path: Path, hookpoint: str, is_sae_dir: Callable[[Path], bool]
) -> Path | None:
    """Locate the SAE directory for ``hookpoint`` under ``path`` (or ``path`` )."""
    hp = normalize_hookpoint(hookpoint)
    candidates = [path / hp, path / f"model.{hp}", path]
    for candidate in candidates:
        if candidate.is_dir() and is_sae_dir(candidate):
            return candidate
    return None


def _startup_reconstruction_check(
    hookpoint: str,
    encode: Callable[[Tensor], Tensor],
    decode: Callable[[Tensor], Tensor],
    *,
    hidden_size: int,
    n_features: int,
    device: torch.device,
    dtype: torch.dtype,
    n_tokens: int = 128,
) -> None:
    """Encode->decode a small standard-normal batch; log reconstruction MSE."""
    generator = torch.Generator(device="cpu").manual_seed(0)
    x_BSH = torch.randn(2, n_tokens // 2, hidden_size, generator=generator).to(
        device=device, dtype=dtype
    )
    with torch.no_grad():
        latents_BSF = encode(x_BSH)
        recon_BSH = decode(latents_BSF)
    mse = (recon_BSH.float() - x_BSH.float()).pow(2).mean().item()
    mean_l0 = (latents_BSF > 0).sum(dim=-1).float().mean().item()
    logger.info(
        f"SAE[{hookpoint}] startup reconstruction check on {device}: "
        f"mse={mse:.6e}, mean_l0={mean_l0:.1f}/{n_features}"
    )


class SparsifySAE:
    """eai-sparsify Top-K SAE at one hookpoint (local dir or W&B artifact)."""

    def __init__(
        self, hookpoint: str, sae: SparseCoder, *, revision: str | None = None
    ) -> None:
        self.hookpoint = hookpoint
        self._sae = sae
        self._revision = revision
        _startup_reconstruction_check(
            hookpoint,
            self.encode,
            self._decode,
            hidden_size=sae.d_in,
            n_features=int(sae.num_latents),
            device=sae.device,
            dtype=sae.dtype,
        )

    @classmethod
    def from_dir(
        cls,
        path: Path,
        hookpoint: str,
        *,
        device: str = "auto",
        revision: str | None = None,
    ) -> SparsifySAE:
        """Load from ``{path}/{hookpoint}/`` (or ``path`` itself), sparsify layout."""
        from sparsify import SparseCoder  # sae group

        sae_dir = _resolve_hookpoint_dir(Path(path), hookpoint, _is_sparsify_dir)
        if sae_dir is None:
            msg = (
                f"No sparsify SAE for hookpoint {hookpoint!r} under {path} "
                "(expected cfg.json + sae.safetensors)"
            )
            raise FileNotFoundError(msg)
        resolved_device = _resolve_sae_device(device)
        logger.info(
            f"Loading sparsify SAE {hookpoint!r} from {sae_dir} on {resolved_device}"
        )
        sae = SparseCoder.load_from_disk(sae_dir, device=resolved_device)
        return cls(hookpoint, sae, revision=revision)

    @property
    def n_features(self) -> int:
        return int(self._sae.num_latents)

    @property
    def revision(self) -> str | None:
        return self._revision

    def encode(self, x_BSH: Tensor) -> Tensor:
        """Scatter the sparse encoding into dense latents ``[B, S, F]``."""
        sae = self._sae
        flat_TH = x_BSH.reshape(-1, x_BSH.shape[-1]).to(
            device=sae.device, dtype=sae.dtype
        )
        encoded = sae.encode(flat_TH)
        latents_TF = torch.zeros(
            flat_TH.shape[0],
            sae.num_latents,
            dtype=flat_TH.dtype,
            device=flat_TH.device,
        )
        latents_TF = latents_TF.scatter_(
            -1, encoded.top_indices, encoded.top_acts.to(latents_TF.dtype)
        )
        return latents_TF.reshape(*x_BSH.shape[:-1], -1)

    def _decode(self, latents_BSF: Tensor) -> Tensor:
        W_dec_FH = self.decoder_weights()
        b_dec_H = self._sae.b_dec
        flat_TF = latents_BSF.reshape(-1, latents_BSF.shape[-1]).to(W_dec_FH.dtype)
        recon_TH = flat_TF @ W_dec_FH + b_dec_H
        return recon_TH.reshape(*latents_BSF.shape[:-1], -1)

    def decoder_weights(self) -> Tensor:
        W_dec_FH = self._sae.W_dec
        if W_dec_FH is None:
            msg = f"SparseCoder at {self.hookpoint!r} has no decoder weights (W_dec)"
            raise RuntimeError(msg)
        return W_dec_FH

    def to(self, device: str) -> SparsifySAE:
        self._sae.to(device)
        return self


def _converted_dictionary_dir(src: Path) -> Path:
    """Cache directory for one converted ``ae.pt``, so the same checkpoint is unpickled
    once.
    """
    ae_pt = (src / "ae.pt").resolve()
    digest = hashlib.sha256(str(ae_pt).encode()).hexdigest()[:16]
    root = Path(
        os.environ.get(
            "ROUTERINTERP_CACHE_DIR", str(Path.home() / ".cache" / "routerinterp")
        )
    )
    return root / "dictionary_learning" / digest


def _load_dictionary_from_safetensors(
    path: Path, *, device: str, trust_pickle: bool = False
) -> Dictionary:
    """Load a dictionary_learning SAE from ``config.json`` and ``ae.safetensors``.

    If the directory only contains ``ae.pt`` and ``trust_pickle`` is set, that file is
    converted once and the converted copy is loaded.
    """
    from dictionary_learning.dictionary import (  # `sae` group
        AutoEncoder,
        AutoEncoderNew,
        GatedAutoEncoder,
        JumpReluAutoEncoder,
    )
    from dictionary_learning.trainers.batch_top_k import BatchTopKSAE
    from dictionary_learning.trainers.top_k import AutoEncoderTopK
    from safetensors.torch import load_model

    ae_path = path / "ae.safetensors"
    if not ae_path.is_file() and trust_pickle and (path / "ae.pt").is_file():
        converted = _converted_dictionary_dir(path)
        if not (converted / "ae.safetensors").is_file():
            convert_pickled_sae_to_safetensors(path, converted, trust_pickle=True)
        path = converted
        ae_path = path / "ae.safetensors"
    if not ae_path.is_file():
        msg = (
            f"{path} holds only a pickled checkpoint (ae.pt). Set "
            "sae.source.trust_pickle: true for a checkpoint you trust, or run "
            "convert_pickled_sae_to_safetensors(..., trust_pickle=True) once."
        )
        raise ValueError(msg)
    trainer_config = json.loads((path / "config.json").read_text())["trainer"]
    dict_class = trainer_config["dict_class"]
    activation_dim = int(trainer_config["activation_dim"])
    dict_size = int(trainer_config["dict_size"])
    k = trainer_config.get("k")

    builders: dict[str, Callable[[], Dictionary]] = {
        "AutoEncoder": lambda: AutoEncoder(activation_dim, dict_size),
        "AutoEncoderNew": lambda: AutoEncoderNew(activation_dim, dict_size),
        "GatedAutoEncoder": lambda: GatedAutoEncoder(activation_dim, dict_size),
        "JumpReluAutoEncoder": lambda: JumpReluAutoEncoder(activation_dim, dict_size),
    }
    if k is not None:
        builders["AutoEncoderTopK"] = lambda: AutoEncoderTopK(
            activation_dim, dict_size, k=int(k)
        )
        builders["BatchTopKSAE"] = lambda: BatchTopKSAE(
            activation_dim, dict_size, k=int(k)
        )
    builder = builders.get(dict_class)
    if builder is None:
        supported = ", ".join(sorted(builders))
        msg = f"Unsupported dictionary_learning class {dict_class!r} ({supported})"
        raise ValueError(msg)
    sae = builder()
    load_model(sae, str(ae_path), device="cpu", strict=False)
    resolved_device = _resolve_sae_device(device)
    sae = sae.to(resolved_device)
    logger.info(
        f"Loaded dictionary_learning SAE ({dict_class}, k={k}) from {path} "
        f"on {resolved_device}"
    )
    return sae


class DictionaryLearningSAE:
    """dictionary_learning SAE at one hookpoint."""

    def __init__(
        self, hookpoint: str, sae: Dictionary, *, revision: str | None = None
    ) -> None:
        self.hookpoint = hookpoint
        self._sae = sae
        self._revision = revision
        param = next(sae.parameters())
        _startup_reconstruction_check(
            hookpoint,
            self.encode,
            self._decode,
            hidden_size=int(sae.activation_dim),
            n_features=int(sae.dict_size),
            device=param.device,
            dtype=param.dtype,
        )

    @classmethod
    def from_hf(
        cls, source: SAESourceConfig, hookpoint: str, *, device: str = "auto"
    ) -> DictionaryLearningSAE:
        """Download ``resid_post_layer_{N}/trainer_{id}`` from an HF repo and load."""
        if source.repo_id is None:
            msg = "repo_id is required for hf_dictionary_learning SAE source"
            raise ValueError(msg)
        if source.trainer_id is None:
            msg = "trainer_id is required for hf_dictionary_learning SAE source"
            raise ValueError(msg)
        layer = parse_layer_index(hookpoint)
        subpath = f"resid_post_layer_{layer}/trainer_{source.trainer_id}"
        folder = Path(
            snapshot_download(
                source.repo_id,
                revision=source.revision,
                allow_patterns=[f"{subpath}/*"],
            )
        )
        # The snapshot directory name is the resolved commit SHA.
        revision = source.revision or folder.name
        return cls.from_dir(
            folder / subpath,
            hookpoint,
            device=device,
            revision=revision,
            trust_pickle=source.trust_pickle,
        )

    @classmethod
    def from_dir(
        cls,
        path: Path,
        hookpoint: str,
        *,
        device: str = "auto",
        revision: str | None = None,
        trust_pickle: bool = False,
    ) -> DictionaryLearningSAE:
        """Load from ``{path}/{hookpoint}/`` (or ``path`` itself)."""
        sae_dir = _resolve_hookpoint_dir(
            Path(path), hookpoint, _is_dictionary_learning_dir
        )
        if sae_dir is None:
            msg = (
                f"No dictionary_learning SAE for hookpoint {hookpoint!r} under "
                f"{path} (expected config.json + ae.safetensors)"
            )
            raise FileNotFoundError(msg)
        sae = _load_dictionary_from_safetensors(
            sae_dir, device=device, trust_pickle=trust_pickle
        )
        return cls(hookpoint, sae, revision=revision)

    @property
    def n_features(self) -> int:
        return int(self._sae.dict_size)

    @property
    def revision(self) -> str | None:
        return self._revision

    def encode(self, x_BSH: Tensor) -> Tensor:
        param = next(self._sae.parameters())
        flat_TH = x_BSH.reshape(-1, x_BSH.shape[-1]).to(
            device=param.device, dtype=param.dtype
        )
        latents_TF = cast(Tensor, self._sae.encode(flat_TH))
        return latents_TF.reshape(*x_BSH.shape[:-1], -1)

    def _decode(self, latents_BSF: Tensor) -> Tensor:
        flat_TF = latents_BSF.reshape(-1, latents_BSF.shape[-1])
        recon_TH = cast(Tensor, self._sae.decode(flat_TF))
        return recon_TH.reshape(*latents_BSF.shape[:-1], -1)

    def decoder_weights(self) -> Tensor:
        decoder = getattr(self._sae, "decoder", None)
        if isinstance(decoder, nn.Linear):
            return decoder.weight.T  # [H, F] -> [F, H]
        W_dec_FH = getattr(self._sae, "W_dec", None)
        if isinstance(W_dec_FH, Tensor):
            return W_dec_FH
        msg = f"Cannot extract decoder weights from {type(self._sae).__name__}"
        raise RuntimeError(msg)

    def to(self, device: str) -> DictionaryLearningSAE:
        self._sae.to(device)
        return self


def _tensor_to_numpy(value: Tensor) -> np.ndarray:
    """Copy a tensor to a float32 NumPy array on CPU."""
    tensor = value.detach().cpu()
    if tensor.dtype in (torch.float16, torch.bfloat16):
        tensor = tensor.float()
    return tensor.numpy()


def _hf_revision_from_path(path: Path) -> str | None:
    """Return the HF commit SHA when ``path`` lives in a hub snapshot cache."""
    parts = path.resolve().parts
    if "snapshots" in parts:
        idx = parts.index("snapshots")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def convert_pickled_sae_to_safetensors(
    src: str | Path, dst: str | Path, *, trust_pickle: bool
) -> Path:
    """Convert a pickled SAE checkpoint to SafeTensors.

    Refuses to run unless ``trust_pickle`` is set, because unpickling can execute
    arbitrary code.
    """
    if not trust_pickle:
        msg = (
            "Refusing to load a pickled SAE checkpoint without explicit trust. "
            "Re-run with trust_pickle=True only for checkpoints you trust."
        )
        raise ValueError(msg)
    src = Path(src)
    dst = Path(dst)
    ae_pt = src / "ae.pt" if src.is_dir() else src
    if not ae_pt.is_file():
        msg = f"No ae.pt checkpoint found at {ae_pt}"
        raise FileNotFoundError(msg)
    config_json = ae_pt.parent / "config.json"

    sha256 = hashlib.sha256(ae_pt.read_bytes()).hexdigest()
    logger.warning(f"Trusted pickle load of {ae_pt} (sha256={sha256})")
    try:
        state_dict: dict[str, Any] = torch.load(
            ae_pt, map_location="cpu", weights_only=True
        )
    except Exception:
        logger.warning(
            "weights_only load failed; retrying full pickle load (trusted source)"
        )
        state_dict = torch.load(ae_pt, map_location="cpu", weights_only=False)

    tensors = {
        key: _tensor_to_numpy(value)
        for key, value in state_dict.items()
        if isinstance(value, Tensor)
    }
    write_safetensors(dst / "ae.safetensors", tensors)
    trainer_config = (
        json.loads(config_json.read_text()) if config_json.is_file() else {}
    )
    write_json(dst / "config.json", trainer_config)
    write_json(
        dst / "provenance.json",
        {
            "source_path": str(ae_pt.resolve()),
            "source_sha256": sha256,
            "hf_revision": _hf_revision_from_path(ae_pt),
            "converted_at": datetime.now(UTC).isoformat(),
        },
    )
    logger.info(f"Converted {ae_pt} -> {dst} ({len(tensors)} tensors)")
    return dst


def _wandb_cache_dir(entity: str, project: str, artifact: str) -> Path:
    root = Path(
        os.environ.get(
            "ROUTERINTERP_CACHE_DIR", str(Path.home() / ".cache" / "routerinterp")
        )
    )
    return root / "wandb_artifacts" / f"{entity}__{project}__{artifact}"


def _download_wandb_artifact(
    source: SAESourceConfig, config: SAEConfig
) -> tuple[Path, str]:
    """Download a sparsify SAE artifact from W&B."""
    import wandb

    if source.wandb_project is None:
        msg = "wandb_project is required for sparsify_wandb SAE source"
        raise ValueError(msg)
    artifact_name = source.wandb_artifact
    if artifact_name is None:
        if config.k is None or config.expansion_factor is None:
            msg = (
                "Set source.wandb_artifact, or sae.k + sae.expansion_factor so "
                "the artifact name can be derived"
            )
            raise ValueError(msg)
        artifact_name = "-".join(
            [*config.hookpoints, str(config.k), str(config.expansion_factor)]
        )
    api = wandb.Api()
    entity = source.wandb_entity or api.default_entity
    if entity is None:
        msg = "Could not resolve W&B entity; set source.wandb_entity"
        raise ValueError(msg)
    ref = f"{entity}/{source.wandb_project}/{artifact_name}:latest"
    logger.info(f"Downloading SAE artifact {ref}")
    artifact = api.artifact(ref, type="sparse_autoencoder")
    root = _wandb_cache_dir(entity, source.wandb_project, artifact_name)
    artifact.download(root=root)
    revision = getattr(artifact, "commit_hash", None) or ref
    return root, str(revision)
