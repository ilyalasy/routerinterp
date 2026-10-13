"""Hooks, residual replacement, and SAE encode/decode for the toy backends."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from routerinterp.backends import (
    DictionaryLearningSAE,
    GptOssBackend,
    OlmoeBackend,
    SparsifySAE,
    install_residual_replacement,
)
from routerinterp.core import ModelConfig
from routerinterp.select import (
    attribute_batch_margins,
    margin_diagonal,
    membership_margins,
)

BATCH, SEQ, HIDDEN, N_EXPERTS, TOP_K = 2, 3, 4, 4, 2
# Basis tokens 0,1,2 repeat. Layer-0 gate sends each basis vector to two experts;
# expert 3 is never selected. Layer-1 gate ignores the residual and always
# selects experts 2 and 3.
LAYER0_TOKENS = {
    0: [0, 2, 3, 5],
    1: [0, 1, 3, 4],
    2: [1, 2, 4, 5],
}
WRITE = torch.tensor([1.0, -2.0, 0.5, 3.0])


def _hidden() -> Tensor:
    flat = torch.zeros(BATCH * SEQ, HIDDEN)
    for token in range(BATCH * SEQ):
        flat[token, token % 3] = 1
    return flat.view(BATCH, SEQ, HIDDEN)


def _gates() -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    weight0 = torch.zeros(N_EXPERTS, HIDDEN)
    for expert in range(3):
        weight0[expert, expert] = 2
        weight0[(expert + 1) % 3, expert] = 1
    weight1 = torch.zeros(N_EXPERTS, HIDDEN)
    return (weight0, torch.zeros(N_EXPERTS)), (
        weight1,
        torch.tensor([0.0, 0.0, 1.0, 5.0]),
    )


def _directions() -> Tensor:
    return torch.stack([WRITE * (expert + 1) for expert in range(N_EXPERTS)])


class _Inner(nn.Module):
    def __init__(self, layers: list[nn.Module]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)


class _ToyModel(nn.Module):
    def __init__(self, layers: list[nn.Module], hidden_BSH: Tensor) -> None:
        super().__init__()
        self.model = _Inner(layers)
        self.register_buffer("hidden_BSH", hidden_BSH)

    def forward(
        self, input_ids: Tensor, output_router_logits: bool = False, **_: object
    ) -> SimpleNamespace:
        hidden = self.hidden_BSH
        router_logits: list[Tensor] = []
        for layer in self.model.layers:
            hidden, logits = layer(hidden)
            router_logits.append(logits)
        return SimpleNamespace(
            router_logits=tuple(router_logits) if output_router_logits else None
        )


class _Layer(nn.Module):
    def __init__(self, mlp: nn.Module) -> None:
        super().__init__()
        self.mlp = mlp

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        delta, extra = self.mlp(hidden)
        return hidden + delta, extra


class _Expert(nn.Module):
    def __init__(self, direction: Tensor) -> None:
        super().__init__()
        self.direction = nn.Parameter(direction.clone())

    def forward(self, hidden_RH: Tensor) -> Tensor:
        return self.direction.expand(hidden_RH.shape[0], -1)


class _OlmoeMlp(nn.Module):
    def __init__(self, weight: Tensor, bias: Tensor, directions: Tensor) -> None:
        super().__init__()
        self.gate = nn.Linear(HIDDEN, N_EXPERTS, bias=True)
        with torch.no_grad():
            self.gate.weight.copy_(weight)
            self.gate.bias.copy_(bias)
        self.experts = nn.ModuleList(_Expert(directions[e]) for e in range(N_EXPERTS))
        self.top_k = TOP_K
        self.num_experts = N_EXPERTS

    def forward(self, hidden_BSH: Tensor) -> tuple[Tensor, Tensor]:
        batch, seq, hidden = hidden_BSH.shape
        flat_TH = hidden_BSH.reshape(-1, hidden)
        logits_TE = self.gate(flat_TH)
        weights_TE = F.softmax(logits_TE, dim=-1, dtype=torch.float)
        top_TK, selected_TK = torch.topk(weights_TE, self.top_k, dim=-1)
        mask_EKT = F.one_hot(selected_TK, num_classes=self.num_experts).permute(2, 1, 0)
        mixed_TH = torch.zeros_like(flat_TH)
        for expert_id, expert in enumerate(self.experts):
            slot_R, token_R = torch.where(mask_EKT[expert_id])
            # Scale happens outside the expert, so a write hook must see the raw direction.
            raw_RH = expert(flat_TH[token_R])
            if token_R.numel():
                mixed_TH.index_add_(0, token_R, raw_RH * top_TK[token_R, slot_R, None])
        return mixed_TH.view(batch, seq, hidden), logits_TE


class _Router(nn.Module):
    def __init__(self, weight: Tensor, bias: Tensor) -> None:
        super().__init__()
        self.proj = nn.Linear(HIDDEN, N_EXPERTS, bias=True)
        with torch.no_grad():
            self.proj.weight.copy_(weight)
            self.proj.bias.copy_(bias)

    @property
    def weight(self) -> Tensor:
        return self.proj.weight

    @property
    def bias(self) -> Tensor:
        return self.proj.bias

    def forward(self, hidden_BSH: Tensor) -> tuple[Tensor, Tensor]:
        flat_TH = hidden_BSH.reshape(-1, hidden_BSH.shape[-1])
        logits_TE = F.linear(flat_TH, self.weight, self.bias)
        values_TK, indices_TK = torch.topk(logits_TE, TOP_K, dim=-1)
        scores_TE = torch.zeros_like(logits_TE).scatter_(
            1, indices_TK, torch.softmax(values_TK, dim=-1)
        )
        return scores_TE, indices_TK


class _FusedExperts(nn.Module):
    """Zero SwiGLU weights, so the unscaled write is exactly ``down_proj_bias``."""

    def __init__(self, biases_EH: Tensor) -> None:
        super().__init__()
        n_experts, hidden = biases_EH.shape
        self.hidden_size = hidden
        self.num_experts = n_experts
        self.gate_up_proj = nn.Parameter(torch.zeros(n_experts, hidden, 2))
        self.gate_up_proj_bias = nn.Parameter(torch.zeros(n_experts, 2))
        self.down_proj = nn.Parameter(torch.zeros(n_experts, 1, hidden))
        self.down_proj_bias = nn.Parameter(biases_EH.clone())
        self.alpha = 1.0
        self.limit = 7.0

    def forward(
        self,
        hidden_states: Tensor,
        router_indices: Tensor | None = None,
        routing_weights: Tensor | None = None,
    ) -> Tensor:
        return torch.ones_like(hidden_states)


class _GptMlp(nn.Module):
    def __init__(self, weight: Tensor, bias: Tensor, directions: Tensor) -> None:
        super().__init__()
        self.router = _Router(weight, bias)
        self.experts = _FusedExperts(directions)

    def forward(self, hidden_BSH: Tensor) -> tuple[Tensor, Tensor]:
        scores_TE, indices_TK = self.router(hidden_BSH)
        out_BSH = self.experts(
            hidden_BSH, router_indices=indices_TK, routing_weights=scores_TE
        )
        return out_BSH, scores_TE


def _model(style: str) -> tuple[nn.Module, OlmoeBackend | GptOssBackend]:
    (w0, b0), (w1, b1) = _gates()
    directions = _directions()
    if style == "olmoe":
        layers = [
            _Layer(_OlmoeMlp(w0, b0, directions)),
            _Layer(_OlmoeMlp(w1, b1, directions)),
        ]
        backend: OlmoeBackend | GptOssBackend = OlmoeBackend(ModelConfig(name="toy"))
    else:
        layers = [
            _Layer(_GptMlp(w0, b0, directions)),
            _Layer(_GptMlp(w1, b1, directions)),
        ]
        backend = GptOssBackend(ModelConfig(name="toy"))
    return _ToyModel(layers, _hidden()), backend


def _stream(model: nn.Module) -> tuple[list[Tensor], list[Tensor]]:
    hidden = model.hidden_BSH
    entering: list[Tensor] = []
    leaving: list[Tensor] = []
    for layer in model.model.layers:
        entering.append(hidden.detach().clone())
        delta, _extra = layer.mlp(hidden)
        hidden = hidden + delta
        leaving.append(hidden.detach().clone())
    return entering, leaving


def _logits(layer: nn.Module, hidden_BSH: Tensor, style: str) -> Tensor:
    flat_TH = hidden_BSH.reshape(-1, HIDDEN)
    if style == "olmoe":
        return F.linear(flat_TH, layer.mlp.gate.weight, layer.mlp.gate.bias)
    return F.linear(flat_TH, layer.mlp.router.weight, layer.mlp.router.bias)


def _write_vector(model: nn.Module, style: str, expert: int) -> Tensor:
    mlp = model.model.layers[0].mlp
    if style == "olmoe":
        return mlp.experts[expert].direction.detach()
    return mlp.experts.down_proj_bias[expert].detach()


@pytest.mark.parametrize("style", ["olmoe", "gpt_oss"])
def test_captures_agree_on_hookpoints_and_expert_ids(style: str) -> None:
    model, backend = _model(style)
    entering, leaving = _stream(model)
    expected_logits = [
        _logits(layer, hidden, style)
        for layer, hidden in zip(model.model.layers, entering, strict=True)
    ]
    activations: dict[str, Tensor] = {}
    router_logits: dict[int, Tensor] = {}
    writes: list[tuple[int, int, Tensor, Tensor]] = []

    def on_activation(hookpoint: str, hidden_BSH: Tensor) -> None:
        activations[hookpoint] = hidden_BSH.detach().clone()

    def on_router(layer_idx: int, logits_TE: Tensor) -> None:
        router_logits[layer_idx] = logits_TE.detach().clone()

    def on_write(layer: int, expert: int, token_idx: Tensor, vec_RH: Tensor) -> None:
        writes.append(
            (layer, expert, token_idx.detach().clone(), vec_RH.detach().clone())
        )

    closer = backend.install_captures(
        model,  # type: ignore[arg-type]
        activation_hookpoints=("model.layers.0.mlp", "layers.1"),
        router_layers=(0, 1),
        expert_write_layers=(0,),
        on_activation=on_activation,
        on_router_logits=on_router,
        on_expert_write=on_write,
    )
    try:
        model(torch.zeros(BATCH, SEQ, dtype=torch.long))
    finally:
        closer()

    # ``layers.N.mlp`` is the MLP input; ``layers.N`` is the block output.
    assert set(activations) == {"layers.0.mlp", "layers.1"}
    assert not activations["layers.1"].requires_grad
    torch.testing.assert_close(activations["layers.0.mlp"], entering[0])
    torch.testing.assert_close(activations["layers.1"], leaving[1])
    assert not torch.allclose(activations["layers.0.mlp"], activations["layers.1"])

    # Both backends report pre-softmax logits (GPT-OSS's module output is not).
    assert set(router_logits) == {0, 1}
    assert router_logits[0].shape == (BATCH * SEQ, N_EXPERTS)
    assert router_logits[0].max().item() > 1
    for layer in (0, 1):
        torch.testing.assert_close(router_logits[layer], expected_logits[layer])
    assert not torch.allclose(router_logits[0], router_logits[1])
    layer1_experts = {
        frozenset(row.tolist()) for row in router_logits[1].topk(TOP_K, dim=-1).indices
    }
    assert layer1_experts == {frozenset({2, 3})}

    by_expert: dict[int, tuple[list[int], Tensor]] = {}
    for layer, expert, token_idx, vec_RH in writes:
        assert layer == 0
        by_expert[expert] = (token_idx.tolist(), vec_RH)
    assert set(by_expert) == set(LAYER0_TOKENS)
    for expert, tokens in LAYER0_TOKENS.items():
        seen, vec_RH = by_expert[expert]
        assert sorted(seen) == tokens
        want = _write_vector(model, style, expert)
        torch.testing.assert_close(vec_RH, want.expand_as(vec_RH))
        # Routing weights are not 1 under top-2, so a scaled write would differ.
        assert not torch.allclose(vec_RH, want * 0.5)

    activations.clear()
    router_logits.clear()
    writes.clear()
    model(torch.zeros(BATCH, SEQ, dtype=torch.long))
    assert activations == {}
    assert router_logits == {}
    assert writes == []


@pytest.mark.parametrize("style", ["olmoe", "gpt_oss"])
def test_capture_rejects_an_unknown_layer(style: str) -> None:
    model, backend = _model(style)
    with pytest.raises(ValueError, match="out of bounds"):
        backend.install_captures(
            model,  # type: ignore[arg-type]
            activation_hookpoints=(),
            router_layers=(),
            expert_write_layers=(9,),
            on_activation=lambda *_: None,
            on_router_logits=lambda *_: None,
            on_expert_write=lambda *_: None,
        )


# Centered tokens. t0 activates two features; t4 activates four (top-k keeps two);
# t3 is exactly the decoder bias and must encode to zeros.
_W_ENC = torch.tensor(
    [
        [2.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 4.0],
        [1.0, 1.0, 0.0],
    ]
)
_W_DEC = torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
)
_B_DEC = torch.tensor([0.5, -0.25, 0.25])


def _inputs() -> Tensor:
    centered = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0]],
        ]
    )
    return centered + _B_DEC


def _sparsify() -> SparsifySAE:
    pytest.importorskip("sparsify")
    from sparsify import SparseCoder
    from sparsify.config import SparseCoderConfig

    coder = SparseCoder(
        3,
        SparseCoderConfig(k=2, num_latents=4, normalize_decoder=False),
        device="cpu",
        dtype=torch.float32,
    )
    with torch.no_grad():
        coder.encoder.weight.copy_(_W_ENC)
        coder.encoder.bias.zero_()
        coder.W_dec.copy_(_W_DEC)
        coder.b_dec.copy_(_B_DEC)
    return SparsifySAE("layers.0.mlp", coder)


def _dictionary(kind: str) -> DictionaryLearningSAE:
    pytest.importorskip("dictionary_learning")
    from dictionary_learning.dictionary import AutoEncoder
    from dictionary_learning.trainers.top_k import AutoEncoderTopK

    if kind == "dictionary_topk":
        sae = AutoEncoderTopK(3, 4, k=2)
        bias = sae.b_dec
    else:
        sae = AutoEncoder(3, 4)
        bias = sae.bias
    with torch.no_grad():
        sae.encoder.weight.copy_(_W_ENC)
        sae.encoder.bias.zero_()
        sae.decoder.weight.copy_(_W_DEC.T)
        bias.copy_(_B_DEC)
    return DictionaryLearningSAE("layers.0.mlp", sae)


@pytest.mark.parametrize("kind", ["sparsify", "dictionary_relu", "dictionary_topk"])
def test_fake_sae_encode_decode(kind: str) -> None:
    backend = _sparsify() if kind == "sparsify" else _dictionary(kind)
    assert backend.hookpoint == "layers.0.mlp"
    assert backend.n_features == 4
    torch.testing.assert_close(backend.decoder_weights(), _W_DEC)

    latents = backend.encode(_inputs())
    assert latents.shape == (2, 3, 4)
    assert torch.all(latents >= 0)
    # t0 = e0: features 0 and 3. t3 = bias: nothing fires.
    torch.testing.assert_close(latents[0, 0], torch.tensor([2.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(latents[1, 0], torch.zeros(4))
    if kind == "dictionary_relu":
        torch.testing.assert_close(latents[1, 1], torch.tensor([2.0, 3.0, 4.0, 2.0]))
        assert int((latents[1, 1] > 0).sum()) == 4
    else:
        torch.testing.assert_close(latents[1, 1], torch.tensor([0.0, 3.0, 4.0, 0.0]))
        assert int((latents[1, 1] > 0).sum()) == 2

    recon = backend._decode(latents)
    torch.testing.assert_close(recon[1, 0], _B_DEC)
    torch.testing.assert_close(recon[0, 0], torch.tensor([2.0, 1.0, 1.0]) + _B_DEC)
    if kind == "dictionary_relu":
        torch.testing.assert_close(recon[1, 1], torch.tensor([2.0, 5.0, 6.0]) + _B_DEC)
    else:
        torch.testing.assert_close(recon[1, 1], torch.tensor([0.0, 3.0, 4.0]) + _B_DEC)


def test_decoder_weights_reject_a_sae_without_a_decoder() -> None:
    pytest.importorskip("sparsify")
    backend = _sparsify()
    backend._sae.W_dec = None
    with pytest.raises(RuntimeError, match="decoder"):
        backend.decoder_weights()

    pytest.importorskip("dictionary_learning")

    class _NoDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(2))
            self.activation_dim = 2
            self.dict_size = 2

        def encode(self, hidden_TH: Tensor) -> Tensor:
            return torch.zeros(hidden_TH.shape[0], 2)

        def decode(self, latents_TF: Tensor) -> Tensor:
            return torch.zeros(latents_TF.shape[0], 2)

    wrapped = DictionaryLearningSAE("layers.0", _NoDecoder())
    with pytest.raises(RuntimeError, match="decoder"):
        wrapped.decoder_weights()


class _AbortForward(Exception):
    pass


def _separated_model(style: str) -> tuple[nn.Module, OlmoeBackend | GptOssBackend]:
    """Build a toy model whose layer-0 logits sit off the top-k boundaries.

    Expert 3's gate row is zero, so its logit stays at the bias and the others stay
    separated.
    """
    model, backend = _model(style)
    model.config = SimpleNamespace(  # type: ignore[assignment]
        num_experts=N_EXPERTS,
        num_experts_per_tok=TOP_K,
        model_type=style,
    )
    row = torch.tensor([1.0, 0.1, 0.2, 0.4])
    with torch.no_grad():
        model.hidden_BSH.copy_(row.view(1, 1, HIDDEN).expand_as(model.hidden_BSH))
    return model, backend


def _margins_at(
    model: nn.Module,
    backend: OlmoeBackend | GptOssBackend,
    hidden_BSH: Tensor,
    input_ids: Tensor,
    position: int,
) -> Tensor:
    store: dict[str, Tensor] = {}

    def on_logits(logits_TE: Tensor) -> None:
        store["logits"] = logits_TE
        raise _AbortForward

    handles = [
        install_residual_replacement(
            model,  # type: ignore[arg-type]
            "layers.0.mlp",
            hidden_BSH,
        ),
        backend.install_router_logit_hook(
            model,  # type: ignore[arg-type]
            0,
            on_logits,
        ),
    ]
    try:
        try:
            model(input_ids)
        except _AbortForward:
            pass
    finally:
        for handle in handles:
            handle.remove()
    batch, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
    pos = position % seq
    logits_BSE = store["logits"].view(batch, seq, -1)
    return membership_margins(logits_BSE, TOP_K)[:, pos, :].detach()


def _record_mlp_inputs(
    model: nn.Module, hookpoint: str, replacement_BSH: Tensor
) -> list[Tensor]:
    seen: list[Tensor] = []

    def _pre(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        seen.append(args[0].detach().clone())

    handles = [
        install_residual_replacement(
            model,  # type: ignore[arg-type]
            hookpoint,
            replacement_BSH,
        )
    ]
    for layer in model.model.layers:  # type: ignore[attr-defined]
        handles.append(layer.mlp.register_forward_pre_hook(_pre))
    try:
        model(torch.zeros(BATCH, SEQ, dtype=torch.long))
    finally:
        for handle in handles:
            handle.remove()
    return seen


@pytest.mark.parametrize("style", ["olmoe", "gpt_oss"])
def test_margin_diagonal_matches_centered_finite_difference(style: str) -> None:
    model, backend = _separated_model(style)
    input_ids = torch.zeros(BATCH, SEQ, dtype=torch.long)
    hidden, margins, _logits, grad = margin_diagonal(
        model,  # type: ignore[arg-type]
        backend,
        hookpoint="layers.0.mlp",
        router_layer_idx=0,
        input_ids_BS=input_ids,
        top_k=TOP_K,
        position=-1,
        expert_chunk=2,  # two vmap chunks over the four experts
    )
    hidden = hidden.detach()
    grad = grad.detach()
    pos = SEQ - 1
    base = _margins_at(model, backend, hidden, input_ids, pos)
    torch.testing.assert_close(base, margins.detach())

    eps = 1e-3
    finite = torch.empty_like(grad)
    for batch in range(BATCH):
        for dim in range(HIDDEN):
            plus = hidden.clone()
            minus = hidden.clone()
            plus[batch, pos, dim] += eps
            minus[batch, pos, dim] -= eps
            higher = _margins_at(model, backend, plus, input_ids, pos)
            lower = _margins_at(model, backend, minus, input_ids, pos)
            finite[:, batch, dim] = (higher - lower)[batch] / (2 * eps)
    torch.testing.assert_close(finite, grad, atol=1e-3, rtol=1e-3)

    # Margins at the last token do not move when an earlier token is nudged.
    off_token = torch.zeros_like(grad)
    for dim in range(HIDDEN):
        plus = hidden.clone()
        minus = hidden.clone()
        plus[0, 0, dim] += eps
        minus[0, 0, dim] -= eps
        higher = _margins_at(model, backend, plus, input_ids, pos)
        lower = _margins_at(model, backend, minus, input_ids, pos)
        off_token[:, 0, dim] = (higher - lower)[0] / (2 * eps)
    torch.testing.assert_close(off_token, torch.zeros_like(off_token))


class _OneFeatureSAE:
    """Constant latent and a fixed decoder row, so c_f = z <d, grad>."""

    n_features = 1
    revision = None

    def __init__(self, latent: float, direction_H: Tensor) -> None:
        self._latent = latent
        self._direction_H = direction_H

    def encode(self, hidden_BSH: Tensor) -> Tensor:
        return hidden_BSH.new_full((*hidden_BSH.shape[:-1], 1), self._latent)

    def decoder_weights(self) -> Tensor:
        return self._direction_H.view(1, -1)

    def to(self, _device: str) -> _OneFeatureSAE:
        return self


@pytest.mark.parametrize("style", ["olmoe", "gpt_oss"])
def test_feature_contribution_is_latent_times_decoder_gradient(style: str) -> None:
    model, backend = _separated_model(style)
    input_ids = torch.zeros(BATCH, SEQ, dtype=torch.long)
    _hidden, _margins, _logits, grad = margin_diagonal(
        model,  # type: ignore[arg-type]
        backend,
        hookpoint="layers.0.mlp",
        router_layer_idx=0,
        input_ids_BS=input_ids,
        top_k=TOP_K,
        position=-1,
        expert_chunk=2,
    )
    grad = grad.detach()
    latent = 2.5
    direction = torch.tensor([0.5, -1.0, 0.25, 3.0])
    attr = attribute_batch_margins(
        model,  # type: ignore[arg-type]
        backend,
        _OneFeatureSAE(latent, direction),  # type: ignore[arg-type]
        hookpoint="layers.0.mlp",
        router_layer_idx=0,
        input_ids_BS=input_ids,
        top_k=TOP_K,
        position=-1,
        expert_chunk=2,
    )
    for batch in range(BATCH):
        assert attr.feature_ids[batch].tolist() == [0]
        torch.testing.assert_close(attr.feature_vals[batch], torch.tensor([latent]))
    for expert in range(N_EXPERTS):
        for batch in range(BATCH):
            want = latent * torch.dot(direction, grad[expert, batch])
            torch.testing.assert_close(
                attr.contributions_E[expert][batch], want.reshape(1)
            )


@pytest.mark.parametrize("style", ["olmoe", "gpt_oss"])
def test_residual_replacement_uses_pre_hook_for_mlp_and_post_hook_for_block(
    style: str,
) -> None:
    model, _backend = _model(style)
    replacement = torch.full_like(model.hidden_BSH, 7.0)

    mlp_inputs = _record_mlp_inputs(model, "layers.0.mlp", replacement)
    assert len(mlp_inputs) == 2
    torch.testing.assert_close(mlp_inputs[0], replacement)

    block_inputs = _record_mlp_inputs(model, "layers.0", replacement)
    assert len(block_inputs) == 2
    torch.testing.assert_close(block_inputs[0], model.hidden_BSH)
    torch.testing.assert_close(block_inputs[1], replacement)
