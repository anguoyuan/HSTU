# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

"""
HSTUWithRankMixer: HSTU sequential encoder followed by RankMixer.

Data flow:
  item sequence
    -> Embedding lookup                      [B, N, D_item]
    -> InputFeaturesPreprocessor             [B, N, D_emb]
    -> HSTU (hierarchical transduction)      [B, N, D_emb]
    -> RankMixer (token mixing + sparse MoE) [B, N, D_emb]
    -> OutputPostprocessor (L2 / LayerNorm)  [B, N, D_emb]

RankMixer is adapted from custome/rankmixer.py with NPU-specific ops replaced
by standard PyTorch equivalents so it runs on any CUDA device.

The RankMixer output dimension is kept equal to the HSTU embedding dimension,
so the existing dot-product similarity and sampled-softmax loss require no
changes.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from generative_recommenders.research.modeling.sequential.embedding_modules import (
    EmbeddingModule,
)
from generative_recommenders.research.modeling.sequential.hstu import (
    HSTU,
    HSTUCacheState,
)
from generative_recommenders.research.modeling.sequential.input_features_preprocessors import (
    InputFeaturesPreprocessorModule,
)
from generative_recommenders.research.modeling.sequential.output_postprocessors import (
    OutputPostprocessorModule,
)
from generative_recommenders.research.rails.similarities.module import SimilarityModule

TIMESTAMPS_KEY = "timestamps"


# ---------------------------------------------------------------------------
# RankMixer building blocks
# (Adapted from custome/rankmixer.py — pure PyTorch, no NPU dependencies)
# ---------------------------------------------------------------------------


class RankMixingInput(nn.Module):
    """
    Splits the flat embedding (BN, D) into T tokens of size dim_per_token and
    projects each token to inner_dim.

        T         = D // dim_per_token
        inner_dim = int(T * t_multiplier)

    Output shape: (BN, T, inner_dim)
    """

    def __init__(
        self,
        embedding_dim: int,
        dim_per_token: int,
        t_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        if embedding_dim % dim_per_token != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by "
                f"dim_per_token ({dim_per_token})"
            )
        self.num_tokens: int = embedding_dim // dim_per_token
        self.dim_per_token: int = dim_per_token
        self.inner_dim: int = int(self.num_tokens * t_multiplier)
        self.proj = nn.Linear(dim_per_token, self.inner_dim, bias=False)
        nn.init.xavier_normal_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (BN, D)
        BN = x.size(0)
        tokens = x.view(BN, self.num_tokens, self.dim_per_token)  # (BN, T, d_tok)
        return self.proj(tokens)  # (BN, T, inner_dim)


class TokenMixing(nn.Module):
    """
    MLP-Mixer-style token mixing across the T dimension.

    For x of shape (BN, T, D):
        tm_x  = reshape(x.T, (BN, T, D))   # swaps T <-> D axes
        output = LayerNorm(x + tm_x)
    """

    def __init__(self, num_tokens: int, inner_dim: int) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.ln = nn.LayerNorm(inner_dim, eps=1e-7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (BN, T, D)
        BN, T, D = x.shape
        # Transpose T <-> D then reshape back to (BN, T, D) to mix tokens
        tm_x = x.transpose(1, 2).contiguous().view(BN, self.num_tokens, D)
        return self.ln(x + tm_x)


class PerTokenFFN(nn.Module):
    """
    Position-wise MLP with separate weight matrices per token position.

    Layer sizes (num_layers >= 2):
        layer 0:       D  -> kD   (GELU)
        layers 1..L-2: kD -> kD   (GELU)
        layer L-1:     kD -> D    (no activation)
    """

    def __init__(
        self,
        inner_dim: int,
        num_tokens: int,
        k: float,
        num_layers: int,
        bias: bool = True,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        assert num_layers >= 2, "num_layers must be >= 2"
        self.D = inner_dim
        self.T = num_tokens
        kD = int(round(k * inner_dim))
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0.0 else nn.Identity()

        in_dims = [inner_dim] + [kD] * (num_layers - 1)
        out_dims = [kD] * (num_layers - 1) + [inner_dim]
        self.W = nn.ParameterList([
            nn.Parameter(torch.empty(num_tokens, din, dout))
            for din, dout in zip(in_dims, out_dims)
        ])
        self.b: Optional[nn.ParameterList] = None
        if bias:
            self.b = nn.ParameterList([
                nn.Parameter(torch.empty(num_tokens, dout))
                for dout in out_dims
            ])
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for i in range(self.num_layers):
            nn.init.xavier_normal_(self.W[i])
            if self.b is not None:
                nn.init.zeros_(self.b[i])

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # s: (BN, T, D)
        x = s
        for i in range(self.num_layers):
            x = torch.einsum("bti,tid->btd", x, self.W[i])
            if self.b is not None:
                x = x + self.b[i]
            if i < self.num_layers - 1:
                x = F.gelu(x)
                x = self.dropout(x)
        return self.dropout(x)


class SparseMoE(nn.Module):
    """
    ReLU-routed Sparse Mixture-of-Experts (no top-k, no softmax normalisation).

        gates   = ReLU(router(s))               # (BN, T, Ne)
        output  = sum_j gates_j * expert_j(s)  # (BN, T, D)
        result  = LayerNorm(s + output)

    The l1 regularisation loss on gate activations encourages sparsity.
    """

    def __init__(
        self,
        num_experts: int,
        inner_dim: int,
        num_tokens: int,
        k: float,
        num_layers_per_expert: int,
        bias: bool = True,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.Ne = num_experts
        self.router = nn.Linear(inner_dim, num_experts, bias=False)
        nn.init.xavier_normal_(self.router.weight)
        self.experts = nn.ModuleList([
            PerTokenFFN(
                inner_dim=inner_dim,
                num_tokens=num_tokens,
                k=k,
                num_layers=num_layers_per_expert,
                bias=bias,
                dropout_p=dropout_p,
            )
            for _ in range(num_experts)
        ])
        self.ln = nn.LayerNorm(inner_dim, eps=1e-7)

    def forward(
        self, s: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # s: (BN, T, D)
        gates = F.relu(self.router(s))                     # (BN, T, Ne)
        reg_loss = gates.sum(dim=(-2, -1))                 # (BN,) — per-sample L1
        sparsity = (gates > 0).float().mean()              # scalar

        # Stack expert outputs then weighted-sum
        expert_out = torch.stack(
            [exp(s) for exp in self.experts], dim=2
        )                                                  # (BN, T, Ne, D)
        v = torch.einsum("btjd,btj->btd", expert_out, gates)  # (BN, T, D)

        return self.ln(s + v), (reg_loss, sparsity)


class RankMixerBlock(nn.Module):
    """One RankMixer block: TokenMixing -> SparseMoE."""

    def __init__(
        self,
        num_tokens: int,
        inner_dim: int,
        num_experts: int,
        k: float,
        ffn_layers: int,
        dropout_p: float,
    ) -> None:
        super().__init__()
        self.token_mixing = TokenMixing(num_tokens=num_tokens, inner_dim=inner_dim)
        self.moe = SparseMoE(
            num_experts=num_experts,
            inner_dim=inner_dim,
            num_tokens=num_tokens,
            k=k,
            num_layers_per_expert=ffn_layers,
            bias=True,
            dropout_p=dropout_p,
        )

    def forward(
        self,
        x: torch.Tensor,
        acc_loss: torch.Tensor,
        acc_sparsity: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.token_mixing(x)
        x, (loss, sparsity) = self.moe(x)
        return x, (acc_loss + loss, acc_sparsity + sparsity)


class RankMixer(nn.Module):
    """
    Full RankMixer module.

    Each item embedding [D] is tokenised into T tokens, processed by
    n_layers of (TokenMixing + SparseMoE), mean-pooled over tokens, and
    projected back to output_dim.

    Input  shape: (B, N, embedding_dim)
    Output shape: (B, N, output_dim)

    Also returns:
        l1_loss  — sparsity regularisation loss (scalar)
        sparsity — mean fraction of active MoE gates (scalar, for logging)
    """

    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        dim_per_token: int = 16,
        num_experts: int = 4,
        k: float = 4.0,
        t_multiplier: float = 1.0,
        ffn_layers: int = 2,
        n_layers: int = 2,
        dropout_p: float = 0.05,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.input_proj = RankMixingInput(
            embedding_dim=embedding_dim,
            dim_per_token=dim_per_token,
            t_multiplier=t_multiplier,
        )
        T = self.input_proj.num_tokens
        inner_dim = self.input_proj.inner_dim

        self.blocks = nn.ModuleList([
            RankMixerBlock(
                num_tokens=T,
                inner_dim=inner_dim,
                num_experts=num_experts,
                k=k,
                ffn_layers=ffn_layers,
                dropout_p=dropout_p,
            )
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(inner_dim, output_dim, bias=False)
        nn.init.xavier_normal_(self.output_proj.weight)
        # Use LayerNorm as a portable alternative to RMSNorm (works on all devices)
        self.output_norm = nn.LayerNorm(output_dim, eps=1e-6)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, embedding_dim)
        Returns:
            y        : (B, N, output_dim)
            l1_loss  : scalar — sparsity regularisation loss (sum over blocks)
            sparsity : scalar — mean active-gate fraction (for logging)
        """
        B, N, D = x.shape
        x_flat = x.view(B * N, D)                         # (BN, D)
        rm = self.input_proj(x_flat)                       # (BN, T, inner_dim)

        # Initialise accumulators on the correct device / dtype
        acc_loss = x.new_zeros(B * N)                      # (BN,)
        acc_sparsity = x.new_zeros(())                     # scalar

        for block in self.blocks:
            rm, (acc_loss, acc_sparsity) = block(rm, acc_loss, acc_sparsity)

        rm_out = rm.mean(dim=1)                            # (BN, inner_dim)

        l1_loss = (acc_loss / self.n_layers).mean()        # scalar
        sparsity = acc_sparsity / self.n_layers            # scalar

        out = self.output_proj(rm_out)                     # (BN, output_dim)
        out = self.output_norm(out)                        # (BN, output_dim)
        y = out.view(B, N, self.output_dim)                # (B, N, output_dim)
        return y, l1_loss, sparsity


# ---------------------------------------------------------------------------
# HSTUWithRankMixer
# ---------------------------------------------------------------------------


class HSTUWithRankMixer(HSTU):
    """
    HSTU encoder with a RankMixer stage inserted between the HSTU layers and
    the output post-processor.

    All constructor parameters from HSTU are accepted unchanged; the extra
    ``rankmixer_*`` parameters control the RankMixer.

    Notes
    -----
    * The RankMixer ``output_dim`` is always set equal to ``embedding_dim`` so
      that dot-product similarity and sampled-softmax loss need no modification.
    * ``embedding_dim`` must be divisible by ``rankmixer_dim_per_token``.
    * The sparsity L1 loss from RankMixer is stored in ``self.rankmixer_l1_loss``
      after each forward pass and can optionally be added to the training loss
      via ``loss_weights`` in the gin config (requires a small trainer change).
    """

    def __init__(
        self,
        # ---- HSTU arguments (unchanged) ----
        max_sequence_len: int,
        max_output_len: int,
        embedding_dim: int,
        num_blocks: int,
        num_heads: int,
        linear_dim: int,
        attention_dim: int,
        normalization: str,
        linear_config: str,
        linear_activation: str,
        linear_dropout_rate: float,
        attn_dropout_rate: float,
        embedding_module: EmbeddingModule,
        similarity_module: SimilarityModule,
        input_features_preproc_module: InputFeaturesPreprocessorModule,
        output_postproc_module: OutputPostprocessorModule,
        enable_relative_attention_bias: bool = True,
        concat_ua: bool = False,
        verbose: bool = True,
        # ---- RankMixer arguments ----
        rankmixer_dim_per_token: int = 16,
        rankmixer_num_experts: int = 4,
        rankmixer_k: float = 4.0,
        rankmixer_t_multiplier: float = 1.0,
        rankmixer_ffn_layers: int = 2,
        rankmixer_n_layers: int = 2,
        rankmixer_dropout: float = 0.05,
    ) -> None:
        super().__init__(
            max_sequence_len=max_sequence_len,
            max_output_len=max_output_len,
            embedding_dim=embedding_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            linear_dim=linear_dim,
            attention_dim=attention_dim,
            normalization=normalization,
            linear_config=linear_config,
            linear_activation=linear_activation,
            linear_dropout_rate=linear_dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            embedding_module=embedding_module,
            similarity_module=similarity_module,
            input_features_preproc_module=input_features_preproc_module,
            output_postproc_module=output_postproc_module,
            enable_relative_attention_bias=enable_relative_attention_bias,
            concat_ua=concat_ua,
            verbose=verbose,
        )

        self._rankmixer = RankMixer(
            embedding_dim=embedding_dim,
            output_dim=embedding_dim,   # keep dim so similarity stays valid
            dim_per_token=rankmixer_dim_per_token,
            num_experts=rankmixer_num_experts,
            k=rankmixer_k,
            t_multiplier=rankmixer_t_multiplier,
            ffn_layers=rankmixer_ffn_layers,
            n_layers=rankmixer_n_layers,
            dropout_p=rankmixer_dropout,
        )
        # Auxiliary losses populated after each forward pass (for logging)
        self.rankmixer_l1_loss: Optional[torch.Tensor] = None
        self.rankmixer_sparsity: Optional[torch.Tensor] = None

    def debug_str(self) -> str:
        rm = self._rankmixer
        return (
            super().debug_str()
            + f"-rm_T{rm.input_proj.num_tokens}"
            + f"_Ne{rm.blocks[0].moe.Ne}"
            + f"_L{rm.n_layers}"
        )

    def generate_user_embeddings(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
    ) -> Tuple[torch.Tensor, List[HSTUCacheState]]:
        """
        Full encoding pipeline: InputPreproc -> HSTU -> RankMixer -> OutputPostproc.

        Returns
        -------
        embeddings    : (B, N, D) — ready for similarity / loss computation
        cached_states : list of HSTU cache states (empty unless return_cache_states)
        """
        float_dtype = past_embeddings.dtype

        # 1. Input preprocessing (positional embeddings, masking, dropout)
        past_lengths, user_embeddings, _ = self._input_features_preproc(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )

        # 2. HSTU layers
        user_embeddings, cached_states = self._hstu(
            x=user_embeddings,
            x_offsets=torch.ops.fbgemm.asynchronous_complete_cumsum(past_lengths),
            all_timestamps=(
                past_payloads[TIMESTAMPS_KEY]
                if TIMESTAMPS_KEY in past_payloads
                else None
            ),
            invalid_attn_mask=1.0 - self._attn_mask.to(float_dtype),
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
        )  # (B, N, D)

        # 3. RankMixer
        user_embeddings, l1_loss, sparsity = self._rankmixer(user_embeddings)
        # Store for optional use in training / logging (access via model.module.*)
        self.rankmixer_l1_loss = l1_loss
        self.rankmixer_sparsity = sparsity

        # 4. Output postprocessor (L2 norm or LayerNorm)
        return self._output_postproc(user_embeddings), cached_states
