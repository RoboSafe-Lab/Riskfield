"""Autoregressive scene-level conditioning decoder.

Given a chain ordering of agents (excluding the ego), the *world-model scene
state* ``s_seq``, the *per-agent embeddings*, and the previously-decoded
agents' future positions ``Y`` (ground truth at training, MAP centroid at
inference), produces per-agent per-step conditioning for the flow:

    cond_k[i] = agent_emb[i] + residual([s_k, agent_emb[i], h_prev_k[i]])

where ``h_prev_k[i]`` is a *causal* cumulative summary of ``Y_{order[<i], k}``
along the chain ordering, computed by running a small GRU along the
agent-chain axis at each time step (parallelized over B and K).

The residual MLP's final layer is zero-initialized so at init
``cond_k[i] == agent_emb[i]`` for every agent and step (per-agent baseline
equivalence; the AR refinement can only add to it).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ARDecoder(nn.Module):
    def __init__(
        self,
        agent_emb_dim: int,
        scene_state_dim: int,
        cond_dim: int,
        pos_dim: int = 2,
        h_dim: int = 64,
    ) -> None:
        super().__init__()
        assert cond_dim == agent_emb_dim, (
            "ARDecoder residual conditioning needs cond_dim == agent_emb_dim"
        )
        self.agent_emb_dim = int(agent_emb_dim)
        self.scene_state_dim = int(scene_state_dim)
        self.cond_dim = int(cond_dim)
        self.pos_dim = int(pos_dim)
        self.h_dim = int(h_dim)

        # Embed a previously-decoded agent's per-step (x,y) -> h_dim.
        self.y_embed = nn.Sequential(
            nn.Linear(self.pos_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.h_dim),
        )
        # Causal cumulative pass along the agent-chain axis (per time step).
        self.chain_gru = nn.GRU(
            input_size=self.h_dim,
            hidden_size=self.h_dim,
            batch_first=True,
        )
        # Per-agent residual conditioning combining scene state, the agent's
        # own embedding and the causal history of previously decoded agents.
        in_dim = self.scene_state_dim + self.agent_emb_dim + self.h_dim
        self.residual = nn.Sequential(
            nn.Linear(in_dim, self.cond_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.cond_dim, self.cond_dim),
        )
        # Zero-init final layer -> at start cond_k[i] == agent_emb[i].
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        agent_emb: torch.Tensor,    # (B, N, E)
        s_seq: torch.Tensor,        # (B, K, scene_state_dim)
        Y: torch.Tensor,            # (B, N, K, 2)  -- GT at train, MAP at infer
        order: torch.Tensor,        # (B, N)  permutation of [0..N-1], chain order
    ) -> torch.Tensor:
        """Returns per-agent per-step conditioning (B, N, K, cond_dim)."""
        B, N, K, _ = Y.shape
        E = agent_emb.shape[-1]
        H = self.h_dim
        Sd = s_seq.shape[-1]

        # Embed per-agent per-step positions (NaN-safe).
        Y_safe = torch.nan_to_num(Y)
        Y_emb = self.y_embed(Y_safe.view(B * N * K, self.pos_dim)).view(B, N, K, H)

        # Reorder agents along the chain.
        idx = order.unsqueeze(-1).unsqueeze(-1).expand(B, N, K, H)
        Y_emb_ord = Y_emb.gather(1, idx)                              # (B, N, K, H)

        # Causal shift along the chain axis: input[chain_i] depends only on
        # Y_emb_ord[chain_{<i}].
        shift = torch.zeros_like(Y_emb_ord)
        shift[:, 1:] = Y_emb_ord[:, :-1]

        # Run a GRU along the chain axis, per time step (parallelized via
        # collapsing B and K into the batch dim).
        inp = shift.permute(0, 2, 1, 3).reshape(B * K, N, H)          # (B*K, N, H)
        h_ord, _ = self.chain_gru(inp)                                # (B*K, N, H)
        h_ord = h_ord.view(B, K, N, H).permute(0, 2, 1, 3)            # (B, N, K, H)

        # Un-reorder back to original agent indices.
        inv = order.argsort(dim=1)                                    # (B, N)
        inv_idx = inv.unsqueeze(-1).unsqueeze(-1).expand(B, N, K, H)
        h_prev = h_ord.gather(1, inv_idx)                             # (B, N, K, H)

        # Per-agent residual conditioning.
        s_exp = s_seq.unsqueeze(1).expand(B, N, K, Sd)
        a_exp = agent_emb.unsqueeze(2).expand(B, N, K, E)
        res_in = torch.cat([s_exp, a_exp, h_prev], dim=-1)            # (B,N,K,Sd+E+H)
        return a_exp + self.residual(res_in)                           # (B, N, K, E)
