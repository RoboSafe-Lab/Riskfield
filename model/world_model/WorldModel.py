"""Deterministic latent world model that conditions the normalizing flow.

Two complementary entry points:

- **Single-target** ``forward(context, n_steps, actions, return_dyn)``:
  rolls ``s_0 = psi(c)``, ``s_k = g(s_{k-1}, u_k)`` and emits a per-frame
  *residual* conditioning ``cond_k = c + residual([s_k, c])`` for the flow.
  Last layer of ``residual`` is zero-initialized so at init ``cond_k == c``
  (equivalent to the strong no-WM baseline; the model can only refine).
  Used in single-target legacy mode (`evaluate.py` etc.).

- **Scene-level** ``forward_scene(per_agent_emb, car_valid, n_steps, actions,
  return_dyn)``: pools per-agent embeddings into a scene context (masked mean
  plus a residual from the ego at index 0), then rolls a *shared* scene
  latent ``s_seq`` `(B, K, state_dim)` driven by the ego's action. Per-agent
  conditioning is produced by the AR decoder using this ``s_seq``.

The control input ``u_k`` is the embedded ego action if supplied,
otherwise a learned autonomous vector (trained via action-dropout).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class WorldModel(nn.Module):
    def __init__(
        self,
        context_dim: int,
        cond_dim: int,
        state_dim: int = 256,
        action_dim: int = 2,
        pos_dim: int = 2,
    ) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        self.cond_dim = int(cond_dim)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.pos_dim = int(pos_dim)
        assert self.cond_dim == self.context_dim, (
            "WorldModel residual conditioning needs cond_dim == context_dim"
        )

        self.init_state = nn.Sequential(
            nn.Linear(self.context_dim, self.state_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.state_dim, self.state_dim),
        )

        self.action_emb = nn.Linear(self.action_dim, self.state_dim)
        self.autonomous_u = nn.Parameter(torch.zeros(self.state_dim))

        self.cell = nn.GRUCell(self.state_dim, self.state_dim)

        # Single-target residual readout. Zero-init -> cond_k == c at start.
        self.residual = nn.Sequential(
            nn.Linear(self.state_dim + self.context_dim, self.cond_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.cond_dim, self.cond_dim),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

        # Dyn aux head: predict the *driving agent*'s next position from s_k.
        self.dyn_head = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.state_dim, self.pos_dim),
        )

    # ----- internal rollout (shared) ---------------------------------------
    def _rollout(self, c_scene: torch.Tensor, n_steps: int,
                 actions: Optional[torch.Tensor]):
        bsz = c_scene.shape[0]
        n_steps = int(n_steps)
        if actions is not None:
            if actions.shape[0] != bsz or actions.shape[1] != n_steps:
                raise ValueError(
                    f"actions must be (B={bsz}, K={n_steps}, "
                    f"action_dim={self.action_dim}); got {tuple(actions.shape)}"
                )
            u_seq = self.action_emb(actions)            # (B, K, state_dim)
        else:
            u_seq = None
        s = self.init_state(c_scene)
        s_list = []
        for k in range(n_steps):
            u = u_seq[:, k] if u_seq is not None else self.autonomous_u.expand(bsz, -1)
            s = self.cell(u, s)
            s_list.append(s)
        return torch.stack(s_list, dim=1)               # (B, K, state_dim)

    # ----- single-target API (legacy) --------------------------------------
    def forward(self, context: torch.Tensor, n_steps: int,
                actions: Optional[torch.Tensor] = None,
                return_dyn: bool = False):
        """Returns per-frame conditioning (B, n_steps, cond_dim) for a
        single-target flow, and optionally next-position predictions."""
        if context.ndim != 2:
            raise ValueError(
                f"WorldModel.forward expects context (B, context_dim); "
                f"got {tuple(context.shape)}"
            )
        s_seq = self._rollout(context, n_steps, actions)
        # cond_k = c + residual([s_k, c])  (zero-init residual -> cond_k == c)
        c_rep = context.unsqueeze(1).expand(-1, n_steps, -1)
        cond = c_rep + self.residual(torch.cat([s_seq, c_rep], dim=-1))
        if return_dyn:
            return cond, self.dyn_head(s_seq)
        return cond

    # ----- scene-level API -------------------------------------------------
    def forward_scene(self, per_agent_emb: torch.Tensor,
                      car_valid: torch.Tensor,
                      n_steps: int,
                      actions: Optional[torch.Tensor] = None,
                      return_dyn: bool = False):
        """Pool per-agent embeddings into a scene context with an ego
        residual, then roll the scene state forward conditioned on the ego
        action. Returns the scene-state sequence ``s_seq`` (B, K, state_dim)
        for the AR decoder; with ``return_dyn`` also returns the ego-future
        next-position predictions for the auxiliary loss.
        """
        if per_agent_emb.ndim != 3:
            raise ValueError(
                f"forward_scene expects (B,N,E); got {tuple(per_agent_emb.shape)}"
            )
        # Masked mean over valid agents, with a residual from the ego (idx 0).
        mask = car_valid.to(per_agent_emb.dtype).unsqueeze(-1)         # (B,N,1)
        denom = mask.sum(dim=1).clamp(min=1.0)                          # (B,1)
        pooled = (per_agent_emb * mask).sum(dim=1) / denom              # (B,E)
        ego = per_agent_emb[:, 0, :]                                    # (B,E)
        c_scene = pooled + ego                                          # ego survives pooling
        s_seq = self._rollout(c_scene, n_steps, actions)
        if return_dyn:
            return s_seq, self.dyn_head(s_seq)
        return s_seq
