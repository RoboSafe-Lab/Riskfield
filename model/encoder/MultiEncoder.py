"""Multi-agent encoder.

Two output modes are supported (selected at call time, not construction time):

- **Scene-level** (``per_agent=True``): symmetric self-attention over all
  agents produces a contextualized embedding *for every agent*, plus a
  per-sample agent-validity mask. The learned ``ego_identity`` vector is
  added to index-0 (the ego) before attention so downstream modules (world
  model, AR decoder) can correlate the ego-action conditioning with the
  right agent.

- **Single-target** (``per_agent=False``): ego-as-Query, others-as-Key/Value
  cross-attention, returning a single (B, E) context vector. This is the
  original behaviour kept for backward compatibility with ``evaluate.py``
  and pre-scene-level checkpoints.

The GRU per-car histories and FiLM vehicle-type modulation are shared
between both modes.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class MultiEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        max_num_cars: int = 8,
        seq_len: int = 100,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 1,
        n_heads: int = 4,
        dropout: float = 0.0,
        num_vehicle_types: int = 5,
    ):
        super(MultiEncoder, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.type_dim = 32
        self.type_emb = nn.Embedding(num_vehicle_types, self.type_dim)
        self.film_gen = nn.Sequential(
            nn.Linear(self.type_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        # FiLM generator last layer zero-init so identity at start.
        nn.init.zeros_(self.film_gen[-1].weight)
        nn.init.zeros_(self.film_gen[-1].bias)

        self.max_num_cars = max_num_cars
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Scene-level path: learned ego identity vector + symmetric self-attn.
        self.ego_identity = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.ego_identity, std=0.02)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads,
            batch_first=True, dropout=dropout,
        )
        self.per_agent_proj = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Legacy single-target path: ego-Q / others-KV cross-attention.
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads,
            batch_first=True, dropout=dropout,
        )
        self.proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def _encode_histories(self, x):
        """Shared GRU+FiLM stage. Returns (per_car, car_valid)."""
        assert x.ndim == 4, "x must be (batch, num_cars, seq_len, feat_dim)"
        batch, num_cars, seq_len, _ = x.shape
        assert (num_cars == self.max_num_cars) and (seq_len == self.seq_len), (
            f"Expected (batch, {self.max_num_cars}, {self.seq_len}, feat); "
            f"got {tuple(x.shape)}"
        )
        x_flat = x.contiguous().view(batch * num_cars, seq_len, -1)

        time_valid = ~torch.isnan(x_flat).any(dim=-1)
        lengths = time_valid.sum(dim=1).to(torch.long)
        x_filled = x_flat.clone()
        x_filled[torch.isnan(x_filled)] = 0.0

        lengths_clamped = lengths.clone()
        lengths_clamped[lengths_clamped == 0] = 1
        lengths_sorted, sort_idx = torch.sort(lengths_clamped, descending=True)
        unsort_idx = sort_idx.argsort()
        x_sorted = x_filled[sort_idx]

        packed = pack_padded_sequence(
            x_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True,
        )
        _, h_n = self.gru(packed)
        h_last = h_n[-1][unsort_idx]
        per_car = h_last.view(batch, num_cars, self.hidden_dim)
        car_valid = (lengths.view(batch, num_cars) > 0)
        return per_car, car_valid

    def _film(self, per_car, v_type):
        t_emb = self.type_emb(v_type)            # (B,N,32)
        gamma, beta = torch.chunk(self.film_gen(t_emb), 2, dim=-1)
        return (1 + gamma) * per_car + beta

    def forward(self, t, x, v_type, per_agent=False):
        """
        x: (B, N, T, F).  Returns:
          per_agent=False -> (embedding: (B, E),  car_valid: (B, N))
          per_agent=True  -> (embedding: (B, N, E), car_valid: (B, N))
        """
        per_car, car_valid = self._encode_histories(x)
        per_car = self._film(per_car, v_type)
        device = per_car.device
        batch, num_cars, _ = per_car.shape

        if per_agent:
            # Add learned ego-identity to position 0 so attention is
            # symmetric over agents but the ego is identifiable.
            ego_mark = torch.zeros_like(per_car)
            ego_mark[:, 0, :] = self.ego_identity
            tokens = per_car + ego_mark

            # Self-attention over agents; mask invalid positions.
            kpm = ~car_valid                          # True = ignore
            # If a row is fully invalid, give it one "false" entry to avoid
            # NaN; the row's downstream use is masked anyway via car_valid.
            safe_kpm = kpm.clone()
            full_row = safe_kpm.all(dim=1)
            safe_kpm[full_row, 0] = False
            attn_out, _ = self.self_attn(
                tokens, tokens, tokens, key_padding_mask=safe_kpm,
            )
            return self.per_agent_proj(attn_out), car_valid

        # ---- legacy single-target path ----
        ego = per_car[:, 0:1, :]                       # (B, 1, hidden)
        if num_cars > 1:
            others = per_car[:, 1:, :]
            kpm = ~car_valid[:, 1:]
            full_row = kpm.all(dim=1)
            safe_kpm = kpm.clone()
            safe_kpm[full_row, 0] = False
            attn_out, _ = self.mha(ego, others, others, key_padding_mask=safe_kpm)
            combined = torch.cat([ego, attn_out], dim=-1).squeeze(1)
        else:
            combined = torch.cat(
                [ego.squeeze(1),
                 torch.zeros(batch, self.hidden_dim, device=device, dtype=per_car.dtype)],
                dim=-1,
            )
        return self.proj(combined), car_valid
