import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from typing import Optional


class MultiEncoder(nn.Module):
    """
    Multi-car discrete encoder.
    forward(t, x):
      - t: placeholder if we need to implement a ContinuousEncoder interface.
      - x: tensor of shape (batch, num_cars, seq_len, feat_dim)
           feat_dim already includes any per-step features (pos + extras).
      - v_type: tensor of shape (batch, num_cars) with vehicle type IDs.
    Returns:
      - embedding: (batch, embedding_dim)
    Behavior:
      - For each car do a GRU over time (time appended as extra channel).
      - Use MultiheadAttention so ego (index 0) attends to other cars.
      - Project concatenated (ego, attended) to embedding_dim.
    """

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
        # NOTE: InD dataset already appends a time channel to features.
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

        # Initialize FiLM generator last layer to zeros
        nn.init.zeros_(self.film_gen[-1].weight)
        nn.init.zeros_(self.film_gen[-1].bias)

        self.max_num_cars = max_num_cars
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # cross-attention: ego queries others
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, batch_first=True, dropout=dropout
        )

        # projection from [ego_hidden, attn_hidden] -> embedding_dim
        self.proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(
        self, t: torch.Tensor, x: torch.Tensor, v_type: torch.Tensor
    ) -> torch.Tensor:
        """
        t: (seq_len,) tensor (device will follow x) -- kept for API compat but NOT used
        x: (batch, num_cars, seq_len, feat_dim)  # feat_dim already includes time if dataset added it
        returns: (batch, embedding_dim)
        """
        assert x.ndim == 4, "x must be (batch, num_cars, seq_len, feat_dim)"
        batch, num_cars, seq_len, feat_dim = x.shape
        assert (num_cars == self.max_num_cars) and (seq_len == self.seq_len), (
            f"Expected input shape (batch, {self.max_num_cars}, {self.seq_len}, feat_dim), "
            f"but got (batch, {num_cars}, {seq_len}, {feat_dim})"
        )
        device = x.device

        ## 1. GRU per car ##
        # flatten cars: (batch*num_cars, seq_len, feat_dim)
        x_flat = x.contiguous().view(batch * num_cars, seq_len, feat_dim)

        # compute valid lengths per car from NaNs: consider a time-step valid only if all features are not NaN
        time_valid = ~torch.isnan(x_flat).any(dim=-1)  # (batch*num_cars, seq_len) bool
        lengths = time_valid.sum(dim=1).to(torch.long)  # (batch*num_cars,)

        # replace NaNs with 0 before feeding to GRU (so pack/GRU won't propagate NaN)
        x_filled = x_flat.clone()
        x_filled[torch.isnan(x_filled)] = 0.0

        # NOTE: Do NOT append t here because datloader already adds a time channel.
        gru_input = x_filled  # (batch*num_cars, seq_len, feat_dim)

        # ensure at least length 1 for pack (pack doesn't accept zero-length). invalid sequences will be masked later.
        lengths_clamped = lengths.clone()
        lengths_clamped[lengths_clamped == 0] = 1

        # sort by length desc for pack_padded_sequence
        lengths_sorted, sort_idx = torch.sort(lengths_clamped, descending=True)
        unsort_idx = sort_idx.argsort()
        gru_input_sorted = gru_input[sort_idx]

        packed = pack_padded_sequence(
            gru_input_sorted,
            lengths_sorted.cpu(),
            batch_first=True,
            enforce_sorted=True,
        )
        _, h_n = self.gru(packed)  # h_n: (num_layers, batch_sorted, hidden_dim)
        h_last = h_n[-1]  # (batch_sorted, hidden_dim)
        # unsort back to original order
        h_last = h_last[unsort_idx]

        # reshape per-car embeddings: (batch, num_cars, hidden_dim)
        per_car = h_last.view(batch, num_cars, self.hidden_dim)

        ## 2. FiLM modulation ##
        # print(f"t_emb max index: {v_type.max()}") 
        t_emb = self.type_emb(v_type)  # (B, N, 32)
        film_params = self.film_gen(t_emb)  # (B, N, hidden_dim*2)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)  # (B, N, hidden_dim) each

        per_car = (1 + gamma) * per_car + beta  # FiLM modulation

        ## 3. Cross-car attention ##
        # car-level valid mask: a car is valid if its original length>0
        car_valid = lengths.view(batch, num_cars) > 0  # (batch, num_cars) bool

        # ego embedding (index 0)
        ego = per_car[:, 0:1, :]  # (batch, 1, hidden_dim)

        # if there are other cars, attend
        if num_cars > 1:
            others = per_car[:, 1:, :]  # (batch, num_cars-1, hidden_dim)
            # key_padding_mask: True for positions that should be ignored -> invalid cars should be True
            key_padding_mask = ~car_valid[:, 1:]  # (batch, num_cars-1)
            ##
            full_masked_rows = key_padding_mask.all(dim=1) # (batch,)
            safe_mask = key_padding_mask.clone()
            safe_mask[full_masked_rows, 0] = False
            ##
            # MultiheadAttention requires float tensors; queries/keys/values are (batch, seq, embed)
            # If all other cars are invalid for a batch row, MHA will return zeros — that's acceptable.
            attn_out, _ = self.mha(
                ego, others, others, key_padding_mask=safe_mask
            )
            if torch.isnan(attn_out).any():
                print("MHA produced NaN!")
                print("Mask sums per row:", safe_mask.sum(dim=1))

            # attn_out: (batch, 1, hidden_dim)
            combined = torch.cat([ego, attn_out], dim=-1).squeeze(
                1
            )  # (batch, 2*hidden_dim)
        else:
            # no other cars -> pad attn with zeros
            zero_attn = torch.zeros(
                batch, self.hidden_dim, device=device, dtype=per_car.dtype
            )
            combined = torch.cat(
                [ego.squeeze(1), zero_attn], dim=-1
            )  # (batch, 2*hidden_dim)

        embedding = self.proj(combined)  # (batch, embedding_dim)
        return embedding
