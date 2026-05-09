import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from model.encoder.MultiEncoder import MultiEncoder
from model.flow.DNF import DNF


class RiskFlow(nn.Module):
    def __init__(
        self,
        seq_len=100,
        input_dim=2,
        feature_dim=5,
        embedding_dim=256,
        hidden_dim=512,
        max_num_cars=4,
        num_classes=5,
        gru_layers=4,
        num_heads=8,
        dropout=0.1,
        norm_rotation=True,
        flow_layers=3,
        flow_hidden_dim=512,
        coupling_layers=10,
        use_cnf=False,
        use_cgmm: bool = False,
        gmm_modes: int = 3,
        gmm_min_sigma: float = 1e-3,
    ):
        super(RiskFlow, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_num_cars = max_num_cars
        self.norm_rotation = norm_rotation
        flow_input_dim = input_dim

        # Conditional GMM base distribution for DNF latent space
        # If use_cgmm is False, or gmm_modes <= 1 (or CNF is used), we fall back to the original standard Normal base.
        self.use_cgmm = bool(use_cgmm)
        self.gmm_modes = int(gmm_modes)
        self.gmm_min_sigma = float(gmm_min_sigma)

        # MultiEncoder for multi-car input
        self.encoder = MultiEncoder(
            input_dim=input_dim + feature_dim + 1,  # +1 for time channel
            max_num_cars=max_num_cars,
            seq_len=seq_len,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=gru_layers,
            n_heads=num_heads,
            dropout=dropout,
            num_vehicle_types=num_classes,
        )

        # Normalizing flow
        self.use_cnf = use_cnf
        if use_cnf:
            # Lazy import so DNF users don't need CNF deps (e.g., torchdiffeq)
            from model.flow.CNF import CNF
            self.flow = CNF(
                input_dim=flow_input_dim,
                condition_dim=embedding_dim,
                hidden_dims=[flow_hidden_dim] * flow_layers,
            )
        else:
            self.flow = DNF(
                n_blocks=flow_layers,
                input_size=flow_input_dim,
                hidden_size=flow_hidden_dim,
                n_hidden=coupling_layers,
                cond_label_size=embedding_dim,
            )

        # Small head to parameterize a conditional diagonal-covariance GMM in latent space
        # Params per batch element:
        #   pi_logits: (B, K)
        #   mu:        (B, K, D)
        #   sigma:     (B, K, D)  (positive via softplus)
        if self.use_cgmm and (not self.use_cnf) and self.gmm_modes > 1:
            out_dim = self.gmm_modes * (1 + 2 * self.input_dim)
            self.gmm_head = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embedding_dim, out_dim),
            )
        else:
            self.gmm_head = None

    def _cgmm_params(self, embedding: torch.Tensor):
        """Compute conditional GMM parameters from encoder embedding."""
        if self.gmm_head is None:
            raise RuntimeError("Conditional GMM head is disabled (gmm_modes <= 1 or use_cnf=True).")

        bsz = embedding.shape[0]
        raw = self.gmm_head(embedding)  # (B, K*(1+2D))

        k = self.gmm_modes
        d = self.input_dim
        pi_logits = raw[:, :k]
        rest = raw[:, k:]
        mu = rest[:, : k * d].view(bsz, k, d)
        sigma_raw = rest[:, k * d :].view(bsz, k, d)
        sigma = F.softplus(sigma_raw) + self.gmm_min_sigma
        return pi_logits, mu, sigma

    @staticmethod
    def _standard_normal_log_prob_per_frame(y: torch.Tensor) -> torch.Tensor:
        """Log prob of standard Normal N(0,I) per frame.

        Args:
            y: (..., T, D)
        Returns:
            logp: (..., T)
        """
        d = y.shape[-1]
        # log N(y;0,I) = -0.5*(||y||^2 + D*log(2*pi))
        return -0.5 * (y.pow(2).sum(dim=-1) + float(d) * torch.log(torch.tensor(2.0 * torch.pi, device=y.device, dtype=y.dtype)))

    def _cgmm_log_prob_per_frame(self, y: torch.Tensor, *, embedding: torch.Tensor) -> torch.Tensor:
        """Log prob under conditional diagonal-covariance GMM per frame.

        Args:
            y: (B, S, T, D)
            embedding: (B, E)
        Returns:
            logp: (B, S, T)
        """
        pi_logits, mu, sigma = self._cgmm_params(embedding)  # (B,K), (B,K,D), (B,K,D)
        log_pi = torch.log_softmax(pi_logits, dim=-1)  # (B,K)

        # Expand to broadcast over samples/time
        # y: (B,S,T,D)
        y_e = y.unsqueeze(3)  # (B,S,T,1,D)
        mu_e = mu.unsqueeze(1).unsqueeze(1)  # (B,1,1,K,D)
        sigma_e = sigma.unsqueeze(1).unsqueeze(1)  # (B,1,1,K,D)

        # Diagonal Gaussian log-prob summed over dims -> (B,S,T,K)
        # log N = -0.5 * [((y-mu)/sigma)^2 + 2*log(sigma) + log(2*pi)]
        log2pi = torch.log(torch.tensor(2.0 * torch.pi, device=y.device, dtype=y.dtype))
        z = (y_e - mu_e) / sigma_e
        log_comp = (-0.5 * (z.pow(2) + 2.0 * torch.log(sigma_e) + log2pi)).sum(dim=-1)

        # Mixture log-sum-exp over K
        return torch.logsumexp(log_pi.unsqueeze(1).unsqueeze(1) + log_comp, dim=-1)  # (B,S,T)

    def _abs_to_rel(self, y, x_t):
        y_rel = y - x_t
        y_rel[:, :, 1:] = y_rel[:, :, 1:] - y_rel[:, :, :-1]
        return y_rel

    def _rel_to_abs(self, y_rel, x_t):
        y_abs = torch.cumsum(y_rel, dim=-2) + x_t
        return y_abs

    def _rotate(self, x, x_t, angles_rad):
        c, s = torch.cos(angles_rad), torch.sin(angles_rad)
        c, s = c.unsqueeze(1), s.unsqueeze(1)
        x_center = x - x_t
        x_vals, y_vals = x_center[..., 0], x_center[..., 1]
        new_x_vals = c * x_vals + (-1 * s) * y_vals
        new_y_vals = s * x_vals + c * y_vals
        x_center[..., 0] = new_x_vals
        x_center[..., 1] = new_y_vals
        return x_center + x_t

    def _rotate_features(self, features, angles_rad):
        c, s = torch.cos(angles_rad), torch.sin(angles_rad)
        c, s = c.unsqueeze(1), s.unsqueeze(1)
        vx_vals, vy_vals = features[..., 0], features[..., 1]
        ax_vals, ay_vals = features[..., 2], features[..., 3]
        new_vx_vals = c * vx_vals + (-1 * s) * vy_vals
        new_vy_vals = s * vx_vals + c * vy_vals
        new_ax_vals = c * ax_vals + (-1 * s) * ay_vals
        new_ay_vals = s * ax_vals + c * ay_vals
        features[..., 0] = new_vx_vals
        features[..., 1] = new_vy_vals
        features[..., 2] = new_ax_vals
        features[..., 3] = new_ay_vals
        return features

    def _normalize_rotation(self, x, y_true=None):
        x_t = x[:, :, -1:, :]
        x_t_rel = x[:, :, -1] - x[:, :, -2]
        rot_angles_rad = -1 * torch.atan2(x_t_rel[:, 0, 1], x_t_rel[:, 0, 0])
        x = self._rotate(x, x_t, rot_angles_rad)

        if y_true is not None:
            y_true = self._rotate(y_true, x_t, rot_angles_rad)
            return x, y_true, rot_angles_rad

        return x, rot_angles_rad

    def forward(self, x, y, feat, v_type, sampling_frequency=1):
        if self.norm_rotation:
            x, y, angle = self._normalize_rotation(x, y)
            feat = self._rotate_features(feat, angle)

        # Encode multi-car input
        embedding = self.encoder(None, torch.cat([x, feat], dim=-1), v_type)

        # Flow forward pass
        z, det = self.flow(y, embedding, sampling_frequency=sampling_frequency)

        return z, det, embedding

    def sample(self, x, feat, futures, v_type, num_samples=1, sampling_frequency=1):
        if self.norm_rotation:
            x, angle = self._normalize_rotation(x)
            feat = self._rotate_features(feat, angle)

        # NOTE: This sampling path assumes batch size == 1 and uses `num_samples` as the flow batch.
        # This matches the previous implementation and current evaluation usage.
        if x.shape[0] != 1:
            raise ValueError(
                f"MultiTrajFlow.sample currently expects batch_size==1, got {x.shape[0]}."
            )
        embedding = self.encoder(None, torch.cat([x, feat], dim=-1), v_type)

        if (not self.use_cnf) and self.use_cgmm and (self.gmm_head is not None):
            pi_logits, mu, sigma = self._cgmm_params(embedding)  # (1,K), (1,K,D), (1,K,D)
            pi = torch.softmax(pi_logits, dim=-1).squeeze(0)  # (K,)
            mu0 = mu.squeeze(0)  # (K,D)
            sigma0 = sigma.squeeze(0)  # (K,D)

            cat = torch.distributions.Categorical(probs=pi)
            comp_idx = cat.sample((num_samples, futures))  # (S, futures)
            mu_sel = mu0[comp_idx]  # (S, futures, D)
            sigma_sel = sigma0[comp_idx]  # (S, futures, D)
            eps = torch.randn_like(mu_sel)
            y = mu_sel + sigma_sel * eps
        else:
            mean = torch.zeros(futures, self.input_dim, device=x.device)
            variance = torch.ones(futures, self.input_dim, device=x.device)
            base_dist = torch.distributions.MultivariateNormal(
                mean, torch.diag_embed(variance)
            )
            y = torch.stack([base_dist.sample().to(x.device) for _ in range(num_samples)])

        embedding = embedding.expand(y.shape[0], embedding.shape[1])
        z, det = self.flow(
            y, embedding, reverse=True, sampling_frequency=sampling_frequency
        )

        if self.norm_rotation:
            x_t = x[:, :, -1:, :]
            z = self._rotate(z, x_t, -1 * angle)

        z = z[:, :futures, :]
        return y, z, det

    def rsample_x_and_logp(
        self,
        x: torch.Tensor,
        feat: torch.Tensor,
        v_type: torch.Tensor,
        *,
        futures: int,
        num_samples: int = 1,
        sampling_frequency: int = 1,
    ):
        """Reparameterized sampling for training-time regularization.

        Returns samples in *data space* and their per-frame log-prob under the model.

        This is designed to be batch-capable and fully differentiable w.r.t. model
        parameters (except for discrete mixture component sampling when CGMM is
        enabled).

        Args:
            x: (B, max_num_cars, seq_len, 2)
            feat: (B, max_num_cars, seq_len, feature_dim)
            v_type: (B, max_num_cars)
            futures: T (number of future frames)
            num_samples: S

        Returns:
            x_samples: (B, S, T, 2)
            logp_x:    (B, S, T)
        """
        if self.use_cnf:
            raise NotImplementedError(
                "rsample_x_and_logp is not enabled for CNF (too slow for training-time regularization)."
            )

        bsz = x.shape[0]
        t = int(futures)
        s = int(num_samples)

        if self.norm_rotation:
            x, angle = self._normalize_rotation(x)
            feat = self._rotate_features(feat, angle)

        embedding = self.encoder(None, torch.cat([x, feat], dim=-1), v_type)  # (B,E)

        # Sample from base (latent) distribution y_base: (B,S,T,D)
        if (not self.use_cnf) and self.use_cgmm and (self.gmm_head is not None):
            pi_logits, mu, sigma = self._cgmm_params(embedding)  # (B,K), (B,K,D), (B,K,D)
            pi = torch.softmax(pi_logits, dim=-1)  # (B,K)

            cat = torch.distributions.Categorical(probs=pi)
            comp_idx = cat.sample((s, t)).permute(2, 0, 1).contiguous()  # (B,S,T)

            # Gather mu/sigma per chosen component
            # Expand to (B,S,T,K,D) then gather at K using comp_idx
            idx = comp_idx.unsqueeze(-1).unsqueeze(-1).expand(bsz, s, t, 1, self.input_dim)  # (B,S,T,1,D)
            mu_sel = (
                mu.unsqueeze(1).unsqueeze(1)
                .expand(bsz, s, t, self.gmm_modes, self.input_dim)
                .gather(dim=3, index=idx)
                .squeeze(3)
            )  # (B,S,T,D)
            sigma_sel = (
                sigma.unsqueeze(1).unsqueeze(1)
                .expand(bsz, s, t, self.gmm_modes, self.input_dim)
                .gather(dim=3, index=idx)
                .squeeze(3)
            )  # (B,S,T,D)

            eps = torch.randn_like(mu_sel)
            y_base = mu_sel + sigma_sel * eps
            logp_base = self._cgmm_log_prob_per_frame(y_base, embedding=embedding)
        else:
            y_base = torch.randn(bsz, s, t, self.input_dim, device=x.device, dtype=x.dtype)
            logp_base = self._standard_normal_log_prob_per_frame(y_base)  # (B,S,T)

        # Flow inverse: latent -> data
        y_flat = y_base.reshape(bsz * s, t, self.input_dim)
        emb_rep = embedding.unsqueeze(1).expand(-1, s, -1).reshape(bsz * s, -1)
        x_flat, det = self.flow(y_flat, emb_rep, reverse=True, sampling_frequency=sampling_frequency)
        x_samples = x_flat.reshape(bsz, s, t, self.input_dim)
        det = det.reshape(bsz, s, t)

        # For reverse=True in DNF, det corresponds to log|d(latent)/d(data)| per frame.
        # So log p(data) = log p(base) + det.
        logp_x = logp_base + det

        if self.norm_rotation:
            x_t = x[:, :, -1:, :]
            x_samples = self._rotate(x_samples, x_t, -1 * angle)

        return x_samples, logp_x

    def log_prob(self, z, det, embedding: Optional[torch.Tensor] = None):

        if (not self.use_cnf) and self.use_cgmm and (self.gmm_head is not None):
            if embedding is None:
                raise ValueError("embedding is required for conditional GMM log_prob.")

            pi_logits, mu, sigma = self._cgmm_params(embedding)  # (B,K), (B,K,D), (B,K,D)
            log_pi = torch.log_softmax(pi_logits, dim=-1)  # (B,K)

            z_e = z.unsqueeze(2)  # (B,N,1,D)
            mu_e = mu.unsqueeze(1)  # (B,1,K,D)
            sigma_e = sigma.unsqueeze(1)  # (B,1,K,D)

            # log N(z | mu, sigma) summed over dims -> (B,N,K)
            log_comp = torch.distributions.Normal(mu_e, sigma_e).log_prob(z_e).sum(dim=-1)
            logpz = torch.logsumexp(log_pi.unsqueeze(1) + log_comp, dim=-1)  # (B,N)
        else:
            # Original standard Normal base
            n_steps = z.shape[1]
            input_dim = z.shape[2]
            mean = torch.zeros(n_steps, input_dim, device=z.device)
            variance = torch.ones(n_steps, input_dim, device=z.device)

            base_dist = torch.distributions.MultivariateNormal(
                mean, torch.diag_embed(variance)
            )
            logpz = base_dist.log_prob(z)

        # Negative to match CNF formulation
        logpx = logpz - det
        return logpz, logpx
