import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from model.encoder.MultiEncoder import MultiEncoder
from model.flow.DNF import DNF
from model.world_model.WorldModel import WorldModel
from model.decoder.ARDecoder import ARDecoder


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
        use_world_model: bool = False,
        wm_state_dim: int = 256,
        action_dim: int = 2,
        scene_level: bool = False,
        agent_ordering: str = "nearest_ego",
        use_map: bool = False,
        map_size: int = 64,
        map_data_dir: str = "data",
        map_dataset: str = "ind",
        map_local: bool = False,
        map_crop_m: float = 40.0,
        map_raster_res: int = 192,
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

        # Deterministic latent world model (temporal backbone). When enabled,
        # the flow is conditioned on a per-frame rolled-out state instead of a
        # single context vector + raw frame index. The conference-version
        # behavior is exactly use_world_model=False.
        self.use_world_model = bool(use_world_model)
        if self.use_world_model:
            if use_cnf:
                raise NotImplementedError(
                    "use_world_model is currently supported only with the DNF flow "
                    "(use_cnf=False)."
                )
            self.world_model = WorldModel(
                context_dim=embedding_dim,
                cond_dim=embedding_dim,  # keep flow cond dims unchanged
                state_dim=int(wm_state_dim),
                action_dim=int(action_dim),
            )
        else:
            self.world_model = None

        # Scene-level autoregressive joint over agents. Requires the world
        # model (provides the scene state s_seq).
        self.scene_level = bool(scene_level)
        self.agent_ordering = str(agent_ordering)
        if self.scene_level:
            if not self.use_world_model:
                raise NotImplementedError(
                    "scene_level=True requires use_world_model=True."
                )
            if use_cnf:
                raise NotImplementedError(
                    "scene_level=True is only supported with the DNF flow."
                )
            self.ar_decoder = ARDecoder(
                agent_emb_dim=embedding_dim,
                scene_state_dim=int(wm_state_dim),
                cond_dim=embedding_dim,
                pos_dim=input_dim,
                h_dim=64,
            )
        else:
            self.ar_decoder = None

        # Map conditioning: per-location BEV raster of the drone orthophoto,
        # encoded and ADDED to the agent embeddings so predictions follow the
        # road. Static per location -> precompute and look up by locationId.
        self.use_map = bool(use_map)
        # map_local=True -> per-agent LOCAL heading-agnostic crops (localizes the
        # prediction to the road right around each agent; fixes the diffuse,
        # poorly-localized occupancy on curved geometry). False -> the original
        # single per-location raster vector (kept so old checkpoints still load).
        self.map_local = bool(map_local) and self.use_map
        self.map_size = int(map_size)
        self.map_crop_m = float(map_crop_m)
        if self.use_map:
            from model.MapEncoder import MapEncoder
            md = str(map_dataset).lower()
            boundaries = None
            if md == "ad4che":
                from datasets.AD4CHE import SCENES, compute_scene_boundaries
                scenes = [int(s) for s in SCENES]
                if self.map_local:               # per-scene rasters in the normalization box
                    from model.MapEncoder import build_ad4che_rasters
                    boundaries = compute_scene_boundaries(map_data_dir)
                    maps = build_ad4che_rasters(map_data_dir, scenes, boundaries, int(map_raster_res))
                else:
                    from model.MapEncoder import build_ad4che_maps
                    maps = build_ad4che_maps(map_data_dir, scenes, map_size)
            else:
                from model.MapEncoder import build_location_maps
                if md == "round":
                    from datasets.RounD import (RounD as _L, ROUND_BOUNDARIES as _B,
                                                ROUND_BG_SCALE_DOWN as _SD)
                else:
                    from datasets.InD import InD as _L, LOCATION_SPATIAL_BOUNDARIES as _B
                    _SD = 12.0
                boundaries = _B
                res = int(map_raster_res) if self.map_local else map_size
                maps = build_location_maps(map_data_dir, _L.LOCATION_RECORDINGS, _B, res,
                                           scale_down=_SD)
            nloc = max(maps.keys()) + 1
            if self.map_local:                    # hi-res per-location raster + box (metres)
                R = int(map_raster_res)
                rbuf = torch.zeros(nloc, 3, R, R); boxm = torch.zeros(nloc, 2)
                for loc, mp in maps.items():
                    rbuf[loc] = torch.from_numpy(mp)
                    (xlo, xhi), (ylo, yhi) = boundaries[loc]
                    boxm[loc] = torch.tensor([xhi - xlo, yhi - ylo])
                self.register_buffer("loc_rasters", rbuf)
                self.register_buffer("loc_box_m", boxm)
            else:                                 # one per-location vector (looked up by locationId)
                buf = torch.zeros(nloc, 3, map_size, map_size)
                for loc, mp in maps.items():
                    buf[loc] = torch.from_numpy(mp)
                self.register_buffer("loc_maps", buf)
            self.map_encoder = MapEncoder(in_ch=3, emb_dim=embedding_dim)
        else:
            self.map_encoder = None

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

        # Velocity head: factorized p(v | x,y,state) ~ Gaussian mean. Maps a
        # position + the per-frame flow conditioning -> (vx,vy), giving a learned
        # per-cell velocity field for the kinetic-energy severity (replaces
        # post-hoc differentiation / optical-flow of the occupancy). Trained
        # head-only with the rest of the model frozen, so the occupancy density
        # (and detection) are unchanged.
        self.velocity_head = nn.Sequential(
            nn.Linear(input_dim + embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, input_dim),
        )

    def velocity_at(self, y, cond):
        """Learned velocity E[v | position y, per-frame conditioning cond].
        y (...,input_dim), cond (...,embedding_dim) broadcastable -> (...,input_dim)."""
        if cond.shape[:-1] != y.shape[:-1]:
            cond = cond.expand(*y.shape[:-1], cond.shape[-1])
        return self.velocity_head(torch.cat([y, cond], dim=-1))

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

    def _flow_condition(self, embedding, n_steps, actions=None):
        """Single-target conditioning (legacy path).

        With the world model enabled this is a per-frame rolled-out state
        sequence ``(B, n_steps, E)``. Otherwise it is the single context vector
        ``(B, E)``. Scene-level conditioning is built by the AR decoder in
        ``forward``; this helper is unused on that path.
        """
        if self.use_world_model and self.world_model is not None:
            return self.world_model(embedding, n_steps, actions)
        return embedding

    def _compute_ordering(self, x, car_valid):
        """Per-sample chain ordering of agents (B, N).

        For ``agent_ordering="nearest_ego"``: the ego stays at slot 0; the
        remaining slots list the neighbours sorted by distance to the ego's
        last history position (closest first). Invalid agents are pushed to
        the end so they're decoded last (their loss is masked by the caller).
        """
        B, N = car_valid.shape
        ego = x[:, 0:1, -1, :]                                       # (B,1,2)
        oth = x[:, 1:, -1, :]                                        # (B,N-1,2)
        valid = car_valid[:, 1:]                                     # (B,N-1)
        valid = valid & ~torch.isnan(oth).any(dim=-1)
        dist = (oth - ego).norm(dim=-1)
        dist = torch.where(valid, dist, dist.new_full((), float("inf")))
        sort_idx = dist.argsort(dim=1) + 1                            # (B,N-1)
        ego_col = torch.zeros(B, 1, dtype=sort_idx.dtype, device=sort_idx.device)
        return torch.cat([ego_col, sort_idx], dim=1)                  # (B,N)

    def _map_emb(self, location_id, device):
        """Global per-location map embedding (B, E) for the given locationId, or None."""
        if not self.use_map or self.map_encoder is None or location_id is None or self.map_local:
            return None
        lid = location_id.long().reshape(-1).clamp(0, self.loc_maps.shape[0] - 1)
        return self.map_encoder(self.loc_maps[lid].to(device))

    def _agent_map_emb(self, x, location_id):
        """Per-agent LOCAL map-crop embedding (B, N, E), or None. Crops an
        ``map_crop_m`` metre window of the location raster centred on each agent's
        current position and encodes it, so every agent is conditioned on the road
        right around it (not one vector for the whole location)."""
        if not self.use_map or self.map_encoder is None or location_id is None or not self.map_local:
            return None
        from model.MapEncoder import crop_agent_maps
        B, N = x.shape[0], x.shape[1]; dev = x.device
        lid = location_id.long().reshape(-1).clamp(0, self.loc_rasters.shape[0] - 1)
        rast = self.loc_rasters[lid].to(dev)                 # (B,3,R,R)
        boxm = self.loc_box_m[lid].to(dev)                   # (B,2) metres
        pos = torch.nan_to_num(x[:, :, -1, :], nan=0.5).clamp(0.0, 1.0)   # (B,N,2) normalized
        hw = ((self.map_crop_m / 2.0) / boxm)[:, None, :].expand(B, N, 2)  # normalized half-width
        crops = crop_agent_maps(rast, pos, hw, self.map_size)             # (B,N,3,S,S)
        emb = self.map_encoder(crops.reshape(B * N, 3, self.map_size, self.map_size))
        return emb.reshape(B, N, -1)

    def forward(self, x, y, feat, v_type, sampling_frequency=1, actions=None,
                return_aux=False, location_id=None, return_vel=False):
        """Two-mode forward.

        - ``y.ndim == 3`` -> single-target legacy path. ``y`` is (B, K, 2),
          the index-0 ego's future. Returns (z, det, embedding [, aux]) with
          z, det shaped (B, K, ...).
        - ``y.ndim == 4`` -> scene-level autoregressive path. ``y`` is
          (B, N, K, 2), every agent's ground-truth future (NaN where absent).
          Returns (z, det, agent_emb [, aux]) with z, det shaped
          (B, N, K, ...) and agent_emb (B, N, E).
        """
        # ------------------------------------------------------------------
        # SCENE-LEVEL PATH
        # ------------------------------------------------------------------
        if y.ndim == 4:
            if not self.scene_level or self.ar_decoder is None:
                raise RuntimeError(
                    "Got 4-D y but the model was built with scene_level=False."
                )
            if self.norm_rotation:
                # Index-0 ego rotation only; futures of other agents would
                # need their own rotation, so we keep norm_rotation off for
                # scene-level (config: norm_rotate=False, default).
                raise NotImplementedError(
                    "scene_level=True requires norm_rotation=False."
                )

            B, N, K, _ = y.shape
            agent_emb, car_valid = self.encoder(
                None, torch.cat([x, feat], dim=-1), v_type, per_agent=True,
            )  # (B, N, E), (B, N)
            amemb = self._agent_map_emb(x, location_id)        # per-agent local crop
            if amemb is not None:
                agent_emb = agent_emb + amemb                  # (B,N,E)
            else:
                memb = self._map_emb(location_id, x.device)    # global per-location vector
                if memb is not None:
                    agent_emb = agent_emb + memb.unsqueeze(1)

            if return_aux:
                s_seq, dyn_pred = self.world_model.forward_scene(
                    agent_emb, car_valid, K, actions, return_dyn=True,
                )                                              # (B,K,Sd), (B,K,2)
            else:
                s_seq = self.world_model.forward_scene(
                    agent_emb, car_valid, K, actions, return_dyn=False,
                )
                dyn_pred = None

            # Chain ordering (ego at 0, neighbours sorted by distance).
            order = self._compute_ordering(x, car_valid)        # (B, N)

            # AR decoder: per-agent per-step conditioning using ground-truth
            # Y_{<i} (teacher forcing during training).
            cond = self.ar_decoder(agent_emb, s_seq, y, order)  # (B,N,K,E)

            # Batch agents into the flow.
            y_flat = y.reshape(B * N, K, self.input_dim)
            cond_flat = cond.reshape(B * N, K, cond.shape[-1])
            # Replace NaN ys with zeros so the flow doesn't propagate them;
            # the loss masks invalid agents per-sample in train.py.
            y_flat = torch.nan_to_num(y_flat)
            z_flat, det_flat = self.flow(
                y_flat, cond_flat, sampling_frequency=sampling_frequency,
            )
            z = z_flat.view(B, N, K, self.input_dim)
            det = det_flat.view(B, N, K)

            if return_aux or return_vel:
                aux = z.new_zeros(())
                if return_aux:
                    # Dyn aux: world-model state predicts the EGO's (index-0)
                    # future since the rollout is driven by the ego action.
                    y_ego = y[:, 0]                              # (B,K,2)
                    mask = ~torch.isnan(y_ego)
                    if mask.any() and dyn_pred is not None:
                        diff = (dyn_pred - torch.nan_to_num(y_ego)) * mask
                        aux = diff.pow(2).sum() / mask.sum().clamp(min=1)
                out = (z, det, agent_emb)
                if return_aux:
                    out = out + (aux,)
                if return_vel:                                   # (B,N,K,2) learned velocity at GT y
                    out = out + (self.velocity_at(torch.nan_to_num(y), cond),)
                return out
            return z, det, agent_emb

        # ------------------------------------------------------------------
        # SINGLE-TARGET LEGACY PATH (y is (B, K, 2))
        # ------------------------------------------------------------------
        if self.norm_rotation:
            x, y, angle = self._normalize_rotation(x, y)
            feat = self._rotate_features(feat, angle)

        # When the model was trained with scene_level=True only the
        # self-attention encoder path is trained; reuse it here and take the
        # ego (index 0) embedding so single-target eval uses trained weights.
        if self.scene_level:
            agent_emb, _ = self.encoder(
                None, torch.cat([x, feat], dim=-1), v_type, per_agent=True,
            )                                                # (B, N, E)
            embedding = agent_emb[:, 0]                      # (B, E) ego
        else:
            embedding, _ = self.encoder(
                None, torch.cat([x, feat], dim=-1), v_type, per_agent=False,
            )  # (B, E)
        amemb = self._agent_map_emb(x, location_id)            # per-agent local crop (ego = idx 0)
        if amemb is not None:
            embedding = embedding + amemb[:, 0]
        else:
            memb = self._map_emb(location_id, x.device)        # global per-location vector
            if memb is not None:
                embedding = embedding + memb

        dyn = None
        if self.use_world_model and self.world_model is not None:
            if return_aux:
                cond, dyn = self.world_model(
                    embedding, y.shape[1], actions, return_dyn=True
                )
            else:
                cond = self.world_model(embedding, y.shape[1], actions)
        else:
            cond = embedding

        z, det = self.flow(y, cond, sampling_frequency=sampling_frequency)

        if return_aux or return_vel:
            aux = z.new_zeros(())
            if return_aux and dyn is not None:
                mask = ~torch.isnan(y)
                if mask.any():
                    diff = (dyn - torch.nan_to_num(y)) * mask
                    aux = diff.pow(2).sum() / mask.sum().clamp(min=1)
            out = (z, det, embedding)
            if return_aux:
                out = out + (aux,)
            if return_vel:
                out = out + (self.velocity_at(y, cond),)
            return out
        return z, det, embedding

    def sample(self, x, feat, futures, v_type, num_samples=1, sampling_frequency=1, actions=None):
        if self.norm_rotation:
            x, angle = self._normalize_rotation(x)
            feat = self._rotate_features(feat, angle)

        # NOTE: This sampling path assumes batch size == 1 and uses `num_samples` as the flow batch.
        # This matches the previous implementation and current evaluation usage.
        if x.shape[0] != 1:
            raise ValueError(
                f"MultiTrajFlow.sample currently expects batch_size==1, got {x.shape[0]}."
            )
        if self.scene_level:
            agent_emb, _ = self.encoder(
                None, torch.cat([x, feat], dim=-1), v_type, per_agent=True,
            )
            embedding = agent_emb[:, 0]
        else:
            embedding, _ = self.encoder(
                None, torch.cat([x, feat], dim=-1), v_type, per_agent=False,
            )

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
        if actions is not None and actions.shape[0] != y.shape[0]:
            actions = actions.expand(y.shape[0], *actions.shape[1:])
        cond = self._flow_condition(embedding, futures, actions)
        z, det = self.flow(
            y, cond, reverse=True, sampling_frequency=sampling_frequency
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

        if self.scene_level:
            agent_emb, _ = self.encoder(
                None, torch.cat([x, feat], dim=-1), v_type, per_agent=True,
            )
            embedding = agent_emb[:, 0]                       # (B,E) ego
        else:
            embedding, _ = self.encoder(
                None, torch.cat([x, feat], dim=-1), v_type, per_agent=False,
            )  # (B,E)

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
        cond_rep = self._flow_condition(emb_rep, t, None)
        x_flat, det = self.flow(y_flat, cond_rep, reverse=True, sampling_frequency=sampling_frequency)
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
