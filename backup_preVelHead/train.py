import os
import time
import torch
import wandb
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets.InD import InD
from model.RiskFlow import RiskFlow
from riskflow_config import PRESET_OVERRIDES, set_wandb_defaults, seed_everything


def _ego_action_proxy(targets, scale):
    """Ego action proxy = 2nd difference of the ego's future positions (the
    target), scaled. Shape (B,K,2). The ego's realized action is what the
    counterfactual at inference will substitute, so we train with this.
    """
    a = targets[:, 2:] - 2.0 * targets[:, 1:-1] + targets[:, :-2]
    a = torch.cat([a[:, :1], a, a[:, -1:]], dim=1)
    return torch.nan_to_num(a) * float(scale)


def _pick_neighbor(inputs, futures):
    """Pick the nearest valid neighbor per sample for multi-agent training.

    Returns:
        n_idx: (B,)  index in 1..N-1 of the chosen neighbor (0 if none)
        has:   (B,)  bool mask, True if a valid neighbor was found
    """
    B, N = inputs.shape[0], inputs.shape[1]
    if futures is None or N < 2:
        return torch.zeros(B, dtype=torch.long, device=inputs.device), \
               torch.zeros(B, dtype=torch.bool, device=inputs.device)
    ego_last = inputs[:, 0, -1, :]                          # (B,2)
    oth_last = inputs[:, 1:, -1, :]                         # (B,N-1,2)
    valid = ~torch.isnan(futures[:, 1:]).any(dim=(2, 3))     # (B,N-1)
    valid &= ~torch.isnan(oth_last).any(dim=-1)
    dist = (oth_last - ego_last.unsqueeze(1)).norm(dim=-1)
    dist = dist.masked_fill(~valid, float("inf"))
    e_rel = dist.argmin(dim=1)
    n_idx = e_rel + 1
    has = valid.any(dim=1)
    n_idx = torch.where(has, n_idx, torch.zeros_like(n_idx))
    return n_idx, has


def _rotate_agents(x, feat, vt, n_idx):
    """Cyclically permute the agent axis per sample so position `n_idx[b]`
    becomes index 0 for sample b. Index 0 (the chosen neighbor) is now the
    prediction target; the original ego ends up at some other index in the
    rotated ordering (it remains in the context attended to by the encoder).
    """
    B, N = vt.shape
    idx = (n_idx.unsqueeze(1) + torch.arange(N, device=x.device).unsqueeze(0)) % N
    idx_x = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])
    idx_f = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, feat.shape[2], feat.shape[3])
    return x.gather(1, idx_x), feat.gather(1, idx_f), vt.gather(1, idx)


def train(observation_site, model, epochs, lr, weight_decay, gamma, verbose, device):
    model.train()

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=gamma)

    total_loss = []
    for epoch in range(epochs):
        losses = []
        for batch in tqdm(observation_site.train_loader, desc=f"Epoch {epoch}"):
            inputs, features, types, targets = (
                batch["input"],
                batch["feature"],
                batch["type"],
                batch["target"],
            )
            futures = batch.get("future")
            loc_id = batch.get("locationId")
            if loc_id is not None:
                loc_id = loc_id.to(device)

            # Move data to device
            inputs = inputs.to(device)  # [batch_size, max_num_cars, seq_len, 2]
            features = features.to(
                device
            )  # [batch_size, max_num_cars, seq_len, feature_dim]
            types = types.to(device)  # [batch_size, max_num_cars]
            targets = targets.to(device)  # [batch_size, pred_len, 2]
            if futures is not None:
                futures = futures.to(device)

            cfg0 = wandb.run.config if (hasattr(wandb, "run") and wandb.run is not None) else None
            p_act = float(getattr(cfg0, "wm_action_dropout", 0.0)) if cfg0 is not None else 0.0
            a_scale = float(getattr(cfg0, "wm_action_scale", 100.0)) if cfg0 is not None else 100.0
            use_ma = bool(getattr(cfg0, "wm_multi_agent", False)) if cfg0 is not None else False
            scene_level = bool(getattr(model, "scene_level", False))
            use_world_model = bool(getattr(model, "use_world_model", False))

            # Ego action proxy + action-dropout: trains both the action path
            # (counterfactual queries) and the autonomous path (plain eval).
            ego_actions = None
            if use_world_model and p_act > 0.0 and torch.rand(()).item() < p_act:
                ego_actions = _ego_action_proxy(targets, a_scale)

            B = inputs.shape[0]
            if scene_level and futures is not None:
                # ----------------------------------------------------------
                # SCENE-LEVEL AR PATH: predict all valid agents jointly via
                # chain rule; condition on the ego's action proxy.
                # ----------------------------------------------------------
                N = futures.shape[1]
                z_t0, det, agent_emb, aux_dyn = model(
                    inputs, futures, features, types,
                    actions=ego_actions, return_aux=True, location_id=loc_id,
                )                                                # z (B,N,K,2), det (B,N,K)
                # Per-agent log-prob via base normal + |det| (flatten BN for
                # the existing log_prob implementation).
                K = z_t0.shape[2]
                z_flat = z_t0.reshape(B * N, K, z_t0.shape[-1])
                d_flat = det.reshape(B * N, K)
                _, lpx_flat = model.log_prob(z_flat, d_flat, embedding=None)
                logpz_t1 = lpx_flat.view(B, N, K)
                logpz_t0 = logpz_t1                              # not used downstream
                # Per-agent validity mask: non-NaN futures + ego is excluded
                # from the chain loss (its prediction conditioned on its own
                # action is trivial; the dyn aux supervises it instead).
                agent_valid = (~torch.isnan(futures).any(dim=(2, 3)))  # (B, N)
                agent_mask = agent_valid.clone()
                agent_mask[:, 0] = False                          # exclude ego
                # Store for downstream loss computation.
                _scene_mask = agent_mask
            else:
                # ----------------------------------------------------------
                # LEGACY SINGLE-TARGET PATH (or scene-level disabled).
                # ----------------------------------------------------------
                if use_world_model and use_ma and futures is not None:
                    n_idx, has = _pick_neighbor(inputs, futures)
                    inputs_use, features_use, types_use = _rotate_agents(
                        inputs, features, types, n_idx
                    )
                    y_target = futures[torch.arange(B, device=device), n_idx]
                    sample_w = has.float()
                else:
                    inputs_use, features_use, types_use, y_target = (
                        inputs, features, types, targets
                    )
                    sample_w = inputs.new_ones(B)
                z_t0, det, embedding, aux_dyn = model(
                    inputs_use, y_target, features_use, types_use,
                    actions=ego_actions, return_aux=True, location_id=loc_id,
                )
                logpz_t0, logpz_t1 = model.log_prob(z_t0, det, embedding=embedding)
                _scene_mask = None

            # Per-frame negative log-likelihood (supports time weighting)
            # logpz_t1: (B, T)
            if hasattr(wandb, "run") and wandb.run is not None:
                cfg = wandb.run.config
            else:
                cfg = None

            loss_time_decay = float(getattr(cfg, "loss_time_decay", 0.0)) if cfg is not None else 0.0
            if _scene_mask is not None:
                # SCENE-LEVEL: logpz_t1 is (B, N, K); mask is (B, N) excluding
                # ego and invalid agents. Per-agent per-frame mean -> per-agent
                # NLL -> masked mean over agents -> batch mean.
                K_ax = logpz_t1.shape[2]
                if loss_time_decay > 0.0:
                    t = torch.arange(K_ax, device=logpz_t1.device, dtype=logpz_t1.dtype)
                    w = torch.exp(-loss_time_decay * t)
                    w = w / (w.mean() + 1e-9)
                    per_agent = (logpz_t1 * w.view(1, 1, K_ax)).mean(dim=2)  # (B,N)
                else:
                    per_agent = logpz_t1.mean(dim=2)                          # (B,N)
                mask_f = _scene_mask.to(per_agent.dtype)
                nll = -(per_agent * mask_f).sum() / mask_f.sum().clamp(min=1.0)
            else:
                if loss_time_decay > 0.0:
                    t = torch.arange(logpz_t1.shape[1], device=logpz_t1.device, dtype=logpz_t1.dtype)
                    w = torch.exp(-loss_time_decay * t)
                    w = w / (w.mean() + 1e-9)
                    per_sample = (logpz_t1 * w.unsqueeze(0)).mean(dim=1)  # (B,)
                else:
                    per_sample = logpz_t1.mean(dim=1)                    # (B,)
                nll = -(per_sample * sample_w).sum() / sample_w.sum().clamp(min=1.0)

            # Optional centroid smoothness regularization (sample-based).
            # Disabled in scene-level mode: rsample is single-target and would
            # not exercise the AR scene-level conditioning path.
            centroid_lambda = float(getattr(cfg, "centroid_smooth_lambda", 0.0)) if cfg is not None else 0.0
            if scene_level:
                centroid_lambda = 0.0
            centroid_pen = None
            if centroid_lambda > 0.0:
                if getattr(model, "use_cnf", False):
                    raise RuntimeError(
                        "centroid_smooth_lambda>0 requires use_cnf=False (CNF reverse pass is too slow for training-time regularization)."
                    )
                centroid_samples = int(getattr(cfg, "centroid_samples", 2))
                centroid_alpha = float(getattr(cfg, "centroid_alpha", 10.0))

                x_samp, logp_samp = model.rsample_x_and_logp(
                    inputs,
                    features,
                    types,
                    futures=targets.shape[1],
                    num_samples=max(1, centroid_samples),
                )
                # x_samp: (B,S,T,2), logp_samp: (B,S,T)
                # Soft "top" centroid across samples per frame: weights = softmax(alpha*logp)
                w_s = F.softmax(centroid_alpha * logp_samp, dim=1)
                centroid = (w_s.unsqueeze(-1) * x_samp).sum(dim=1)  # (B,T,2)

                # Smoothness: penalize acceleration magnitude ||c_{t+1}-2c_t+c_{t-1}||^2
                vel = centroid[:, 1:] - centroid[:, :-1]  # (B,T-1,2)
                acc = vel[:, 1:] - vel[:, :-1]  # (B,T-2,2)
                acc2 = (acc ** 2).sum(dim=-1)  # (B,T-2)

                if loss_time_decay > 0.0 and acc2.shape[1] > 0:
                    # Align weights to acceleration time index (t=1..T-2 roughly)
                    w_acc = w[1:-1]
                    w_acc = w_acc / (w_acc.mean() + 1e-9)
                    centroid_pen = (acc2 * w_acc.unsqueeze(0)).mean()
                else:
                    centroid_pen = acc2.mean()

            wm_dyn_lambda = float(getattr(cfg, "wm_dyn_lambda", 0.0)) if cfg is not None else 0.0
            aux_term = wm_dyn_lambda * aux_dyn if wm_dyn_lambda > 0.0 else 0.0

            loss = (
                nll
                + (centroid_lambda * centroid_pen if centroid_pen is not None else 0.0)
                + aux_term
            )

            if verbose:
                print(f"logpz_t0 (latent): {-torch.mean(logpz_t0)}")
                print(f"logpz_t1 (prior): {nll}")
                if centroid_pen is not None:
                    print(f"centroid_smooth_pen: {centroid_pen}")
                print(f"wm_dyn_aux: {float(aux_dyn.detach().item())}")

            # Backward pass
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()

            if cfg is not None:
                wandb.log(
                    {
                        "loss/nll": float(nll.detach().item()),
                        "loss/total": float(loss.detach().item()),
                        "loss_time_decay": loss_time_decay,
                        "centroid_smooth_lambda": centroid_lambda,
                        "wm_dyn_lambda": wm_dyn_lambda,
                        "loss/wm_dyn_aux": float(aux_dyn.detach().item()),
                        **({"loss/centroid_smooth": float(centroid_pen.detach().item())} if centroid_pen is not None else {}),
                    }
                )

            if verbose:
                total_loss.append(loss.item())
            losses.append(loss)

        # Calculate epoch loss
        losses = torch.stack(losses)
        epoch_loss = torch.mean(torch.mean(losses))
        if not verbose:
            total_loss.append(epoch_loss.item())
        print(f"epoch: {epoch}, loss: {epoch_loss:.4f}")

    # Visualize loss if verbose
    if verbose:
        loss_visual = "loss.png"

        if os.path.exists(loss_visual):
            os.remove(loss_visual)

        plt.plot(total_loss)
        plt.savefig(loss_visual)
        plt.close()

    return total_loss


if __name__ == "__main__":
    with wandb.init() as run:
        # Centralized defaults (one place to edit): riskflow_config.py
        set_wandb_defaults(run, overrides=PRESET_OVERRIDES["train"])
        seed_everything(run.config.seed)

        # Dataset and observation site
        ind = InD(
            root="data",
            max_samples=run.config.maximum_samples,
            train_ratio=run.config.train_ratio,
            train_batch_size=run.config.train_batch_size,
            test_batch_size=run.config.test_batch_size,
            missing_rate=run.config.masked_data_ratio,
            max_num_cars=run.config.max_num_cars,
            max_empty_frames=run.config.max_empty_frames,
            seq_len=run.config.seq_len,
            moving_window=run.config.seq_len * 2,
            sampling_step=run.config.sampling_step,
            should_shuffle=run.config.should_shuffle,
            include_future=run.config.include_future,
        )
        observation_site = ind.observation_site_by_scope(
            getattr(run.config, "site_scope", "08")
        )

        # Initialize model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = RiskFlow(
            seq_len=run.config.seq_len,
            input_dim=run.config.input_dim,
            feature_dim=run.config.feature_dim,
            embedding_dim=run.config.embedding_dim,
            hidden_dim=run.config.hidden_dim,
            max_num_cars=run.config.max_num_cars,
            num_classes=run.config.num_classes,
            gru_layers=run.config.gru_layers,
            num_heads=run.config.num_heads,
            dropout=run.config.dropout,
            norm_rotation=run.config.norm_rotate,
            flow_layers=run.config.flow_layers,
            flow_hidden_dim=run.config.flow_hidden_dim,
            coupling_layers=run.config.coupling_layers,
            use_cnf=run.config.use_cnf,
            use_cgmm=run.config.use_cgmm,
            gmm_modes=run.config.gmm_modes,
            use_world_model=getattr(run.config, "use_world_model", False),
            wm_state_dim=getattr(run.config, "wm_state_dim", 256),
            action_dim=getattr(run.config, "action_dim", 2),
            scene_level=getattr(run.config, "scene_level", False),
            agent_ordering=getattr(run.config, "agent_ordering", "nearest_ego"),
        ).to(device)

        # Log model parameters
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_parameters}")
        wandb.log({"parameters": num_parameters})

        # Train the model
        train_start_time = time.time()
        total_loss = train(
            observation_site=observation_site,
            model=model,
            epochs=run.config.training_epochs,
            lr=run.config.lr,
            weight_decay=run.config.weight_decay,
            gamma=run.config.gamma,
            verbose=run.config.verbose,
            device=device,
        )
        train_end_time = time.time()

        # Log training runtime
        train_runtime = train_end_time - train_start_time
        print(f"Training runtime: {train_runtime:.2f} seconds")
        wandb.log({"train runtime": train_runtime})

        # Log training loss
        for loss in total_loss:
            wandb.log({"loss": loss})

        # Save the model
        model_name = "risk_flow.pt"
        torch.save(model.state_dict(), model_name)
        print(f"Model saved as {model_name}")
