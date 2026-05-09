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

            # Move data to device
            inputs = inputs.to(device)  # [batch_size, max_num_cars, seq_len, 2]
            features = features.to(
                device
            )  # [batch_size, max_num_cars, seq_len, feature_dim]
            types = types.to(device)  # [batch_size, max_num_cars]
            targets = targets.to(device)  # [batch_size, pred_len, 2]

            # Forward pass
            z_t0, det, embedding = model(inputs, targets, features, types)
            logpz_t0, logpz_t1 = model.log_prob(z_t0, det, embedding=embedding)

            # Per-frame negative log-likelihood (supports time weighting)
            # logpz_t1: (B, T)
            if hasattr(wandb, "run") and wandb.run is not None:
                cfg = wandb.run.config
            else:
                cfg = None

            loss_time_decay = float(getattr(cfg, "loss_time_decay", 0.0)) if cfg is not None else 0.0
            if loss_time_decay > 0.0:
                t = torch.arange(logpz_t1.shape[1], device=logpz_t1.device, dtype=logpz_t1.dtype)
                w = torch.exp(-loss_time_decay * t)
                w = w / (w.mean() + 1e-9)
                nll = -(logpz_t1 * w.unsqueeze(0)).mean()
            else:
                nll = -logpz_t1.mean()

            # Optional centroid smoothness regularization (sample-based)
            centroid_lambda = float(getattr(cfg, "centroid_smooth_lambda", 0.0)) if cfg is not None else 0.0
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

            loss = nll + (centroid_lambda * centroid_pen if centroid_pen is not None else 0.0)

            if verbose:
                print(f"logpz_t0 (latent): {-torch.mean(logpz_t0)}")
                print(f"logpz_t1 (prior): {nll}")
                if centroid_pen is not None:
                    print(f"centroid_smooth_pen: {centroid_pen}")

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
        observation_site = ind.observation_site_08

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
