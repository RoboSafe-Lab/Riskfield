import argparse
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from datasets.InD import InD


def makedir(directory: str) -> None:
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def _denormalize_xy(xy_norm: torch.Tensor) -> torch.Tensor:
    from datasets.InD import spatial_boundaries

    bounds = torch.as_tensor(spatial_boundaries, dtype=xy_norm.dtype, device=xy_norm.device)
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    return xy_norm * (hi - lo) + lo


def _is_invalid_xy(xy: np.ndarray) -> bool:
    if xy is None:
        return True
    if np.isnan(xy).any():
        return True
    if float(xy[0]) == 0.0 and float(xy[1]) == 0.0:
        return True
    return False


def compute_px(
    model,
    inputs: torch.Tensor,
    features: torch.Tensor,
    types: torch.Tensor,
    grid: torch.Tensor,
    *,
    frame_indices: Optional[list[int]] = None,
    batch_size: int = 512,
) -> torch.Tensor:
    """Compute normalized likelihood px over `grid`.

    Returns px: (num_points, len(frame_indices) or model.seq_len)
    """
    device = inputs.device
    model.eval()

    with torch.no_grad():
        cond = torch.cat([inputs, features], dim=-1)
        embedding = model.encoder(None, cond, types)
        emb_single = embedding[0:1].to(device)

        px_parts = []
        for grid_batch in grid.split(batch_size, dim=0):
            b = grid_batch.shape[0]
            frame_indices_tensor = None

            if frame_indices is None:
                flattened_grid = grid_batch.unsqueeze(1).expand(-1, model.seq_len, -1).to(device)
            else:
                frame_indices_tensor = torch.as_tensor(frame_indices, device=device)
                if frame_indices_tensor.dim() == 1:
                    frame_indices_tensor = frame_indices_tensor.unsqueeze(0).expand(b, -1)
                flattened_grid = grid_batch.unsqueeze(1).expand(-1, len(frame_indices), -1).to(device)

            emb_rep = emb_single.expand(b, emb_single.shape[1]).to(device)
            z, det = model.flow(flattened_grid, emb_rep, frame_indices=frame_indices_tensor)
            _, logpx = model.log_prob(z, det, embedding=emb_rep)
            px_parts.append(logpx.exp().cpu())

        px = torch.cat(px_parts, dim=0)
        px_max = px.max(dim=0, keepdim=True).values
        px = px / (px_max + 1e-9)
        return px


def get_px_centroid(px: torch.Tensor, grid_xy_m: torch.Tensor, top_k: int = 30) -> torch.Tensor:
    """Centroid of top_k probability points, per frame.

    px: (num_points, T)
    grid_xy_m: (num_points,2) physical coords
    returns: (T,2) physical coords
    """
    device = px.device
    grid_xy_m = grid_xy_m.to(device)

    T = px.shape[1]
    centroids = []
    for t in range(T):
        prob_t = px[:, t]
        _, idxs = torch.topk(prob_t, k=min(top_k, prob_t.shape[0]))
        centroids.append(grid_xy_m[idxs].mean(dim=0))
    return torch.stack(centroids, dim=0)


def _smooth_traj(traj: torch.Tensor, window_size: int = 5) -> torch.Tensor:
    """Simple moving-average smoothing with replicate padding."""
    if traj.dim() != 2 or traj.shape[1] != 2:
        raise ValueError("traj must be (T,2)")
    if traj.shape[0] < 3:
        return traj

    pad = window_size // 2
    traj_t = traj.t().unsqueeze(0)  # (1,2,T)
    traj_p = torch.nn.functional.pad(traj_t, (pad, pad), mode="replicate")
    sm = torch.nn.functional.avg_pool1d(traj_p, kernel_size=window_size, stride=1)
    return sm.squeeze(0).t()


def _vel_from_positions(future_xy_m: torch.Tensor, *, dt: float, last_xy_m: Optional[torch.Tensor]) -> torch.Tensor:
    """Compute per-step velocities aligned with future_xy_m rows."""
    if future_xy_m.dim() != 2 or future_xy_m.shape[1] != 2:
        raise ValueError("future_xy_m must be (T,2)")
    T = future_xy_m.shape[0]
    if T == 0:
        raise ValueError("T must be >0")

    if T == 1:
        if last_xy_m is None:
            return torch.zeros((1, 2), dtype=future_xy_m.dtype, device=future_xy_m.device)
        return (future_xy_m[0:1] - last_xy_m.view(1, 2)) / float(dt)

    v = (future_xy_m[1:] - future_xy_m[:-1]) / float(dt)  # (T-1,2)
    v_last = v[-1:].clone()

    if last_xy_m is not None:
        v0 = (future_xy_m[0:1] - last_xy_m.view(1, 2)) / float(dt)
        v = torch.cat([v0, v], dim=0)  # (T,2)
    else:
        v = torch.cat([v, v_last], dim=0)  # (T,2)

    if v.shape[0] != T:
        v = torch.cat([v, v_last], dim=0)
    return v


def _rankdata_ordinal(x: np.ndarray) -> np.ndarray:
    """Fast ordinal ranks [0..n-1]. Ties are broken arbitrarily (rare for floats)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(order.size, dtype=np.float64)
    return ranks


def spearmanr_fast(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return float("nan")

    rx = _rankdata_ordinal(x)
    ry = _rankdata_ordinal(y)

    rx = rx - rx.mean()
    ry = ry - ry.mean()

    denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def build_objective_risk(
    *,
    grid_xy_m: np.ndarray,
    ego_v_gt: np.ndarray,  # (T,2)
    future_xy_m: np.ndarray,  # (N,T,2)
    last_obs_xy_m: np.ndarray,  # (N,2)
    dt: float,
    sigma: float,
    objective: str = "gaussian_relv",
    eps: float = 1e-6,
    ttc_max_s: float = 5.0,
    ttc_tau_s: float = 2.0,
    ego_index: int = 0,
) -> np.ndarray:
    """Objective risk map using ground-truth env positions/velocities.

    Produces a risk value at each grid point x for each time step t.

    Supported `objective` modes:
    - gaussian_relv: sum_i exp(-||x-pos_i||^2/(2*sigma^2)) * ||v_ego(t)-v_i(t)||^2
    - gaussian_dist: sum_i exp(-||x-pos_i||^2/(2*sigma^2))
    - inv_dist:      sum_i 1/(||x-pos_i|| + eps)
    - min_dist:      1/(min_i ||x-pos_i|| + eps)
    - ttc:           sum_i exp(-ttc/tau) * exp(-dca^2/(2*sigma^2)) for approaching pairs

    Returns:
        risk_obj: (T, num_points)
    """
    obj = (objective or "gaussian_relv").lower().strip()
    grid = torch.as_tensor(grid_xy_m, dtype=torch.float32)  # (P,2)
    N, T, _ = future_xy_m.shape

    neigh_ids = [i for i in range(N) if i != ego_index]
    if len(neigh_ids) == 0:
        return np.zeros((T, grid.shape[0]), dtype=np.float32)

    pos_all = torch.as_tensor(future_xy_m[neigh_ids], dtype=torch.float32)  # (M,T,2)
    valid_all = torch.isfinite(pos_all).all(dim=-1) & ~(
        (pos_all[..., 0] == 0.0) & (pos_all[..., 1] == 0.0)
    )  # (M,T)

    # GT velocities for neighbors (M,T,2)
    vel_all = torch.zeros_like(pos_all)
    for j, car_id in enumerate(neigh_ids):
        last_i = last_obs_xy_m[car_id]
        last_i_t = None if _is_invalid_xy(last_i) else torch.as_tensor(last_i, dtype=torch.float32)
        vel_all[j] = _vel_from_positions(
            pos_all[j],
            dt=float(dt),
            last_xy_m=last_i_t,
        )

    ego_v = torch.as_tensor(ego_v_gt, dtype=torch.float32)  # (T,2)

    risk = torch.zeros((T, grid.shape[0]), dtype=torch.float32)
    denom = 2.0 * float(sigma) * float(sigma)
    eps_f = float(eps)

    for t in range(T):
        mask = valid_all[:, t]
        if not bool(mask.any()):
            continue

        pos_t = pos_all[mask, t, :]  # (K,2)
        vel_t = vel_all[mask, t, :]  # (K,2)

        # (P,K,2)
        diff = grid[:, None, :] - pos_t[None, :, :]
        d2 = torch.sum(diff * diff, dim=-1)  # (P,K)
        d = torch.sqrt(d2 + eps_f)  # (P,K)

        if obj == "gaussian_relv":
            density = torch.exp(-d2 / denom)  # (P,K)
            dv = ego_v[t : t + 1, :] - vel_t  # (1,2)-(K,2) -> (K,2)
            cost = torch.sum(dv * dv, dim=-1).view(1, -1)  # (1,K)
            risk[t] = torch.sum(density * cost, dim=1)

        elif obj == "gaussian_dist":
            density = torch.exp(-d2 / denom)
            risk[t] = torch.sum(density, dim=1)

        elif obj == "inv_dist":
            risk[t] = torch.sum(1.0 / (d + eps_f), dim=1)

        elif obj == "min_dist":
            min_d = torch.amin(d, dim=1)
            risk[t] = 1.0 / (min_d + eps_f)

        elif obj == "ttc":
            # Relative motion in ego frame: neighbor relative velocity.
            v_rel = vel_t - ego_v[t : t + 1, :]  # (K,2)
            v2 = torch.sum(v_rel * v_rel, dim=-1).view(1, -1) + eps_f  # (1,K)
            # r = pos - grid (note sign compared to diff)
            r = -diff  # (P,K,2)
            dot = torch.sum(r * v_rel.view(1, -1, 2), dim=-1)  # (P,K)
            ttc = -dot / v2  # (P,K)
            approaching = (dot < 0.0) & (ttc > 0.0) & (ttc < float(ttc_max_s))

            # distance at closest approach
            r_ca = r + v_rel.view(1, -1, 2) * ttc.unsqueeze(-1)  # (P,K,2)
            dca2 = torch.sum(r_ca * r_ca, dim=-1)  # (P,K)

            w_ttc = torch.exp(-ttc / float(ttc_tau_s))
            w_dca = torch.exp(-dca2 / denom)
            contrib = w_ttc * w_dca
            contrib = torch.where(approaching, contrib, torch.zeros_like(contrib))
            risk[t] = torch.sum(contrib, dim=1)

        else:
            raise ValueError(
                f"Unknown objective='{objective}'. "
                "Use: gaussian_relv | gaussian_dist | inv_dist | min_dist | ttc"
            )

    return risk.numpy()


def run_experiment_for_segment(
    *,
    observation_site,
    model,
    segment_index: int,
    test_data: Optional[list[dict]] = None,
    steps: int,
    device: str,
    cost_step: int,
    sigma: float,
    objective: str,
    eps: float,
    ttc_max_s: float,
    ttc_tau_s: float,
    cache_dir: Optional[str],
    ego_index: int = 0,
) -> dict:
    """Compute model/objective risk and per-t Spearman correlations for one segment.

    If `test_data` is provided, it should be a materialized list(iter(test_loader))
    to avoid repeating IO/work.
    """

    model.eval()

    if test_data is None:
        test_data = list(iter(observation_site.test_loader))
    if segment_index < 0 or segment_index >= len(test_data):
        raise IndexError(f"segment_index {segment_index} out of range (0..{len(test_data)-1})")

    batch = test_data[segment_index]
    return run_experiment_for_batch(
        observation_site=observation_site,
        model=model,
        batch=batch,
        segment_index=segment_index,
        steps=steps,
        device=device,
        cost_step=cost_step,
        sigma=sigma,
        objective=objective,
        eps=eps,
        ttc_max_s=ttc_max_s,
        ttc_tau_s=ttc_tau_s,
        cache_dir=cache_dir,
        ego_index=ego_index,
    )


def run_experiment_for_batch(
    *,
    observation_site,
    model,
    batch: dict,
    segment_index: int,
    steps: int,
    device: str,
    cost_step: int,
    sigma: float,
    objective: str,
    eps: float,
    ttc_max_s: float,
    ttc_tau_s: float,
    cache_dir: Optional[str],
    ego_index: int = 0,
    radius: int = 0,
) -> dict:
    """Compute model/objective risk and per-t Spearman correlations for a provided batch."""

    model.eval()

    cache_dir_final = cache_dir
    if cache_dir_final is None:
        cache_dir_final = "videos/cnf_cache" if getattr(model, "use_cnf", False) else "videos/dnf_cache"
    makedir(cache_dir_final)

    input_batch = batch["input"].to(device)  # (B,N,H,2)
    feature_batch = batch["feature"].to(device)
    type_batch = batch["type"].to(device)
    target_batch = batch["target"].to(device)  # (B,T,2)

    future_batch = batch.get("future")
    if future_batch is None:
        raise KeyError("Batch missing 'future'. Create InD(..., include_future=True) to run objective risk.")

    car_mask = batch.get("carMask")

    track_id = batch["trackId"]
    start_frame = batch["startFrame"]
    s_frame = start_frame[0].item() if torch.is_tensor(start_frame) else int(start_frame[0])
    ego_id = track_id[0].item() if torch.is_tensor(track_id) else int(track_id[0])

    B, N, H, _ = input_batch.shape

    # Grid
    lin = torch.linspace(0, 1, steps)
    xg, yg = torch.meshgrid(lin, lin, indexing="ij")
    grid_norm = torch.stack((xg.flatten(), yg.flatten()), dim=-1).to(device)
    grid_xy_m = observation_site.denormalize(grid_norm.cpu().numpy())
    grid_xy_m_t = torch.as_tensor(grid_xy_m, dtype=torch.float32)

    # Ego GT velocity (T,2)
    dt = 0.04 * float(cost_step)
    ego_future_xy_m = _denormalize_xy(target_batch[0]).detach().cpu()
    ego_last_obs_xy_m = _denormalize_xy(input_batch[0, ego_index, -1]).detach().cpu()
    ego_v_gt = _vel_from_positions(ego_future_xy_m, dt=dt, last_xy_m=ego_last_obs_xy_m).cpu().numpy()

    # Last observed positions for all cars (for objective v0)
    last_obs_xy_m = _denormalize_xy(input_batch[0, :, -1]).detach().cpu().numpy()  # (N,2)

    # Future positions for all cars (objective)
    future_xy_m = _denormalize_xy(future_batch[0].to(device)).detach().cpu().numpy()  # (N,T,2)

    # Apply carMask if provided
    if car_mask is not None:
        m = car_mask[0].detach().cpu().numpy().astype(bool)  # (N,)
        for i in range(N):
            if not m[i]:
                future_xy_m[i, :, :] = np.nan

    T = ego_future_xy_m.shape[0]

    def _apply_px_radius(px_i: torch.Tensor) -> torch.Tensor:
        """Optional spatial smoothing of px over the (steps x steps) grid.

        px_i: (P, T) where P=steps*steps.
        radius: integer in grid cells (0 = no-op).
        """
        r = int(radius)
        if r <= 0:
            return px_i

        P_local, T_local = px_i.shape
        if steps * steps != int(P_local):
            raise ValueError(
                f"radius smoothing requires P==steps^2, got P={P_local}, steps={steps}"
            )

        kernel_size = 2 * r + 1
        yy, xx = torch.meshgrid(
            torch.arange(kernel_size),
            torch.arange(kernel_size),
            indexing="ij",
        )
        dist = torch.sqrt((xx - r) ** 2 + (yy - r) ** 2)
        kernel = (dist <= r).float()
        kernel = kernel / (kernel.sum() + 1e-12)
        kernel = kernel.view(1, 1, kernel_size, kernel_size)

        # (P,T) -> (T,1,steps,steps)
        px_2d = px_i.t().reshape(T_local, 1, steps, steps).to(torch.float32)
        px_2d = F.conv2d(px_2d, kernel, padding=r)
        return px_2d.reshape(T_local, P_local).t()

    # Compute px per car and env velocities estimated from centroid motion
    px_by_car: list[torch.Tensor] = []
    env_v_by_car: list[np.ndarray] = []  # each (T,2)
    valid_env_car_indices: list[int] = []
    px_env_by_car_idx: dict[int, np.ndarray] = {}

    for car_idx in range(N):
        car_data = input_batch[0, car_idx].detach().cpu().numpy()
        if np.all(car_data == 0) or np.isnan(car_data).all():
            continue

        cache_name = f"px_ego{ego_id}_idx{car_idx}_start{s_frame}_steps{steps}.pt"
        cache_path = os.path.join(cache_dir_final, cache_name)

        if os.path.exists(cache_path):
            px_i = torch.load(cache_path, map_location="cpu", weights_only=True)
        else:
            perm = [car_idx] + [j for j in range(N) if j != car_idx]
            t0 = time.time()
            px_i = compute_px(
                model,
                input_batch[:, perm],
                feature_batch[:, perm],
                type_batch[:, perm],
                grid_norm,
                frame_indices=None,
            )
            t1 = time.time()
            print(f"segment={segment_index} car={car_idx} compute_px took {t1-t0:.2f}s")
            torch.save(px_i, cache_path)

        px_i = _apply_px_radius(px_i)

        # Align T just in case
        if px_i.shape[1] != T:
            raise ValueError(f"px horizon {px_i.shape[1]} != target horizon {T}")

        px_by_car.append(px_i)

        if car_idx == ego_index:
            continue

        px_env_by_car_idx[int(car_idx)] = px_i.detach().cpu().numpy().astype(np.float32)

        # Estimate env v(t) from centroid motion of px
        ctr = get_px_centroid(px_i.to(torch.float32), grid_xy_m_t, top_k=30)
        ctr = _smooth_traj(ctr, window_size=5)

        if ctr.shape[0] < 2:
            v_env = torch.zeros_like(ctr)
        else:
            v_env = (ctr[1:] - ctr[:-1]) / float(dt)
            v_env = torch.cat([v_env, v_env[-1:].clone()], dim=0)
        env_v_by_car.append(v_env.cpu().numpy())
        valid_env_car_indices.append(car_idx)

    if len(env_v_by_car) == 0:
        raise RuntimeError("No valid environment cars found in this segment")

    # Build model risk: (T,P)
    P = grid_xy_m.shape[0]
    risk_model = np.zeros((T, P), dtype=np.float32)

    env_iter = 0
    # Robust mapping: reuse cached px tensors.
    for car_idx in valid_env_car_indices:
        px_i = px_env_by_car_idx[int(car_idx)]  # (P,T)
        v_env = env_v_by_car[env_iter]  # (T,2)
        env_iter += 1

        dv = ego_v_gt - v_env
        cost_t = np.sum(dv * dv, axis=1).astype(np.float32)  # (T,)
        risk_model += (px_i.T * cost_t[:, None])

    # Objective risk
    risk_obj = build_objective_risk(
        grid_xy_m=grid_xy_m,
        ego_v_gt=ego_v_gt,
        future_xy_m=future_xy_m,
        last_obs_xy_m=last_obs_xy_m,
        dt=dt,
        sigma=float(sigma),
        objective=str(objective),
        eps=float(eps),
        ttc_max_s=float(ttc_max_s),
        ttc_tau_s=float(ttc_tau_s),
        ego_index=ego_index,
    )  # (T,P)

    # Correlations per t
    corr = np.array([
        spearmanr_fast(risk_model[t].reshape(-1), risk_obj[t].reshape(-1)) for t in range(T)
    ], dtype=np.float64)

    return {
        "segment": int(segment_index),
        "T": int(T),
        "steps": int(steps),
        "spearman_per_t": corr,
        "spearman_mean": float(np.nanmean(corr)),
        "risk_model": risk_model.reshape(T, steps, steps),
        "risk_objective": risk_obj.reshape(T, steps, steps),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Risk ranking experiment: compute model risk field and GT-based objective risk field, "
            "then evaluate per-t monotonic correlation (Spearman) between them."
        )
    )
    p.add_argument("--indices", type=int, nargs="+", required=True, help="Segment indices, e.g. --indices 363 1072")
    p.add_argument("--steps", type=int, default=64, help="Grid resolution per axis (use smaller for speed)")
    p.add_argument("--site", type=str, default="08", help="InD site id, e.g. 08")
    p.add_argument("--model", type=str, required=True, help="Path to model .pt (state_dict)")
    p.add_argument("--device", type=str, default=None, help="cpu | cuda | auto")
    p.add_argument("--cost-step", type=int, default=2, help="dt = 0.04 * cost_step")
    p.add_argument("--sigma", type=float, default=1.5, help="Gaussian sigma (meters) for objective density")
    p.add_argument(
        "--objective",
        type=str,
        default="gaussian_relv",
        choices=["gaussian_relv", "gaussian_dist", "inv_dist", "min_dist", "ttc"],
        help="Objective risk definition",
    )
    p.add_argument("--eps", type=float, default=1e-6, help="Numerical epsilon")
    p.add_argument("--ttc-max", type=float, default=5.0, help="Max TTC seconds for objective=ttc")
    p.add_argument("--ttc-tau", type=float, default=2.0, help="TTC decay time constant (s) for objective=ttc")
    p.add_argument("--cache-dir", type=str, default=None, help="Cache dir for px tensors")
    p.add_argument("--out", type=str, default="videos/risk_ranking", help="Output directory")
    p.add_argument("--save-fields", action="store_true", help="Save risk tensors (.npz) for inspection")
    return p


def main() -> None:
    from model.RiskFlow import RiskFlow
    from riskflow_config import preset, seed_everything

    args = build_arg_parser().parse_args()

    cfg = preset("vis_field")
    seed_everything(cfg["seed"])

    if args.device is None or args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    ind = InD(
        root="data",
        max_samples=cfg["maximum_samples"],
        train_ratio=cfg["train_ratio"],
        train_batch_size=cfg["train_batch_size"],
        test_batch_size=cfg["test_batch_size"],
        missing_rate=cfg["masked_data_ratio"],
        max_num_cars=cfg["max_num_cars"],
        max_empty_frames=cfg["max_empty_frames"],
        seq_len=cfg["seq_len"],
        moving_window=cfg["seq_len"] * 2,
        sampling_step=cfg["sampling_step"],
        should_shuffle=cfg["should_shuffle"],
        include_future=True,
    )

    site_id = str(args.site)
    obs = getattr(ind, f"observation_site_{site_id}", None)
    if obs is None:
        obs = ind._get_observation_site([site_id])
    observation_site = obs

    traj_flow = RiskFlow(
        seq_len=cfg["seq_len"],
        input_dim=cfg["input_dim"],
        feature_dim=cfg["feature_dim"],
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        max_num_cars=cfg["max_num_cars"],
        num_classes=cfg["num_classes"],
        gru_layers=cfg["gru_layers"],
        num_heads=cfg["num_heads"],
        dropout=cfg["dropout"],
        norm_rotation=cfg["norm_rotate"],
        flow_layers=cfg["flow_layers"],
        flow_hidden_dim=cfg["flow_hidden_dim"],
        coupling_layers=cfg["coupling_layers"],
        use_cnf=cfg["use_cnf"],
        use_cgmm=cfg["use_cgmm"],
        gmm_modes=cfg["gmm_modes"],
    ).to(device)

    model_path = args.model
    if not os.path.exists(model_path):
        candidates: list[str] = []
        # If user passed just a filename, try common locations.
        base = os.path.basename(model_path)
        candidates.append(os.path.join("serialized", base))
        # If extension omitted, try adding .pt
        if not base.lower().endswith(".pt"):
            candidates.append(os.path.join("serialized", base + ".pt"))
            candidates.append(base + ".pt")
        # Also try serialized/<original> (in case original had subpath-like string)
        candidates.append(os.path.join("serialized", model_path))

        found = None
        for c in candidates:
            if os.path.exists(c):
                found = c
                break
        if found is None:
            cand_msg = "\n".join([f"  - {c}" for c in candidates])
            raise FileNotFoundError(
                f"Model file not found: '{model_path}'. Tried:\n{cand_msg}\n"
                "Tip: pass '--model serialized/multi_trajflow_ind_2.pt'"
            )
        model_path = found

    traj_flow.load_state_dict(torch.load(model_path, map_location=device))
    traj_flow.eval()

    out_dir = args.out
    makedir(out_dir)

    all_rows = []
    for seg in args.indices:
        res = run_experiment_for_segment(
            observation_site=observation_site,
            model=traj_flow,
            segment_index=int(seg),
            steps=int(args.steps),
            device=device,
            cost_step=int(args.cost_step),
            sigma=float(args.sigma),
            objective=str(args.objective),
            eps=float(args.eps),
            ttc_max_s=float(args.ttc_max),
            ttc_tau_s=float(args.ttc_tau),
            cache_dir=args.cache_dir,
            ego_index=0,
        )

        corr = res["spearman_per_t"]
        print(f"segment={seg} spearman_mean={res['spearman_mean']:.4f}  (T={res['T']})")

        all_rows.append((int(seg), float(res["spearman_mean"])))

        # Save per-t correlations and optionally fields
        np.savez(
            os.path.join(out_dir, f"seg{int(seg)}_spearman.npz"),
            segment=int(seg),
            spearman_per_t=corr,
            spearman_mean=float(res["spearman_mean"]),
            steps=int(args.steps),
            sigma=float(args.sigma),
            cost_step=int(args.cost_step),
            objective=str(args.objective),
            eps=float(args.eps),
            ttc_max_s=float(args.ttc_max),
            ttc_tau_s=float(args.ttc_tau),
        )

        if args.save_fields:
            np.savez_compressed(
                os.path.join(out_dir, f"seg{int(seg)}_fields_steps{int(args.steps)}.npz"),
                risk_model=res["risk_model"],
                risk_objective=res["risk_objective"],
            )

    if len(all_rows) > 1:
        np.savez(
            os.path.join(out_dir, "summary.npz"),
            segments=np.array([r[0] for r in all_rows], dtype=np.int64),
            spearman_mean=np.array([r[1] for r in all_rows], dtype=np.float64),
        )


if __name__ == "__main__":
    main()
