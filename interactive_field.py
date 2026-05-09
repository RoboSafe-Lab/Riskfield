import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from datasets.InD import InD


def makedir(directory: str) -> None:
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def _frame_indices_cache_tag(frame_indices: list[int]) -> str:
    if frame_indices is None:
        return ""
    if len(frame_indices) == 0:
        return "_fEMPTY"
    fi = [int(x) for x in frame_indices]
    is_contiguous = all((fi[i] - fi[i - 1]) == 1 for i in range(1, len(fi)))
    if is_contiguous:
        return f"_f{fi[0]}-{fi[-1]}"
    import hashlib

    h = hashlib.sha1(",".join(map(str, fi)).encode("utf-8")).hexdigest()[:8]
    return f"_fH{h}_n{len(fi)}"


def _denormalize_xy(xy_norm: torch.Tensor) -> torch.Tensor:
    from datasets.InD import spatial_boundaries

    bounds = torch.as_tensor(spatial_boundaries, dtype=xy_norm.dtype, device=xy_norm.device)
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    return xy_norm * (hi - lo) + lo


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

        # Normalize per frame for visualization
        px_max = px.max(dim=0, keepdim=True).values
        px = px / (px_max + 1e-9)
        return px


def get_px_centroid(px: torch.Tensor, grid: torch.Tensor, top_k: int = 30) -> np.ndarray:
    """Centroid of top_k probability points, per frame."""
    device = px.device
    grid = grid.to(device)
    seq_len = px.shape[1]
    centroids = []
    for t in range(seq_len):
        prob_t = px[:, t]
        vals, idxs = torch.topk(prob_t, k=min(top_k, prob_t.shape[0]))
        top_points = grid[idxs]
        centroids.append(top_points.mean(dim=0))
    return torch.stack(centroids, dim=0).cpu().numpy()


def _select_frame_window(t: int, horizon: int) -> list[int]:
    """Choose a small set of frame indices to estimate env velocity at frame t."""
    t = int(t)
    horizon = int(horizon)
    if horizon <= 1:
        return [0]
    if t < 0 or t >= horizon:
        raise ValueError(f"t must be in [0,{horizon-1}] but got {t}")
    if t < horizon - 1:
        return [t, t + 1]
    if t > 0:
        return [t - 1, t]
    return [t]


@dataclass
class PreparedFrame:
    background_path: str
    ortho_px_to_meter: float
    x: np.ndarray  # (steps,steps) plot coords
    y: np.ndarray  # (steps,steps) plot coords
    grid_xy_m: np.ndarray  # (num_points,2) physical coords
    density_sum: np.ndarray  # (num_points,)
    env_velocities: list[np.ndarray]  # each (2,) physical m/s
    env_densities: list[np.ndarray]  # each (num_points,)


def prepare_environment_for_frame(
    *,
    observation_site,
    model,
    segment_index: int,
    t: int,
    steps: int,
    device: str,
    ego_index: int = 0,
    cost_step: int = 2,
    cache_root: Optional[str] = None,
) -> PreparedFrame:
    """Precompute per-car density at frame t and estimated env velocities."""

    model.eval()

    # Match visualize_field's scaling
    fudge_factor = 11.5
    ortho_px_to_meter = observation_site.ortho_px_to_meter * fudge_factor

    cache_dir = cache_root
    if cache_dir is None:
        cache_dir = "videos/cnf_cache" if getattr(model, "use_cnf", False) else "videos/dnf_cache"
    makedir(cache_dir)

    test_data = list(iter(observation_site.test_loader))
    if segment_index < 0 or segment_index >= len(test_data):
        raise IndexError(f"segment_index {segment_index} out of range (0..{len(test_data)-1})")

    batch = test_data[segment_index]
    input_batch = batch["input"].to(device)
    feature_batch = batch["feature"].to(device)
    type_batch = batch["type"].to(device)
    track_id = batch["trackId"]
    start_frame = batch["startFrame"]

    s_frame = start_frame[0].item() if torch.is_tensor(start_frame) else int(start_frame[0])
    ego_id = track_id[0].item() if torch.is_tensor(track_id) else int(track_id[0])

    num_cars = input_batch.shape[1]

    # Build grid in normalized coords
    linspace = torch.linspace(0, 1, steps)
    xg, yg = torch.meshgrid(linspace, linspace, indexing="ij")
    grid_norm = torch.stack((xg.flatten(), yg.flatten()), dim=-1).to(device)
    grid_xy_m = observation_site.denormalize(grid_norm.cpu().numpy())  # (num_points,2) physical

    # Plot coords (flip y)
    x_plot = grid_xy_m[:, 0].reshape(steps, steps)
    y_plot = (-grid_xy_m[:, 1]).reshape(steps, steps)

    horizon = int(model.seq_len)
    frame_indices = _select_frame_window(t, horizon)
    fi_tag = _frame_indices_cache_tag(frame_indices)

    env_velocities: list[np.ndarray] = []
    env_densities: list[np.ndarray] = []

    density_sum = np.zeros((grid_xy_m.shape[0],), dtype=np.float32)

    grid_xy_m_torch = torch.from_numpy(grid_xy_m)

    for car_idx in range(num_cars):
        car_data = input_batch[0, car_idx].detach().cpu().numpy()
        if np.all(car_data == 0) or np.isnan(car_data).all():
            continue

        cache_name = f"px_ego{ego_id}_idx{car_idx}_start{s_frame}_steps{steps}{fi_tag}.pt"
        cache_path = os.path.join(cache_dir, cache_name)

        if os.path.exists(cache_path):
            px_i = torch.load(cache_path, map_location="cpu")
        else:
            perm = [car_idx] + [j for j in range(num_cars) if j != car_idx]
            input_perm = input_batch[:, perm]
            feature_perm = feature_batch[:, perm]
            type_perm = type_batch[:, perm]

            t0 = time.time()
            px_i = compute_px(model, input_perm, feature_perm, type_perm, grid_norm, frame_indices=frame_indices)
            t1 = time.time()
            print(f"compute_px car={car_idx} frames={frame_indices} took {t1-t0:.2f}s")
            torch.save(px_i, cache_path)

        # Map the requested global t to local index within window.
        if frame_indices[0] == t:
            t_local = 0
        elif frame_indices[-1] == t:
            t_local = len(frame_indices) - 1
        else:
            # Shouldn't happen with current window selection
            t_local = frame_indices.index(t)

        px_t = px_i[:, t_local].numpy().astype(np.float32)

        # Skip ego for environment aggregation
        if car_idx == ego_index:
            continue

        # Estimate env velocity at t using centroid motion over the window.
        centroid_traj = get_px_centroid(px_i, grid_xy_m_torch)
        dt = 0.04 * float(cost_step)

        if len(frame_indices) < 2:
            v_env = np.zeros((2,), dtype=np.float32)
        else:
            if t_local == 0:
                v_env = (centroid_traj[1] - centroid_traj[0]) / dt
            else:
                v_env = (centroid_traj[t_local] - centroid_traj[t_local - 1]) / dt
            v_env = v_env.astype(np.float32)

        env_velocities.append(v_env)
        env_densities.append(px_t)
        density_sum += px_t

    # Normalize for display convenience
    density_sum = density_sum / (density_sum.max() + 1e-9)

    return PreparedFrame(
        background_path=observation_site.background,
        ortho_px_to_meter=float(ortho_px_to_meter),
        x=x_plot,
        y=y_plot,
        grid_xy_m=grid_xy_m,
        density_sum=density_sum,
        env_velocities=env_velocities,
        env_densities=env_densities,
    )


def run_interactive_view(prep: PreparedFrame, *, alpha: float = 0.65) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise SystemExit("matplotlib is required for interactive view") from e

    background = plt.imread(prep.background_path)

    min_x = 0
    max_x = background.shape[1] * prep.ortho_px_to_meter
    min_y = background.shape[0] * prep.ortho_px_to_meter
    max_y = 0

    fig, ax = plt.subplots()
    ax.set_title("Interactive Field")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.imshow(background, extent=[min_x, max_x, min_y, max_y], aspect="equal")
    ax.set_box_aspect(1)

    origin_plot: Optional[tuple[float, float]] = None
    origin_scatter = None
    arrow_artist = None

    # Overlay mesh (density by default)
    mesh = ax.pcolormesh(prep.x, prep.y, prep.density_sum.reshape(prep.x.shape), shading="auto", cmap=plt.cm.turbo, alpha=alpha)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Density (sum, normalized)")

    info_text = ax.text(
        0.01,
        0.99,
        "Click to set origin.\n(No velocity yet: showing density)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    grid_xy = prep.grid_xy_m  # physical coords (y negative)

    def nearest_grid_index(x_phys: float, y_phys: float) -> int:
        dx = grid_xy[:, 0] - x_phys
        dy = grid_xy[:, 1] - y_phys
        return int(np.argmin(dx * dx + dy * dy))

    def compute_risk_field(v_ego_phys: np.ndarray) -> np.ndarray:
        # cost per env car is a scalar: ||v_ego - v_env||^2
        risk = np.zeros_like(prep.density_sum, dtype=np.float32)
        for px_t, v_env in zip(prep.env_densities, prep.env_velocities):
            dv = v_ego_phys - v_env
            cost = float(dv[0] * dv[0] + dv[1] * dv[1])
            risk += px_t * cost
        # Normalize for display (keep raw for reporting)
        risk_disp = risk / (risk.max() + 1e-9)
        return risk, risk_disp

    def redraw_overlay(field_2d: np.ndarray, *, label: str) -> None:
        nonlocal mesh, cbar
        mesh.remove()
        mesh = ax.pcolormesh(prep.x, prep.y, field_2d, shading="auto", cmap=plt.cm.turbo, alpha=alpha)
        cbar.update_normal(mesh)
        cbar.set_label(label)

    def clear_state() -> None:
        nonlocal origin_plot, origin_scatter, arrow_artist
        origin_plot = None
        if origin_scatter is not None:
            origin_scatter.remove()
            origin_scatter = None
        if arrow_artist is not None:
            arrow_artist.remove()
            arrow_artist = None
        redraw_overlay(prep.density_sum.reshape(prep.x.shape), label="Density (sum, normalized)")
        info_text.set_text("Click to set origin.\n(No velocity yet: showing density)")
        fig.canvas.draw_idle()

    def on_click(event):
        nonlocal origin_plot, origin_scatter, arrow_artist
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        if origin_plot is None:
            origin_plot = (float(event.xdata), float(event.ydata))
            origin_scatter = ax.scatter([origin_plot[0]], [origin_plot[1]], s=30, c="white", edgecolors="black", linewidths=0.8, zorder=5)

            # report density at origin
            origin_phys = (origin_plot[0], -origin_plot[1])
            idx = nearest_grid_index(origin_phys[0], origin_phys[1])
            dens_val = float(prep.density_sum[idx])
            info_text.set_text(
                f"Origin set at (x={origin_plot[0]:.2f}, y={origin_plot[1]:.2f})\n"
                f"Density@origin (norm) = {dens_val:.4f}\n"
                "Click again to set velocity (2nd point defines v in 1s).\n"
                "Press C / Delete / Backspace to clear."
            )
            fig.canvas.draw_idle()
            return

        # Second click sets velocity vector
        end_plot = (float(event.xdata), float(event.ydata))
        dx_plot = end_plot[0] - origin_plot[0]
        dy_plot = end_plot[1] - origin_plot[1]

        # Convert from plot coords (y flipped) to physical coords
        v_ego_phys = np.array([dx_plot, -dy_plot], dtype=np.float32)  # m/s (interpreting second click as 1s endpoint)

        if arrow_artist is not None:
            arrow_artist.remove()
        arrow_artist = ax.annotate(
            "",
            xy=end_plot,
            xytext=origin_plot,
            arrowprops=dict(arrowstyle="->", color="white", linewidth=2.0),
            zorder=6,
        )

        risk_raw, risk_disp = compute_risk_field(v_ego_phys)
        redraw_overlay(risk_disp.reshape(prep.x.shape), label="Risk (normalized for display)")

        origin_phys = (origin_plot[0], -origin_plot[1])
        idx = nearest_grid_index(origin_phys[0], origin_phys[1])
        risk_at_origin = float(risk_raw[idx])

        speed = float(np.sqrt(v_ego_phys[0] ** 2 + v_ego_phys[1] ** 2))
        info_text.set_text(
            f"Origin: (x={origin_plot[0]:.2f}, y={origin_plot[1]:.2f})\n"
            f"v_ego (phys) = [{v_ego_phys[0]:.2f}, {v_ego_phys[1]:.2f}] m/s, |v|={speed:.2f}\n"
            f"Risk@origin (raw) = {risk_at_origin:.6f}\n"
            "Press C / Delete / Backspace to clear."
        )

        fig.canvas.draw_idle()

    def on_key(event):
        k = (event.key or "").lower()
        if k in {"c", "delete", "backspace"}:
            clear_state()
        elif k in {"escape", "q"}:
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Interactive risk field viewer for a single future frame. "
            "1st click sets origin; 2nd click sets velocity vector (interpreted as endpoint after 1s), "
            "then overlays risk field and shows risk at origin. Press C/Delete/Backspace to clear."
        )
    )
    p.add_argument("--segment", type=int, required=True, help="Index into test_loader batches (same semantics as visualize_field.py)")
    p.add_argument("--t", type=int, required=True, help="Future frame index t in [0, model.seq_len-1]")
    p.add_argument("--steps", type=int, default=128, help="Grid resolution per axis")
    p.add_argument("--site", type=str, default="08", help="InD site id, e.g. 08")
    p.add_argument("--model", type=str, required=True, help="Path to model .pt (state_dict)")
    p.add_argument("--device", type=str, default=None, help="cpu | cuda | auto")
    p.add_argument("--cost-step", type=int, default=2, help="dt = 0.04 * cost_step")
    p.add_argument("--cache-dir", type=str, default=None, help="Cache dir for px tensors (defaults to videos/*nf_cache)")
    p.add_argument("--alpha", type=float, default=0.65, help="Overlay alpha")
    return p


def main() -> None:
    from model.MultiTrajFlow import MultiTrajFlow
    from trajflow_config import preset, seed_everything

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
        include_future=cfg["include_future"],
    )

    # Prefer public properties when available; fallback to internal loader.
    site_id = str(args.site)
    obs = getattr(ind, f"observation_site_{site_id}", None)
    if obs is None:
        obs = ind._get_observation_site([site_id])
    observation_site = obs

    traj_flow = MultiTrajFlow(
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

    traj_flow.load_state_dict(torch.load(args.model, map_location=device))
    traj_flow.eval()

    prep = prepare_environment_for_frame(
        observation_site=observation_site,
        model=traj_flow,
        segment_index=args.segment,
        t=args.t,
        steps=args.steps,
        device=device,
        ego_index=0,
        cost_step=args.cost_step,
        cache_root=args.cache_dir,
    )

    run_interactive_view(prep, alpha=float(args.alpha))


if __name__ == "__main__":
    main()
