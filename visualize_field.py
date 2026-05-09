import os
import subprocess
import torch
import numpy as np
import time
import torch.nn.functional as F
from typing import List, Optional

from datasets.InD import InD
from safety_metrics import (
    compute_has_intersection_ego_neighbors_from_batch,
    compute_min_distance_ego_neighbors_from_batch,
)

try:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
except ModuleNotFoundError:
    matplotlib = None
    plt = None


col_width = 6
fig_height = col_width / 1.618 

if plt is not None:
    plt.rcParams.update({
        "figure.figsize": (col_width, fig_height), # 设置画布真实大小
        "font.serif": ["Times"],                   # 指定具体衬线字体
        "font.size": 8,                            # 基础字体大小设置为 8pt
        "axes.labelsize": 8,                       # 坐标轴标签大小
        "legend.fontsize": 7,                      # 图例字体稍小一点
        "xtick.labelsize": 7,                      # 刻度字体
        "ytick.labelsize": 7,
        "figure.dpi": 300                          # 设置显示分辨率
    })



import warnings

warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*weights_only=False.*"
)


def _frame_indices_cache_tag(frame_indices: Optional[List[int]]) -> str:
    if frame_indices is None:
        return ""
    if len(frame_indices) == 0:
        return "_fEMPTY"
    fi = [int(x) for x in frame_indices]
    # Compact tag for contiguous indices; otherwise include a short hash.
    is_contiguous = all((fi[i] - fi[i - 1]) == 1 for i in range(1, len(fi)))
    if is_contiguous:
        return f"_f{fi[0]}-{fi[-1]}"
    import hashlib

    h = hashlib.sha1(",".join(map(str, fi)).encode("utf-8")).hexdigest()[:8]
    return f"_fH{h}_n{len(fi)}"


def _velocity_from_positions(
    future_xy: torch.Tensor,
    *,
    dt: float,
    last_xy: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute per-step velocity (m/s) aligned with each row in `future_xy`.

    future_xy: (T,2) metric coords
    last_xy: (2,) metric coord at time -1 (last observed). If provided, v[0]
             uses (future_xy[0]-last_xy)/dt.
    """
    if future_xy.dim() != 2 or future_xy.shape[-1] != 2:
        raise ValueError("future_xy must be (T,2)")
    if future_xy.shape[0] == 0:
        raise ValueError("future_xy must have T>0")

    if dt is None or float(dt) <= 0:
        raise ValueError("dt must be positive")

    if future_xy.shape[0] == 1:
        if last_xy is None:
            return torch.zeros((1, 2), dtype=future_xy.dtype, device=future_xy.device)
        return (future_xy[0:1] - last_xy.view(1, 2)) / float(dt)

    v = (future_xy[1:] - future_xy[:-1]) / float(dt)  # (T-1,2)
    v_last = v[-1:].clone()
    if last_xy is not None:
        v0 = (future_xy[0:1] - last_xy.view(1, 2)) / float(dt)
        v = torch.cat([v0, v], dim=0)
    else:
        v = torch.cat([v, v_last], dim=0)

    if v.shape[0] != future_xy.shape[0]:
        # last_xy path yields T velocities already; else we appended last.
        v = torch.cat([v, v_last], dim=0)
    return v


def _expand_or_slice_velocity(
    v: torch.Tensor,
    *,
    seq_len: int,
    frame_indices: Optional[List[int]],
) -> torch.Tensor:
    """Normalize ego velocity to shape (seq_len,2) for current horizon."""
    if v.dim() == 1 and v.numel() == 2:
        return v.view(1, 2).expand(seq_len, 2)
    if v.dim() == 2 and v.shape[1] == 2:
        if frame_indices is None:
            if v.shape[0] != seq_len:
                raise ValueError(f"Per-frame velocity length {v.shape[0]} != seq_len {seq_len}")
            return v
        # frame_indices refer to the original horizon indices.
        if v.shape[0] <= max(frame_indices):
            raise ValueError("Per-frame velocity is shorter than requested frame_indices")
        return v[torch.as_tensor(frame_indices, device=v.device, dtype=torch.long)]
    raise ValueError("ego velocity must be shape (2,) or (T,2)")


def _denormalize_xy(xy_norm: torch.Tensor) -> torch.Tensor:
    """Denormalize normalized (x,y) using InD spatial_boundaries."""
    from datasets.InD import spatial_boundaries

    bounds = torch.as_tensor(spatial_boundaries, dtype=xy_norm.dtype, device=xy_norm.device)
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    return xy_norm * (hi - lo) + lo


def _dedup_candidates_by_start_frame(
    candidates: list[tuple[int, int]],
    *,
    start_frame_gap: int,
) -> list[int]:
    """Deduplicate risky candidates by `startFrame` proximity.

    candidates: list of (start_frame, scanned_index)

    Rule: if a later candidate's start_frame is within `start_frame_gap` frames
    of an earlier kept candidate, drop the later one (keep the earlier).
    """
    if start_frame_gap is None:
        return [idx for _, idx in candidates]
    gap = int(start_frame_gap)
    if gap < 0:
        gap = 0

    # Sort by start_frame ascending; if tie, keep the smaller scanned index.
    candidates_sorted = sorted(candidates, key=lambda x: (x[0], x[1]))

    kept: list[int] = []
    last_start: int | None = None
    for sf, idx in candidates_sorted:
        if last_start is None:
            kept.append(idx)
            last_start = sf
            continue
        # If too close (within gap), drop later
        if (sf - last_start) <= gap:
            continue
        kept.append(idx)
        last_start = sf

    return kept


def select_risky_indices_by_gt(
    observation_site,
    *,
    method: str = "min_distance",
    methods: Optional[List[str]] = None,
    logic: str = "or",
    sampling_step: int = 1,
    distance_threshold_m: float = 3.0,
    deduplicate_by_start_frame: bool = True,
    start_frame_gap: int = 100,
    max_results: int = 10,
    max_scan: int = 2000,
    mode: str = "test",
):
    """Scan dataset sequentially and return indices classified as risky (GT future).

    Only uses future *ground truth* and only checks ego-vs-others.

    Supported atomic methods:
    - "min_distance": risk if min_{neighbor,t} ||x_ego(t)-x_neighbor(t)|| < distance_threshold_m
    - "trajectory_intersection": risk if ego future polyline intersects any neighbor polyline

    Composition:
    - Pass `methods=[...]` to combine multiple atomic methods.
    - Use `logic="or"` (default) or `logic="and"`.
    """

    loader = observation_site.test_loader if mode == "test" else observation_site.train_loader

    if methods is None:
        method_list = [method]
    else:
        if not isinstance(methods, list) or len(methods) == 0:
            raise ValueError("methods must be a non-empty list[str] when provided")
        method_list = methods

    method_list_l = [m.lower().strip() for m in method_list]
    logic_l = logic.lower().strip()
    if logic_l not in {"or", "and"}:
        raise ValueError(f"Unknown logic: {logic}. Use 'or' or 'and'.")

    risky_candidates: list[tuple[int, int]] = []  # (startFrame, scanned_index)
    scanned = 0

    for batch in loader:
        bsz = batch["input"].shape[0]
        start_frames = batch.get("startFrame")

        is_risky: torch.Tensor | None = None
        for m in method_list_l:
            if m in {"min_distance", "distance", "nearest_distance", "closest"}:
                min_dist = compute_min_distance_ego_neighbors_from_batch(batch)
                m_risky = min_dist < float(distance_threshold_m)

            elif m in {"trajectory_intersection", "intersection", "intersect"}:
                m_risky = compute_has_intersection_ego_neighbors_from_batch(batch)

            else:
                raise ValueError(f"Unknown GT risk method: {m}")

            if is_risky is None:
                is_risky = m_risky
            else:
                is_risky = (is_risky | m_risky) if logic_l == "or" else (is_risky & m_risky)

        if is_risky is None:
            raise RuntimeError("Internal error: empty methods list")

        for j in range(bsz):
            if scanned >= max_scan:
                break

            if bool(is_risky[j].item()):
                if start_frames is None:
                    sf = scanned
                else:
                    sf = int(start_frames[j].item())
                risky_candidates.append((sf, scanned))
            scanned += 1

        if scanned >= max_scan:
            break

    if deduplicate_by_start_frame:
        risky = _dedup_candidates_by_start_frame(
            risky_candidates,
            start_frame_gap=start_frame_gap,
        )
    else:
        risky = [idx for _, idx in sorted(risky_candidates, key=lambda x: (x[0], x[1]))]

    return risky[: int(max_results)]


def makedir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def compute_px(
    model,
    inputs,
    features,
    types,
    grid,
    frame_indices=None,
    batch_size=512,
):
    """
    Compute the likelihood px over the grid points during a period of time.
    input: (batch, max_num_cars, seq_len, 2)
    features: (batch, max_num_cars, seq_len, feature_dim)
    grid: (num_points, 2) in normalized coords
    returns: px of shape (num_points, seq_len)
    """
    device = inputs.device
    model.eval()
    with torch.no_grad():
        # build embedding from multi-car input+features
        cond = torch.cat(
            [inputs, features], dim=-1
        )  # (batch, max_num_cars, seq_len, feat+in)
        embedding = model.encoder(None, cond, types)  # (batch, emb_dim)
        # take first sample in batch (we visualize one example) and repeat for grid batches
        emb_single = embedding[0:1].to(device)

        px_parts = []
        for grid_batch in grid.split(batch_size, dim=0):
            b = grid_batch.shape[0]
            frame_indices_tensor = None
            # expand each grid batch for processing (a flattened batch of query points)
            if frame_indices is None:
                flattened_grid = (
                    grid_batch.unsqueeze(1).expand(-1, model.seq_len, -1).to(device)
                )  # (b, seq_len, 2)
            else:
                frame_indices_tensor = torch.as_tensor(frame_indices, device=device)
                if frame_indices_tensor.dim() == 1:
                    frame_indices_tensor = frame_indices_tensor.unsqueeze(0).expand(
                        grid_batch.shape[0], -1
                    )
                flattened_grid = (
                    grid_batch.unsqueeze(1)
                    .expand(-1, len(frame_indices), -1)
                    .to(device)
                )
            emb_rep = emb_single.expand(b, emb_single.shape[1]).to(
                device
            )  # (b, emb_dim)
            z, det = model.flow(
                flattened_grid, emb_rep, frame_indices=frame_indices_tensor
            )  # z is the z_t0 in CNF
            _, logpx = model.log_prob(
                z, det, embedding=emb_rep
            )  # logpz is the logpz_t0, logpx is logpz_t1
            px = logpx.exp().cpu()  # (b, seq_len)
            px_parts.append(px)

        px = torch.cat(px_parts, dim=0)  # (num_points, seq_len)
        # Normalize px per frame
        px_max = px.max(dim=0, keepdim=True).values  # (1, seq_len)
        px = px / (px_max + 1e-9)

        return px


def get_px_centroid(px, grid, top_k=30):
    """
    Compute the centroid of top_k probability grid points.
    px: (num_points, seq_len)
    grid: (num_points, 2)
    returns: (seq_len, 2) centroid coordinates
    """
    device = px.device
    grid = grid.to(device)
    seq_len = px.shape[1]
    centroids = []

    for t in range(seq_len):
        prob_t = px[:, t]
        vals, idxs = torch.topk(prob_t, k=min(top_k, prob_t.shape[0]))
        top_points = grid[idxs]  # (top_k, 2)
        centroid = top_points.mean(dim=0)
        centroids.append(centroid)

    return torch.stack(centroids, dim=0).cpu().numpy()


def compute_cost(v, px, grid, dt=0.08, step=None):
    """
    Compute the cost of the ego car and another car
    v: ego velocity. Accepts shape (2,) constant (m/s) or (seq_len,2) per-frame (m/s)
    px: (num_points, seq_len) likelihood map of the other car
    grid: (num_points, 2)
    returns: The cost map of shape(num_points, seq_len)
    """
    device = px.device

    # Backward-compatible: some call sites pass step=N (frames) instead of dt.
    # If provided, derive dt from dataset sampling (25Hz).
    if step is not None:
        dt = 0.04 * float(step)
    if not isinstance(dt, (int, float)):
        dt = 0.08

    # px: (num_points, seq_len)
    # grid: (num_points, 2)
    # Get centroid trajectory (seq_len, 2)
    traj = torch.from_numpy(get_px_centroid(px, grid)).to(device)  # (seq_len, 2)

    window_size = 5
    padding = window_size // 2
    traj_t = traj.t().unsqueeze(0)  # (1, 2, seq_len)
    traj_padded = torch.nn.functional.pad(traj_t, (padding, padding), mode="replicate")
    traj = (
        torch.nn.functional.avg_pool1d(traj_padded, kernel_size=window_size, stride=1)
        .squeeze(0)
        .t()
    )

    # Compute velocity of centroid at each frame
    if traj.shape[0] < 2:
        v_centroid = torch.zeros_like(traj)
    else:
        v_centroid = (traj[1:] - traj[:-1]) / dt  # (seq_len-1, 2)
        v_centroid = torch.cat([v_centroid, v_centroid[-1:].clone()], dim=0)  # (seq_len, 2)

    # v: (2,) or (seq_len,2)
    v_t = v.to(device)
    if v_t.dim() == 1:
        v_ego = v_t.view(1, 2).expand_as(v_centroid)
    else:
        if v_t.shape[0] != v_centroid.shape[0] or v_t.shape[1] != 2:
            raise ValueError("ego velocity sequence must be (seq_len,2)")
        v_ego = v_t

    # Relative velocity difference
    rel_v = v_ego - v_centroid  # (seq_len, 2)
    cost_per_step = (rel_v**2).sum(dim=1)  # (seq_len,)
    # cost_per_step = torch.norm(rel_v, dim=-1)  # (seq_len,)

    # Expand to (num_points, seq_len)
    cost_map = cost_per_step.unsqueeze(0).expand(px.shape[0], -1).clone()
    return cost_map


def generate_gt_traj_frame(
    background,
    trajs,
    history_trajs,
    min_x,
    max_x,
    min_y,
    max_y,
    t,
    output_dir,
    zoom_factor=1.8,
    title_prefix="GT Future Trajectories",
):
    """Render a single frame that overlays ground-truth future trajectories on background."""
    frame = os.path.join(output_dir, f"frame_{t:03d}.png")
    if os.path.exists(frame):
        os.remove(frame)

    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    half_width = (max_x - min_x) / (2 * zoom_factor)
    half_height = (max_y - min_y) / (2 * zoom_factor)

    plt.xlim(center_x - half_width, center_x + half_width)
    plt.ylim(center_y - half_height, center_y + half_height)
    plt.gca().set_box_aspect(1)

    plt.imshow(background, extent=[min_x, max_x, min_y, max_y], aspect="equal")

    # Past (history) trajectories: static across all frames.
    if history_trajs:
        for k, htraj in enumerate(history_trajs):
            if htraj is None or len(htraj) == 0:
                continue
            if htraj.ndim != 2 or htraj.shape[1] != 2:
                continue

            # Filter NaNs
            mask = ~(np.isnan(htraj[:, 0]) | np.isnan(htraj[:, 1]))
            htraj = htraj[mask]
            if htraj.shape[0] < 2:
                continue

            if k == 0:
                color = "#FF9100"  # ego
                lw = 1.2
                alpha = 0.55
            else:
                color = "#00AEEF"  # neighbors
                lw = 1.0
                alpha = 0.45

            plt.plot(
                htraj[:, 0],
                htraj[:, 1],
                color=color,
                linewidth=lw,
                alpha=alpha,
            )

    # trajs: list of (T,2) in metric coords, y already flipped for plotting
    has_label = False
    for k, traj in enumerate(trajs):
        if traj is None or len(traj) == 0:
            continue
        seg = traj[: t + 1]
        if seg.ndim != 2 or seg.shape[1] != 2:
            continue
        # Filter NaNs
        mask = ~(np.isnan(seg[:, 0]) | np.isnan(seg[:, 1]))
        seg = seg[mask]
        if seg.shape[0] < 2:
            continue

        if k == 0:
            color = "#FF9100"  # ego
            label = "Ego (GT)"
            lw = 1.8
            alpha = 1.0
        else:
            color = "#00AEEF"  # neighbors
            label = "Others (GT)" if k == 1 else None
            lw = 1.2
            alpha = 0.9

        if label is not None:
            has_label = True

        plt.plot(
            seg[:, 0],
            seg[:, 1],
            color=color,
            linewidth=lw,
            alpha=alpha,
            label=label,
        )

        # Mark current point
        pt = seg[-1]
        plt.scatter([pt[0]], [pt[1]], s=6, c=color, alpha=alpha)

    plt.title(f"{title_prefix}, Frame: {t}")
    plt.xlabel("X")
    plt.ylabel("Y")
    if has_label:
        plt.legend(loc="best")
    plt.savefig(frame)
    plt.close()


def generate_gt_traj_video(
    background_image,
    gt_future_trajs,
    history_trajs,
    ortho_px_to_meter,
    output_dir,
    name="gt_trajs",
    fps=10,
    zoom_factor=1.8,
):
    """Generate an MP4 showing all vehicles' ground-truth future trajectories (no field)."""
    frames_dir = os.path.join(output_dir, f"{name}_frames")
    makedir(frames_dir)

    background = plt.imread(background_image)

    min_x = 0
    max_x = background.shape[1] * ortho_px_to_meter
    min_y = background.shape[0] * ortho_px_to_meter
    max_y = 0

    # Determine sequence length from the first valid trajectory
    seq_len = 0
    for tr in gt_future_trajs:
        if tr is not None and len(tr) > 0:
            seq_len = tr.shape[0]
            break
    if seq_len <= 0:
        return

    for t in range(seq_len):
        generate_gt_traj_frame(
            background=background,
            trajs=gt_future_trajs,
            history_trajs=history_trajs,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            t=t,
            output_dir=frames_dir,
            zoom_factor=zoom_factor,
        )

    frame_source = os.path.join(frames_dir, "frame_%03d.png")
    video_destination = os.path.join(output_dir, f"{name}.mp4")

    if os.path.exists(video_destination):
        os.remove(video_destination)

    command = [
        "ffmpeg",
        "-y",
        "-r",
        str(int(fps)),
        "-i",
        frame_source,
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        video_destination,
    ]
    subprocess.run(command, check=True)


def compute_velocity_field(
    model, inputs, features, types, grid, frame_indices, dt=0.08, batch_size=512
):
    """
    Compute velocity field based on latent space consistency.
    grid: (num_points, 2) physical space grid coordinates
    frame_indices: (seq_len,) time indices
    returns: v_field (num_points, seq_len, 2)
    """
    device = inputs.device
    model.eval()

    num_points = grid.shape[0]
    seq_len = len(frame_indices)

    # 1. Extract conditional embedding (consistent with compute_px)
    with torch.no_grad():
        cond = torch.cat([inputs, features], dim=-1)
        embedding = model.encoder(None, cond, types)
        emb_single = embedding[0:1].to(device)  # (1, emb_dim)

    # Initialize velocity field
    v_field = torch.zeros((num_points, seq_len, 2), device="cpu")

    # Edge case: single-frame query has no meaningful finite-diff.
    if seq_len <= 1:
        return v_field

    # Process grid in batches to save memory
    for i in range(0, num_points, batch_size):
        end_idx = min(i + batch_size, num_points)
        curr_grid_batch = grid[i:end_idx].to(device)  # (b, 2)
        b = curr_grid_batch.shape[0]

        # Expand grid to all time steps (b, seq_len, 2)
        x_t = curr_grid_batch.unsqueeze(1).expand(-1, seq_len, -1)

        # Prepare frame_indices
        # t: [0, 1, 2, ..., T-1]
        t_current = torch.as_tensor(frame_indices, device=device).float()
        t_current_rep = t_current.unsqueeze(0).expand(b, -1)  # (b, seq_len)

        # t_next: [1, 2, ..., T-1, T-1] (last frame velocity set to 0 or forward difference)
        t_next = t_current.clone()
        t_next[:-1] = t_current[1:]
        t_next_rep = t_next.unsqueeze(0).expand(b, -1)

        emb_rep = emb_single.expand(b, -1)

        with torch.no_grad():
            # Step A: Map to latent space z_t = f(x_t, t)
            # Note: forward here corresponds to self.net.forward in the code
            z, _ = model.flow(x_t, emb_rep, frame_indices=t_current_rep)

            # Step B: Inverse map from latent space to next physical space x_{t+1} = f^{-1}(z_t, t+1)
            # Note: Use reverse=True to call inverse
            x_next, _ = model.flow(z, emb_rep, frame_indices=t_next_rep, reverse=True)

            # Step C: Compute displacement and convert to velocity (x_next - x_t) / dt
            # x_next shape: (b, seq_len, 2)
            v_batch = (x_next - x_t) / dt

            # Last frame handling: fill with previous frame velocity.
            v_batch[:, -1, :] = v_batch[:, -2, :]

            v_field[i:end_idx] = v_batch.cpu()

    return v_field  # (num_points, seq_len, 2)


def compute_cost_pred(px_ego, px, grid, step=2):
    """
    Compute the cost of the ego car and another car using predicted trajectories
    px_ego: (num_points, seq_len) likelihood map of the ego car
    px: (num_points, seq_len) likelihood map of the other car
    grid: (num_points, 2)
    returns: The cost map of shape(num_points, seq_len)
    """
    device = px.device
    dt = 0.04 * step  # 25 Hz * step

    traj_ego = torch.from_numpy(get_px_centroid(px_ego, grid)).to(
        device
    )  # (seq_len, 2)
    traj_other = torch.from_numpy(get_px_centroid(px, grid)).to(device)  # (seq_len, 2)

    window_size = 5
    padding = window_size // 2
    traj_ego_t = traj_ego.t().unsqueeze(0)  # (1, 2, seq_len)
    traj_ego_padded = torch.nn.functional.pad(
        traj_ego_t, (padding, padding), mode="replicate"
    )
    traj_ego = (
        torch.nn.functional.avg_pool1d(
            traj_ego_padded, kernel_size=window_size, stride=1
        )
        .squeeze(0)
        .t()
    )

    traj_other_t = traj_other.t().unsqueeze(0)  # (1, 2, seq_len)
    traj_other_padded = torch.nn.functional.pad(
        traj_other_t, (padding, padding), mode="replicate"
    )
    traj_other = (
        torch.nn.functional.avg_pool1d(
            traj_other_padded, kernel_size=window_size, stride=1
        )
        .squeeze(0)
        .t()
    )

    # Compute velocity of centroid at each frame
    if traj_ego.shape[0] < 2:
        v_ego = torch.zeros_like(traj_ego)
        v_other = torch.zeros_like(traj_other)
    else:
        v_ego = (traj_ego[1:] - traj_ego[:-1]) / dt  # (seq_len-1, 2)
        v_ego = torch.cat([v_ego, v_ego[-1:].clone()], dim=0)  # (seq_len, 2)

        v_other = (traj_other[1:] - traj_other[:-1]) / dt  # (seq_len-1, 2)
        v_other = torch.cat([v_other, v_other[-1:].clone()], dim=0)  # (seq_len, 2)

    # Relative velocity difference
    rel_v = v_ego - v_other  # (seq_len, 2)
    cost_per_step = (rel_v**2).sum(dim=1)  # (seq_len,)
    # cost_per_step = torch.norm(rel_v, dim=-1)  # (seq_len,)

    # Expand to (num_points, seq_len)
    cost_map = cost_per_step.unsqueeze(0).expand(px.shape[0], -1).clone()
    return cost_map


def generate_frame(
    background,
    x,
    y,
    likelihood,
    trajs,
    history_trajs,
    min_x,
    max_x,
    min_y,
    max_y,
    t,
    output_dir,
    label="Future Likelihood",
    vmax=None,
    velocity_field_frame=None,
    zoom_factor=1.8,
):
    frame = os.path.join(output_dir, f"frame_{t:03d}.png")
    if os.path.exists(frame):
        os.remove(frame)

    # plt.figure(figsize=(10, 8))
    # plt.xlim(min_x, max_x)
    # plt.ylim(min_y, max_y)

    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    half_width = (max_x - min_x) / (2 * zoom_factor)
    half_height = (max_y - min_y) / (2 * zoom_factor)
    
    plt.xlim(center_x - half_width, center_x + half_width)
    plt.ylim(center_y - half_height, center_y + half_height)

    plt.gca().set_box_aspect(1) 

    plt.imshow(background, extent=[min_x, max_x, min_y, max_y], aspect="equal")

    # Past (history) trajectories: static across all frames.
    if history_trajs:
        for k, htraj in enumerate(history_trajs):
            if htraj is None or len(htraj) == 0:
                continue
            if htraj.ndim != 2 or htraj.shape[1] != 2:
                continue

            mask = ~(np.isnan(htraj[:, 0]) | np.isnan(htraj[:, 1]))
            htraj = htraj[mask]
            if htraj.shape[0] < 2:
                continue

            if k == 0:
                color = "#FF9100"  # ego
                lw = 1.2
                alpha = 0.55
            else:
                color = "#00AEEF"  # neighbors
                lw = 1.0
                alpha = 0.45

            plt.plot(
                htraj[:, 0],
                htraj[:, 1],
                color=color,
                linewidth=lw,
                alpha=alpha,
            )

    color_map = plt.cm.turbo
    color_map.set_bad(color="none")
    heat_map = plt.pcolormesh(
        x, y, likelihood, shading="auto", cmap=color_map, vmin=0, vmax=vmax
    )

    plt.colorbar(heat_map, label=label)

    # if velocity_field_frame is not None:
    #     # velocity_field_frame: (num_points, 2)
    #     u = velocity_field_frame[:, 0].reshape(x.shape)
    #     v = velocity_field_frame[:, 1].reshape(y.shape)

    #     # Quiver plot with subsampling for better visualization
    #     skip = 16
    #     plt.quiver(
    #         x[::skip, ::skip],
    #         y[::skip, ::skip],
    #         u[::skip, ::skip] * 1.8,
    #         v[::skip, ::skip] * 1.8,
    #         color="white",
    #         alpha=0.6,
    #         width=0.004,
    #         scale=30 / zoom_factor,
    #     )

    if trajs:
        for k, traj in enumerate(trajs):
            plt.plot(
                traj[:, 0],
                traj[:, 1],
                color="#FF9100",
                linewidth=1.5,
                label="Ego Trajectory" if k == 0 else None,
            )

    plt.title(f"Risk Field Heatmap with lower resolution, Frame: {t}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()

    plt.savefig(frame)
    plt.close()


def generate_video(
    background_image,
    grid,
    px,
    prob_threshold,
    trajs,
    history_trajs,
    ortho_px_to_meter,
    steps,
    output_dir,
    i,
    is_risk=False,
    velocity_field=None,
):
    frames_dir = os.path.join(f"{output_dir}", "frames", f"video{i}")
    makedir(frames_dir)

    x = grid[:, 0].reshape(steps, steps)
    y = -grid[:, 1].reshape(steps, steps)

    background = plt.imread(background_image)

    min_x = 0
    max_x = background.shape[1] * ortho_px_to_meter
    min_y = background.shape[0] * ortho_px_to_meter
    max_y = 0

    px = px.cpu()
    max_val = torch.max(px).item() if is_risk else 1.0
    label = "Future Risk" if is_risk else "Future Likelihood"

    for t in range(px.shape[1]):
        likelihood = px[:, t].numpy().reshape(steps, steps)
        if not is_risk:
            likelihood = likelihood / (np.max(likelihood) + 1e-9)
        likelihood = np.where(likelihood < prob_threshold, np.nan, likelihood)

        plot_trajs = []
        for traj in trajs:
            current_seg = traj[: t + 1]
            plot_seg = np.stack([current_seg[:, 0], -current_seg[:, 1]], axis=-1)
            plot_trajs.append(plot_seg)

        # history_trajs are already in plotting coordinate (y flipped).
        plot_history = history_trajs if history_trajs is not None else []

        v_frame = velocity_field[:, t, :] if velocity_field is not None else None

        generate_frame(
            background,
            x,
            y,
            likelihood,
            plot_trajs,
            plot_history,
            min_x,
            max_x,
            min_y,
            max_y,
            t,
            frames_dir,
            label=label,
            vmax=max_val,
            velocity_field_frame=v_frame,
        )

    frame_source = os.path.join(f"{frames_dir}", "frame_%03d.png")
    video_destination = os.path.join(output_dir, f"video{i}.mp4")

    if os.path.exists(video_destination):
        os.remove(video_destination)
    command = [
        "ffmpeg",
        "-y",
        "-r",
        "10",
        "-i",
        frame_source,
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        video_destination,
    ]
    subprocess.run(command, check=True)


def visualize(
    observation_site,
    model,
    indices,
    steps,
    prob_threshold,
    output_dir,
    device,
    given_v=None,
    *,
    ego_velocity_source: str = "feature_last",
    ego_velocity_traj: Optional[np.ndarray] = None,
    ego_velocity_overrides: Optional[dict[int, List[float]]] = None,
    frame_indices: Optional[List[int]] = None,
    cost_step: int = 2,
    ego_index=0,
    all_cars=False,
    without_traj=False,
    pred_cost=False,
    radius=0,
    show_velocity_field=False,
    ground_truth_future=False,
):
    makedir(output_dir)
    cache_dir = "cnf_cache" if model.use_cnf else "dnf_cache"
    cache_dir = os.path.join("videos", cache_dir)
    makedir(cache_dir)

    model.eval()

    fudge_factor = 11.5
    ortho_px_to_meter = observation_site.ortho_px_to_meter * fudge_factor

    test_data = list(iter(observation_site.test_loader))

    for idx in indices:
        cur_dir = os.path.join(output_dir, f"sample_{idx}")
        makedir(cur_dir)
        if idx >= len(test_data):
            print(f"Index {idx} out of range for test data.")
            continue
        batch = test_data[idx]
        input_batch, feature_batch, type_batch, target_batch, track_id, start_frame = (
            batch["input"],
            batch["feature"],
            batch["type"],
            batch["target"],
            batch["trackId"],
            batch["startFrame"],
        )

        future_batch = batch.get("future", None)

        print(
            f"Visualizing index {idx}, track IDs: {track_id}, start frames: {start_frame}"
        )

        s_frame = (
            start_frame[0].item() if torch.is_tensor(start_frame) else start_frame[0]
        )
        ego_id = track_id[0].item() if torch.is_tensor(track_id) else track_id[0]

        # move to device
        input_batch = input_batch.to(device)  # (batch, max_num_cars, seq_len, 2)
        feature_batch = feature_batch.to(
            device
        )  # (batch, max_num_cars, seq_len, feature_dim)
        target_batch = target_batch.to(device)  # (batch, seq_len, 2)
        type_batch = type_batch.to(device)  # (batch, max_num_cars)

        if future_batch is not None:
            future_batch = future_batch.to(device)  # (batch, max_num_cars, pred_len, 2)

        num_cars = input_batch.shape[1]
        valid_car_indices = []
        for car_idx in range(num_cars):
            car_data = input_batch[0, car_idx].cpu().numpy()
            if not (np.all(car_data == 0) or np.isnan(car_data).all()):
                valid_car_indices.append(car_idx)
        print(f"Sample idx {idx}: valid cars = {len(valid_car_indices)}")

        # Build history trajectories (observed past) for all valid vehicles.
        # These are drawn statically from the first frame onwards.
        past_xy = _denormalize_xy(input_batch[0]).detach().cpu().numpy()  # (N, H, 2)

        def _history_polyline(car_idx: int) -> Optional[np.ndarray]:
            tr = past_xy[car_idx]
            if tr is None:
                return None
            if np.isnan(tr).all() or np.all(tr == 0):
                return None
            # Drop padded all-zero rows and NaNs
            mask = ~(np.isnan(tr[:, 0]) | np.isnan(tr[:, 1]))
            mask = mask & ~((tr[:, 0] == 0) & (tr[:, 1] == 0))
            tr = tr[mask]
            if tr.shape[0] < 2:
                return None
            # Flip Y for plotting coordinates
            return np.stack([tr[:, 0], -tr[:, 1]], axis=-1)

        # Keep ordering: ego first, then others. Preserve ego slot even if missing.
        ego_hist = _history_polyline(ego_index)
        history_trajs_plot: list[Optional[np.ndarray]] = [ego_hist]
        for car_idx in range(num_cars):
            if car_idx == ego_index:
                continue
            car_data = input_batch[0, car_idx].cpu().numpy()
            if np.all(car_data == 0) or np.isnan(car_data).all():
                continue
            h = _history_polyline(car_idx)
            if h is not None:
                history_trajs_plot.append(h)

        # Extra: Ground-truth future trajectories video (no field)
        if future_batch is not None and ground_truth_future:
            # Denormalize and flip Y for plotting to match background coordinate system.
            future_xy = _denormalize_xy(future_batch[0])  # (N, T, 2)

            def _valid_traj(arr: np.ndarray) -> bool:
                if arr is None:
                    return False
                if np.isnan(arr).all() or np.all(arr == 0):
                    return False
                return True

            gt_trajs: list[np.ndarray] = []
            gt_hist_trajs: list[Optional[np.ndarray]] = []
            # Put ego first (index 0) so it gets highlighted.
            ego_fut = future_xy[0].detach().cpu().numpy()
            if _valid_traj(ego_fut):
                gt_trajs.append(np.stack([ego_fut[:, 0], -ego_fut[:, 1]], axis=-1))
                gt_hist_trajs.append(ego_hist)

            for car_idx in range(1, future_xy.shape[0]):
                car_fut = future_xy[car_idx].detach().cpu().numpy()  # (T,2)
                if not _valid_traj(car_fut):
                    continue
                car_plot = np.stack([car_fut[:, 0], -car_fut[:, 1]], axis=-1)
                gt_trajs.append(car_plot)

                h = _history_polyline(car_idx)
                if h is not None:
                    gt_hist_trajs.append(h)

            if len(gt_trajs) > 0:
                generate_gt_traj_video(
                    background_image=observation_site.background,
                    gt_future_trajs=gt_trajs,
                    history_trajs=gt_hist_trajs,
                    ortho_px_to_meter=ortho_px_to_meter,
                    output_dir=cur_dir,
                    name="gt_trajs",
                    fps=10,
                    zoom_factor=1.8,
                )

        trajs = []

        # Resolve ego velocity for cost computation.
        # - Default: use last observed feature velocity (constant).
        # - Supports GT future / external trajectory differencing / per-frame overrides.
        frame_indices_list: Optional[List[int]] = None
        if frame_indices is not None:
            frame_indices_list = [int(x) for x in frame_indices]
            if len(frame_indices_list) == 0:
                raise ValueError("frame_indices must be non-empty when provided")

        # Determine full horizon length from model (for slicing velocities)
        # Note: compute_px uses `model.seq_len` when frame_indices is None.
        full_horizon = int(model.seq_len)
        dt_cost = 0.04 * float(cost_step)

        # Backward compatibility: `given_v` implies manual constant (or per-frame) ego velocity.
        if given_v is not None:
            ego_velocity_source = "manual"

        ego_velocity_seq: Optional[torch.Tensor] = None  # (seq_len,2)

        src = (ego_velocity_source or "feature_last").lower().strip()
        if src in {"manual", "given", "given_v"}:
            v = torch.as_tensor(given_v, dtype=torch.float32, device=device)
            seq_len_now = len(frame_indices_list) if frame_indices_list is not None else full_horizon
            ego_velocity_seq = _expand_or_slice_velocity(
                v,
                seq_len=seq_len_now,
                frame_indices=frame_indices_list,
            )

        elif src in {"feature_last", "feat_last", "feature", "feat"}:
            # feature: (batch, max_num_cars, history_len, feature_dim)
            ego_feat = feature_batch[0, ego_index, -1].detach().cpu().numpy()
            from datasets.InD import feature_boundaries, denormalize as inD_denormalize

            xy_vel_norm = ego_feat[1:3]  # normalized
            xy_vel_bound = feature_boundaries[1:3]
            xy_vel = inD_denormalize(xy_vel_norm, xy_vel_bound)  # (2,) in m/s
            v = torch.tensor(xy_vel, dtype=torch.float32, device=device)
            seq_len_now = len(frame_indices_list) if frame_indices_list is not None else full_horizon
            ego_velocity_seq = v.view(1, 2).expand(seq_len_now, 2)

        elif src in {"gt_future", "gt", "target", "ground_truth"}:
            # Use ego ground-truth future positions (from target) and finite-diff velocities.
            # target_batch: (batch, pred_len, 2) normalized coords
            ego_future_xy = _denormalize_xy(target_batch[0]).detach()  # (T,2) metric
            ego_last_obs = _denormalize_xy(input_batch[0, ego_index, -1]).detach()  # (2,) metric
            v_full = _velocity_from_positions(ego_future_xy, dt=dt_cost, last_xy=ego_last_obs)
            if frame_indices_list is None:
                if v_full.shape[0] != full_horizon:
                    raise ValueError(
                        f"GT-derived ego velocity length {v_full.shape[0]} != model horizon {full_horizon}. "
                        "Check dataset moving_window/seq_len config."
                    )
                ego_velocity_seq = v_full
            else:
                ego_velocity_seq = _expand_or_slice_velocity(
                    v_full,
                    seq_len=len(frame_indices_list),
                    frame_indices=frame_indices_list,
                )

        elif src in {"traj", "trajectory", "traj_diff", "trajectory_diff"}:
            if ego_velocity_traj is None:
                raise ValueError("ego_velocity_traj is required when ego_velocity_source='traj'")
            traj_xy = torch.as_tensor(ego_velocity_traj, dtype=torch.float32, device=device)
            if traj_xy.dim() != 2 or traj_xy.shape[1] != 2:
                raise ValueError("ego_velocity_traj must be shape (T,2) in meters")
            last_xy = _denormalize_xy(input_batch[0, ego_index, -1]).detach()
            v_full = _velocity_from_positions(traj_xy, dt=dt_cost, last_xy=last_xy)
            if frame_indices_list is None:
                if v_full.shape[0] != full_horizon:
                    raise ValueError(
                        f"Trajectory-derived ego velocity length {v_full.shape[0]} != model horizon {full_horizon}. "
                        "Provide a trajectory of matching length or pass frame_indices to slice."
                    )
                ego_velocity_seq = v_full
            else:
                ego_velocity_seq = _expand_or_slice_velocity(
                    v_full,
                    seq_len=len(frame_indices_list),
                    frame_indices=frame_indices_list,
                )

        else:
            raise ValueError(
                f"Unknown ego_velocity_source='{ego_velocity_source}'. "
                "Use: feature_last | gt_future | traj | manual"
            )

        # Apply per-frame overrides if provided (keys are original horizon indices).
        if ego_velocity_overrides:
            if frame_indices_list is None:
                for t, vv in ego_velocity_overrides.items():
                    tt = int(t)
                    if 0 <= tt < ego_velocity_seq.shape[0]:
                        ego_velocity_seq[tt] = torch.as_tensor(vv, dtype=torch.float32, device=device)
            else:
                # Map original horizon index -> local index
                idx_map = {int(orig_t): local_i for local_i, orig_t in enumerate(frame_indices_list)}
                for t, vv in ego_velocity_overrides.items():
                    tt = int(t)
                    if tt in idx_map:
                        ego_velocity_seq[idx_map[tt]] = torch.as_tensor(vv, dtype=torch.float32, device=device)

        # Collect px for each car by rotating the car to index 0, skip ego_index
        px_list = []

        linspace = torch.linspace(0, 1, steps)
        xg, yg = torch.meshgrid(linspace, linspace, indexing="ij")
        grid = torch.stack((xg.flatten(), yg.flatten()), dim=-1).to(device)
        denormalized_grid = observation_site.denormalize(grid.cpu().numpy())

        fi_tag = _frame_indices_cache_tag(frame_indices_list)

        for car_idx in range(num_cars):
            # Skip ego car
            # if car_idx == ego_index and not pred_cost:
            #     continue
            # Skip empty or nan cars
            car_data = input_batch[0, car_idx].cpu().numpy()
            if np.all(car_data == 0) or np.isnan(car_data).all():
                continue

            cache_name = f"px_ego{ego_id}_idx{car_idx}_start{s_frame}_steps{steps}{fi_tag}.pt"
            cache_path = os.path.join(cache_dir, cache_name)

            if os.path.exists(cache_path):
                print(f"  Loading cached px for car {car_idx} from {cache_name}")
                px_i = torch.load(cache_path, map_location=device)
            else:
                # Swap car_idx to position 0
                perm = [car_idx] + [j for j in range(num_cars) if j != car_idx]
                input_perm = input_batch[:, perm, :, :]
                feature_perm = feature_batch[:, perm, :, :]
                type_perm = type_batch[:, perm]
                t0 = time.time()
                # Note: apply_kernel
                px_i = compute_px(
                    model,
                    input_perm,
                    feature_perm,
                    type_perm,
                    grid,
                    frame_indices=frame_indices_list,
                )
                t1 = time.time()
                print(f"  compute_px for car {car_idx}: {t1-t0:.3f} s")
                torch.save(px_i, cache_path)

            v_field_i = None
            if show_velocity_field:
                v_field_cache = (
                    f"vf_ego{ego_id}_idx{car_idx}_start{s_frame}_steps{steps}{fi_tag}.pt"
                )
                v_field_path = os.path.join(cache_dir, v_field_cache)
                if os.path.exists(v_field_path):
                    v_field_i = torch.load(v_field_path)
                else:
                    perm = [car_idx] + [j for j in range(num_cars) if j != car_idx]
                    v_field_i = compute_velocity_field(
                        model,
                        input_batch[:, perm],
                        feature_batch[:, perm],
                        type_batch[:, perm],
                        grid,
                        frame_indices_list if frame_indices_list is not None else range(px_i.shape[1]),
                    )
                    torch.save(v_field_i, v_field_path)

            if radius > 0:
                num_points, seq_len = px_i.shape
                steps = int(np.sqrt(num_points))
                kernel_size = 2 * radius + 1

                yy, xx = torch.meshgrid(
                    torch.arange(kernel_size), torch.arange(kernel_size), indexing="ij"
                )
                dist = torch.sqrt((xx - radius) ** 2 + (yy - radius) ** 2)
                kernel = (dist <= radius).float().to(device)
                kernel = kernel / kernel.sum()
                kernel = kernel.view(1, 1, kernel_size, kernel_size)

                # (B, C, H, W) -> (seq_len, 1, steps, steps)
                px_2d = px_i.t().reshape(seq_len, 1, steps, steps).to(device)

                px_2d = F.conv2d(px_2d, kernel, padding=radius)
                # (num_points, seq_len)
                px_i = px_2d.reshape(seq_len, num_points).t()

                # px_max = px_i.max(dim=0, keepdim=True).values
                # px_i = px_i / (px_max + 1e-9)
            px_list.append(px_i)  # (num_points, seq_len)

            traj = []
            if not without_traj or car_idx == ego_index:
                # For ego, prefer plotting the *ground-truth future* (if provided by dataset)
                # rather than the centroid-estimated trajectory from px.
                if car_idx == ego_index and future_batch is not None:
                    try:
                        ego_fut_xy = _denormalize_xy(future_batch[0, ego_index]).detach().cpu().numpy()
                        if not (np.isnan(ego_fut_xy).all() or np.all(ego_fut_xy == 0)):
                            traj = ego_fut_xy
                        else:
                            traj = get_px_centroid(px_i, torch.from_numpy(denormalized_grid))
                    except Exception:
                        traj = get_px_centroid(px_i, torch.from_numpy(denormalized_grid))
                else:
                    traj = get_px_centroid(px_i, torch.from_numpy(denormalized_grid))
                trajs.append(traj)
                traj = [traj]  # wrap in list for consistent plotting

            if all_cars:
                car_dir = os.path.join(cur_dir, f"car_{car_idx}")
                makedir(car_dir)
                generate_video(
                    observation_site.background,
                    denormalized_grid,
                    px_i,
                    prob_threshold,
                    traj,
                    [],
                    ortho_px_to_meter,
                    steps,
                    car_dir,
                    idx * 10 + car_idx,
                    velocity_field=v_field_i,
                )

        if not px_list:
            continue  # No valid cars

        # Risk = sum(Px_i * Cost_ego_i) for each car i except ego
        num_points, seq_len = px_list[0].shape
        risk_combined = torch.zeros((num_points, seq_len), device=px_list[0].device)
        for px_i in px_list:
            # (num_points, seq_len)
            if px_i is px_list[ego_index]:
                continue
            if pred_cost:
                cost_map = compute_cost_pred(
                    px_list[ego_index],
                    px_i,
                    torch.from_numpy(denormalized_grid).to(device),
                    step=2,
                )
            else:
                cost_map = compute_cost(
                    ego_velocity_seq,
                    px_i,
                    torch.from_numpy(denormalized_grid).to(device),
                    step=cost_step,
                )
            risk_combined += px_i * cost_map

        generate_video(
            observation_site.background,
            denormalized_grid,
            risk_combined.cpu(),
            prob_threshold,
            trajs,
            [],
            # [],
            ortho_px_to_meter,
            steps,
            cur_dir,
            idx,
            is_risk=True,
        )


if __name__ == "__main__":
    from model.RiskFlow import RiskFlow

    from riskflow_config import preset, seed_everything

    cfg = preset("vis_field")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(cfg["seed"])

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
    observation_site = ind.observation_site_08

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

    traj_flow.eval()

    model_name = "multi_trajflow_ind_2.pt"
    model_path = os.path.join("serialized", model_name)
    traj_flow.load_state_dict(torch.load(model_path, map_location=device))

    visualize(
        observation_site=observation_site,
        model=traj_flow,
        # indices=[1045, 406, 515, 1017, 1164, 1072, 363, 1648],
        indices=[363, 1072],
        steps=128,
        prob_threshold=0.005,
        # given_v=[7.6, -8.3],
        output_dir="videos/combined",
        device=device,
        without_traj=True,
        pred_cost=True,
        all_cars=False,
        radius=0,
        show_velocity_field=False,
        ground_truth_future=True,
    )
