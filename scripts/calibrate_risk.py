"""Global risk-scale calibration over the full InD test set.

Computes the risk field
    Risk_k(g) = sum_j P_ego(g,k) * P_j(g,k) * C_j(k)
(meeting probability x per-step collision energy; energy applied where they
meet) for EVERY processable test scene, pools the per-cell risk values, reports
the running 95 / 99 / 99.9 percentiles every 250 scenes so the convergence is
visible. The final 99th percentile is the fixed color-scale `vmax` for all
animations/figures (replaces per-frame normalization).

Every scene is processed; within each scene 2000 of the ~S*S*K cells are
uniformly sampled purely to bound memory (~0.5 GB). Uniform sampling -> the
pooled-percentile estimate is unbiased and, from ~10^8 samples, exact to ~5
significant figures.

Output: risk_calibration.json
Env: RF_CKPT (default serialized/riskflow_ind_5.pt), RF_GRID (64),
     RF_PER_SCENE (2000).
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.ndimage import gaussian_filter, convolve as ndi_convolve  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402

from datasets.InD import InD, boundaries_for_location  # noqa: E402
from model.RiskFlow import RiskFlow  # noqa: E402
from riskflow_config import default_dict  # noqa: E402

ckpt = os.environ.get("RF_CKPT", "serialized/riskflow_ind_5.pt")
S = int(os.environ.get("RF_GRID", "64"))
PER_SCENE = int(os.environ.get("RF_PER_SCENE", "2000"))
DT, DILATE_SIGMA = 0.08, 2.0
M_R = 1500.0 / 2.0                                     # reduced mass (kg) -> C in J
SG_WIN, SG_POLY = 11, 2                               # Savitzky-Golay vel smoothing
MIN_HIST = int(os.environ.get("RF_MIN_HIST", "30"))   # min history for a neighbour
VEH_LW = {0: (4.5, 1.9), 1: (10.0, 2.6), 2: (1.8, 0.6), 3: (0.7, 0.7)}  # L,W (m)

c = default_dict()
ind = InD(
    root="data", max_samples=c["maximum_samples"], train_ratio=c["train_ratio"],
    train_batch_size=c["train_batch_size"], test_batch_size=1,
    missing_rate=c["masked_data_ratio"], max_num_cars=c["max_num_cars"],
    max_empty_frames=c["max_empty_frames"], seq_len=c["seq_len"],
    moving_window=c["seq_len"] * 2, sampling_step=c["sampling_step"],
    should_shuffle=False, include_future=c["include_future"],
)
site = ind.observation_site_by_scope("all")
dev = "cuda" if torch.cuda.is_available() else "cpu"

m = RiskFlow(
    seq_len=c["seq_len"], input_dim=c["input_dim"], feature_dim=c["feature_dim"],
    embedding_dim=c["embedding_dim"], hidden_dim=c["hidden_dim"],
    max_num_cars=c["max_num_cars"], num_classes=c["num_classes"],
    gru_layers=c["gru_layers"], num_heads=c["num_heads"], dropout=c["dropout"],
    norm_rotation=c["norm_rotate"], flow_layers=c["flow_layers"],
    flow_hidden_dim=c["flow_hidden_dim"], coupling_layers=c["coupling_layers"],
    use_cnf=c["use_cnf"], use_cgmm=c["use_cgmm"], gmm_modes=c["gmm_modes"],
    use_world_model=True, wm_state_dim=c["wm_state_dim"], action_dim=c["action_dim"],
    scene_level=False,
).to(dev)
m.load_state_dict(torch.load(ckpt, map_location=dev), strict=False)
m.eval()
K = c["seq_len"]

g1 = torch.linspace(0.05, 0.95, S)
GX, GY = torch.meshgrid(g1, g1, indexing="ij")
grid = torch.stack([GX.reshape(-1), GY.reshape(-1)], -1).to(dev)
G = grid.shape[0]
Y_GRID = grid.view(G, 1, 2).expand(G, K, 2).contiguous()


def base_logpx(z, det):
    d = z.shape[-1]
    return -0.5 * (z.pow(2).sum(-1) + d * np.log(2 * np.pi)) - det


def _cond_grid(x, feat, vt, a):
    xr = torch.roll(x, -a, 1); fr = torch.roll(feat, -a, 1); vr = torch.roll(vt, -a, 1)
    emb, _ = m.encoder(None, torch.cat([xr, fr], -1), vr, per_agent=False)
    cond = m._flow_condition(emb, K, None)
    if cond.dim() == 2:
        cond = cond.unsqueeze(1).expand(-1, K, -1)
    return cond.expand(G, K, cond.shape[-1]).contiguous()


def agent_density(x, feat, vt, a):
    z, det = m.flow(Y_GRID, _cond_grid(x, feat, vt, a), sampling_frequency=1)
    P = base_logpx(z, det).exp()
    return P / P.sum(0, keepdim=True).clamp(min=1e-9)


def centroid_vel(P, scale_t, lo_t):
    """Smoothed per-step velocity (m/s) of an agent's predicted-density centroid,
    via a Savitzky-Golay derivative (denoises jitter/multimodal wobble; no cap)."""
    cen = (torch.einsum("gd,gk->kd", grid, P) * scale_t + lo_t)
    c = cen.detach().cpu().numpy()
    w = min(SG_WIN, c.shape[0] if c.shape[0] % 2 else c.shape[0] - 1)
    if w >= SG_POLY + 2:
        v = savgol_filter(c, window_length=w, polyorder=SG_POLY, deriv=1,
                          delta=DT, axis=0)
    else:
        v = np.gradient(c, DT, axis=0)
    return torch.tensor(v, dtype=torch.float32, device=P.device)


def agent_class(vt, a):
    return int(vt[0, a].reshape(-1)[0])


def heading_seq(v_np, fallback):
    sp = np.hypot(v_np[:, 0], v_np[:, 1])
    th = np.arctan2(v_np[:, 1], v_np[:, 0])
    th[sp < 0.5] = fallback
    return th


def _sat_mask(dxm, dym, the, hle, hwe, thj, hlj, hwj):
    ue = (np.cos(the), np.sin(the)); ve = (-np.sin(the), np.cos(the))
    uj = (np.cos(thj), np.sin(thj)); vj = (-np.sin(thj), np.cos(thj))
    sep = np.zeros(dxm.shape, dtype=bool)
    for n in (ue, ve, uj, vj):
        re = hle * abs(ue[0] * n[0] + ue[1] * n[1]) + hwe * abs(ve[0] * n[0] + ve[1] * n[1])
        rj = hlj * abs(uj[0] * n[0] + uj[1] * n[1]) + hwj * abs(vj[0] * n[0] + vj[1] * n[1])
        sep |= np.abs(dxm * n[0] + dym * n[1]) > (re + rj)
    return ~sep


def box_overlap(P, th_e, th_j, dims_e, dims_j, scale_t):
    """(G,K) P(ego box @ g intersects agent-j box) via oriented-box SAT kernels."""
    dx, dy = float(scale_t[0]) / S, float(scale_t[1]) / S
    reach = dims_e[0] + dims_j[0]
    kx, ky = max(int(np.ceil(reach / dx)), 1), max(int(np.ceil(reach / dy)), 1)
    ii, jj = np.mgrid[-kx:kx + 1, -ky:ky + 1]
    dxm, dym = ii * dx, jj * dy
    arr = P.reshape(S, S, K).detach().cpu().numpy()
    out = np.empty_like(arr)
    cache = {}
    for k in range(K):
        key = (round(float(th_e[k]), 1), round(float(th_j[k]), 1))
        m = cache.get(key)
        if m is None:
            m = _sat_mask(dxm, dym, th_e[k], dims_e[0], dims_e[1],
                          th_j[k], dims_j[0], dims_j[1]).astype(np.float32)
            cache[key] = m
        out[:, :, k] = ndi_convolve(arr[:, :, k], m, mode="constant")
    return torch.tensor(out.reshape(G, K), device=P.device)


_HS_AVG = np.array([[1, 2, 1], [2, 0, 2], [1, 2, 1]], np.float32) / 12.0


def horn_schunck(I1, I2, alpha=0.5, n_iter=40):
    s = max(float(I1.max()), float(I2.max()), 1e-12)
    I1 = (I1 / s).astype(np.float32); I2 = (I2 / s).astype(np.float32)
    Ix = 0.5 * (np.gradient(I1, axis=0) + np.gradient(I2, axis=0))
    Iy = 0.5 * (np.gradient(I1, axis=1) + np.gradient(I2, axis=1))
    It = I2 - I1
    denom = alpha ** 2 + Ix ** 2 + Iy ** 2
    u = np.zeros_like(I1); w = np.zeros_like(I1)
    for _ in range(n_iter):
        ubar = ndi_convolve(u, _HS_AVG, mode="nearest")
        wbar = ndi_convolve(w, _HS_AVG, mode="nearest")
        d = (Ix * ubar + Iy * wbar + It) / denom
        u = ubar - Ix * d
        w = wbar - Iy * d
    return u, w


def optical_flow_vel(P, scale_t):
    """Per-cell velocity field (S,S,K,2) m/s from optical-flow transport of the
    density evolution P(.,k)->P(.,k+1). Capped to a physical speed."""
    arr = P.reshape(S, S, K).detach().cpu().numpy()
    cx, cy = float(scale_t[0]) / S, float(scale_t[1]) / S
    v = np.zeros((S, S, K, 2), np.float32)
    for k in range(K - 1):
        u, w = horn_schunck(arr[:, :, k], arr[:, :, k + 1])
        v[:, :, k, 0] = u * cx / DT
        v[:, :, k, 1] = w * cy / DT
    v[:, :, K - 1] = v[:, :, K - 2]
    np.clip(v, -30.0, 30.0, out=v)
    return v


def save_calibration(samples, used, skipped, final):
    """Write the per-cell risk distribution to risk_calibration.json. A SINGLE
    distribution drives both the colorbar and the benign/critical bands, so
    criticality is readable directly from colour: a cell at the colorbar top
    (>= T_critical) is in the top 0.1% of all risk ever seen -> critical.
    Levels (Joules): p50/p90 (benign->elevated), p99 (elevated->high),
    p99.9 = T_critical = vmax. Saved every checkpoint (survives a time-kill)."""
    pool = np.concatenate(samples) if samples else np.zeros(1, np.float32)
    p = np.percentile(pool, [50.0, 90.0, 99.0, 99.9, 99.99])
    res = {
        "checkpoint": ckpt, "grid_S": S, "min_hist": MIN_HIST,
        "vel": "optical-flow transport per-cell velocity field (Horn-Schunck)",
        "scenes_used": used, "scenes_skipped": skipped, "complete": final,
        "pool_size": int(pool.size),
        "units": "joules (expected collision energy, per cell)",
        "cell_p50": float(p[0]), "cell_p90": float(p[1]), "cell_p99": float(p[2]),
        "cell_p99_9": float(p[3]), "cell_p99_99": float(p[4]),
        "cell_max": float(pool.max()),
        # one value for BOTH colorbar ceiling and the critical level:
        "T_critical": float(p[3]),                  # per-cell p99.9 (J)
        "vmax_recommended": float(p[3]),            # == T_critical
        "band_benign_max": float(p[1]),             # below p90 = benign
        "band_high_min": float(p[2]),               # above p99 = high
        "formula": "sum_j P_ego*(P_j*1_disk) * 1/2 mu ||v_ego - v_j(g,k)||^2; "
                   "per-cell optical-flow velocity field (location-dependent C)",
    }
    with open("risk_calibration.json", "w") as f:
        json.dump(res, f, indent=2)
    tag = "FINAL" if final else f"scenes={used}"
    print(f"RESULT {tag} (skip {skipped})  pool={pool.size}  per-cell[J] "
          f"p50={p[0]:.3g} p90={p[1]:.3g} p99={p[2]:.3g} "
          f"p99.9={p[3]:.3g}(=T_crit) p99.99={p[4]:.3g} max={float(pool.max()):.3g}",
          flush=True)


rng = np.random.default_rng(0)
samples = []           # per-cell expected collision energy (J): the one distribution
used = skipped = 0
with torch.no_grad():
    for batch in site.test_loader:
        x = batch["input"].to(dev)
        feat = batch["feature"].to(dev)
        vt = batch["type"].to(dev)
        ft = batch["future"].to(dev)
        loc = int(batch["locationId"].view(-1)[0])
        # need a usable ego (history + future) and at least one neighbour
        if torch.isnan(x[:, 0, -2:, :]).any() or torch.isnan(ft[0, 0]).any():
            skipped += 1
            continue
        neigh = [a for a in range(1, x.shape[1])
                 if not torch.isnan(x[0, a, -1]).any()
                 and int((~torch.isnan(x[0, a, :, 0])).sum()) >= MIN_HIST]
        if not neigh:
            skipped += 1
            continue

        bx = boundaries_for_location(loc)
        scale_t = torch.tensor([float(bx[0, 1] - bx[0, 0]),
                                float(bx[1, 1] - bx[1, 0])],
                               dtype=torch.float32, device=dev)
        lo_t = torch.tensor([float(bx[0, 0]), float(bx[1, 0])], device=dev)

        P_ego = agent_density(x, feat, vt, 0)
        tr0 = ft[0, 0] * scale_t + lo_t
        v_ego = torch.zeros(K, 2, device=dev)
        v_ego[:-1] = (tr0[1:] - tr0[:-1]) / DT
        v_ego[-1] = v_ego[-2]

        # Risk_k(g) = sum_j meet_j(g,k) * C_j(k);  meet_j = P_ego * P_j is the
        # prob. ego AND agent j occupy cell g at step k (the reachability test);
        # C_j(k) = 1/2 m_r ||v_ego(k) - v_j(k)||^2 is the collision energy IF
        # they meet, from the agents' predicted-centroid velocities (per step,
        # not a grid field). MEET_FRAC defines "they meet".
        # Risk_k(g) = sum_j P_ego(g)*(P_j*1_disk)(g) * C_j(g,k); C_j is the
        # LOCATION-DEPENDENT collision energy 1/2 mu ||v_ego(k) - v_j(g,k)||^2,
        # where v_j(g,k) is the per-cell velocity field from optical-flow transport
        # of agent j's density. Box orientation uses the centroid heading.
        dims_e = tuple(d / 2 for d in VEH_LW.get(agent_class(vt, 0), (4.5, 1.9)))
        th_e = heading_seq(v_ego.cpu().numpy(), float(feat[0, 0, -1, 0]) * 2 * np.pi)
        v_ego_field = optical_flow_vel(P_ego, scale_t)             # (S,S,K,2) ego per-cell
        rr = np.zeros((S, S, K), dtype=np.float32)
        for j in neigh:
            P = agent_density(x, feat, vt, j)
            v_j = centroid_vel(P, scale_t, lo_t)
            dims_j = tuple(d / 2 for d in VEH_LW.get(agent_class(vt, j), (4.5, 1.9)))
            th_j = heading_seq(v_j.cpu().numpy(), float(feat[0, j, -1, 0]) * 2 * np.pi)
            pcoll = (P_ego * box_overlap(P, th_e, th_j, dims_e, dims_j, scale_t)
                     ).reshape(S, S, K).cpu().numpy()              # (S,S,K) collision prob
            vfield = optical_flow_vel(P, scale_t)                  # (S,S,K,2) per cell
            dv = vfield - v_ego_field                              # both per-cell
            Cfield = 0.5 * M_R * (dv ** 2).sum(-1)                 # (S,S,K) J per cell
            rr += pcoll * Cfield                                   # expected E (J)
        for k in range(K):
            rr[:, :, k] = gaussian_filter(rr[:, :, k], DILATE_SIGMA)

        flat = rr.reshape(-1)
        idx = rng.choice(flat.shape[0], min(PER_SCENE, flat.shape[0]), replace=False)
        samples.append(flat[idx].astype(np.float32))
        used += 1

        if used % 250 == 0:
            save_calibration(samples, used, skipped, final=False)

save_calibration(samples, used, skipped, final=True)
