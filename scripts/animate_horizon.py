"""Horizon-evolution animation for a single observed frame.

From one observation ("current frame"), the model forecasts a K-step risk
field. This animation plays that forecast:

  - frame 0     : current frame -- every agent at its observed position;
  - frames 1..K : the predicted risk field Risk_k(g) evolving over the
                  horizon, with every agent (the ego included) drawn as a
                  dynamic agent moving along its trajectory;
  - final frame : all agents return to the current-frame position (loops).

Coordinate registration follows the official drone-dataset-tools
(tracks_import.py / visualizer_params.json): a world point maps to the
provided background image by  px = (x / orthoPxToMeter) / scale_down_factor,
py = (-y / orthoPxToMeter) / scale_down_factor,  with scale_down_factor = 12
for InD. Agents are drawn along their GROUND-TRUTH trajectories (real recorded
positions -> always on the map; no outliers); the risk field is the model's
forecast. Output: riskfield_horizon.gif
Env: RF_CKPT (default serialized/riskflow_ind_5.pt), RF_GRID (64), RF_FPS (9).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.ndimage import gaussian_filter, convolve as ndi_convolve  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from datasets.InD import InD, boundaries_for_location, feature_boundaries  # noqa: E402
from model.RiskFlow import RiskFlow  # noqa: E402
from riskflow_config import default_dict  # noqa: E402

ckpt = os.environ.get("RF_CKPT", "serialized/riskflow_ind_5.pt")
S = int(os.environ.get("RF_GRID", "64"))
FPS = int(os.environ.get("RF_FPS", "9"))
DT = 0.08
M_R = 1500.0 / 2.0         # reduced mass of two 1500 kg vehicles (kg) -> C in J
SG_WIN, SG_POLY = 11, 2    # Savitzky-Golay window/order for smoothed velocity
SCALE_DOWN = 12.0          # InD scale_down_factor (drone-dataset-tools)
DILATE_SIGMA, GAMMA_DISP = 2.0, 0.55
# Min observed-history frames for a neighbour to contribute risk. Below ~30/50
# the model is not robust to missing history (k=0 error ~tens of m; see
# hist_error_scan.py), so its prediction is unreliable and excluded.
MIN_HIST = int(os.environ.get("RF_MIN_HIST", "30"))
# Vehicle bounding-box (length, width) in metres by class. Collision = the two
# ORIENTED boxes intersect (Separating-Axis Test), so a truck is long-but-narrow
# and a car passing alongside it does NOT collide -- fixes the parked-truck
# side-pass false positive that an isotropic disk produced.
VEH_LW = {0: (4.5, 1.9), 1: (10.0, 2.6), 2: (1.8, 0.6), 3: (0.7, 0.7)}

# Single dataset-calibrated scale (risk_calibration.json) drives BOTH the
# colorbar and the benign/critical bands, so criticality is readable directly
# from colour: a cell at the colorbar top (>= T_critical = per-cell p99.9) is in
# the top 0.1% of all risk -> critical. BANDS = (benign<p90, high>p99) in J.
BANDS = None
try:
    with open("risk_calibration.json") as _cf:
        _cal = json.load(_cf)
    VMAX = float(_cal["vmax_recommended"])            # == T_critical (per-cell p99.9)
    BANDS = (_cal.get("band_benign_max"), _cal.get("band_high_min"),
             _cal.get("T_critical", VMAX))
    _cal_note = f"T_critical=p99.9 over {_cal['scenes_used']} scenes"
except (FileNotFoundError, KeyError):
    VMAX, _cal_note = 18.0, "fallback (no calibration file)"
if os.environ.get("RF_VMAX"):                         # manual override for previews
    VMAX = float(os.environ["RF_VMAX"])
    _cal_note = "RF_VMAX override (provisional)"
print(f"RESULT vmax_global = {VMAX:.4g} J  [{_cal_note}]")

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
cmap = matplotlib.colormaps["inferno"]


def base_logpx(z, det):
    d = z.shape[-1]
    return -0.5 * (z.pow(2).sum(-1) + d * np.log(2 * np.pi)) - det


def _cond_grid(x, feat, vt, a):
    """Per-frame flow conditioning of agent a, broadcast to the grid (G,K,E)."""
    xr = torch.roll(x, -a, 1); fr = torch.roll(feat, -a, 1); vr = torch.roll(vt, -a, 1)
    emb, _ = m.encoder(None, torch.cat([xr, fr], -1), vr, per_agent=False)
    cond = m._flow_condition(emb, K, None)
    if cond.dim() == 2:
        cond = cond.unsqueeze(1).expand(-1, K, -1)
    return cond.expand(G, K, cond.shape[-1]).contiguous()


def agent_density(x, feat, vt, a):
    condG = _cond_grid(x, feat, vt, a)
    y = grid.view(G, 1, 2).expand(G, K, 2).contiguous()
    z, det = m.flow(y, condG, sampling_frequency=1)
    P = base_logpx(z, det).exp()
    return P / P.sum(0, keepdim=True).clamp(min=1e-9)


def centroid_vel(P, scale_t, lo_t):
    """Smoothed per-step velocity (m/s) of an agent's predicted-density centroid.
    cen(k) = sum_g g * P_j(g,k) is the expected position; its time-derivative is
    the velocity. We use a Savitzky-Golay derivative (local polynomial slope in
    a sliding window) instead of raw consecutive differencing -- this denoises
    the centroid jitter / multimodal wobble that otherwise produced spurious
    huge velocities, so no velocity cap is needed."""
    cen = (torch.einsum("gd,gk->kd", grid, P) * scale_t + lo_t)  # (K,2) metres
    c = cen.detach().cpu().numpy()
    w = min(SG_WIN, c.shape[0] if c.shape[0] % 2 else c.shape[0] - 1)
    if w >= SG_POLY + 2:
        v = savgol_filter(c, window_length=w, polyorder=SG_POLY, deriv=1,
                          delta=DT, axis=0)
    else:
        v = np.gradient(c, DT, axis=0)
    return torch.tensor(v, dtype=torch.float32, device=P.device)  # (K,2) m/s


def centroid_path(P, scale_t, lo_t):
    """Predicted-position path (K,2) metres = density centroid per horizon step."""
    cen = torch.einsum("gd,gk->kd", grid, P) * scale_t + lo_t
    return cen.detach().cpu().numpy()


_HS_AVG = np.array([[1, 2, 1], [2, 0, 2], [1, 2, 1]], np.float32) / 12.0


def horn_schunck(I1, I2, alpha=0.5, n_iter=40):
    """Horn-Schunck optical flow between two density maps -> (u, w) displacement
    (cells/step) solving the continuity/brightness-constancy eqn with a global
    smoothness prior. Densities are normalized so the data term is well-scaled."""
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
    """Per-cell velocity field (S,S,K,2) in m/s from the agent's density evolution
    P(.,k)->P(.,k+1) via optical flow -- captures multimodal / location-dependent
    motion that a single centroid velocity cannot. Capped to a physical speed."""
    arr = P.reshape(S, S, K).detach().cpu().numpy()
    cx, cy = float(scale_t[0]) / S, float(scale_t[1]) / S
    v = np.zeros((S, S, K, 2), np.float32)
    for k in range(K - 1):
        u, w = horn_schunck(arr[:, :, k], arr[:, :, k + 1])
        v[:, :, k, 0] = u * cx / DT
        v[:, :, k, 1] = w * cy / DT
    v[:, :, K - 1] = v[:, :, K - 2]
    np.clip(v, -30.0, 30.0, out=v)                            # physical cap (m/s)
    return v


def agent_class(vt, a):
    return int(vt[0, a].reshape(-1)[0])


def heading_seq(v_np, fallback):
    """Per-step orientation (rad): motion direction where moving (|v|>0.5 m/s),
    else the observed heading `fallback` (rad). v_np: (K,2) m/s."""
    sp = np.hypot(v_np[:, 0], v_np[:, 1])
    th = np.arctan2(v_np[:, 1], v_np[:, 0])
    th[sp < 0.5] = fallback
    return th


def _sat_mask(dxm, dym, the, hle, hwe, thj, hlj, hwj):
    """Boolean mask: do an ego box (half-len hle, half-wid hwe, heading `the`) at
    the origin and an agent box (hlj,hwj, heading `thj`) at offset (dxm,dym)
    intersect?  Separating-Axis Test over the 4 box-edge normals."""
    ue = (np.cos(the), np.sin(the)); ve = (-np.sin(the), np.cos(the))
    uj = (np.cos(thj), np.sin(thj)); vj = (-np.sin(thj), np.cos(thj))
    sep = np.zeros(dxm.shape, dtype=bool)
    for n in (ue, ve, uj, vj):
        re = hle * abs(ue[0] * n[0] + ue[1] * n[1]) + hwe * abs(ve[0] * n[0] + ve[1] * n[1])
        rj = hlj * abs(uj[0] * n[0] + uj[1] * n[1]) + hwj * abs(vj[0] * n[0] + vj[1] * n[1])
        sep |= np.abs(dxm * n[0] + dym * n[1]) > (re + rj)
    return ~sep


def box_overlap(P, th_e, th_j, dims_e, dims_j, scale_t):
    """(G,K) collision probability: P(ego box at cell g intersects agent-j box),
    = sum_{g'} P_j(g') * 1[boxes overlap], via per-step oriented-box SAT kernels.
    dims = (half_length, half_width) metres; th_* are (K,) headings (rad)."""
    dx, dy = float(scale_t[0]) / S, float(scale_t[1]) / S
    reach = dims_e[0] + dims_j[0]                               # max centre dist (m)
    kx, ky = max(int(np.ceil(reach / dx)), 1), max(int(np.ceil(reach / dy)), 1)
    ii, jj = np.mgrid[-kx:kx + 1, -ky:ky + 1]
    dxm, dym = ii * dx, jj * dy                                 # offset (m)
    arr = P.reshape(S, S, K).detach().cpu().numpy()
    out = np.empty_like(arr)
    cache = {}
    for k in range(K):
        key = (round(float(th_e[k]), 1), round(float(th_j[k]), 1))  # ~6 deg bins
        m = cache.get(key)
        if m is None:
            m = _sat_mask(dxm, dym, th_e[k], dims_e[0], dims_e[1],
                          th_j[k], dims_j[0], dims_j[1]).astype(np.float32)
            cache[key] = m
        out[:, :, k] = ndi_convolve(arr[:, :, k], m, mode="constant")
    return torch.tensor(out.reshape(G, K), device=P.device)


def of_heading_field(vfield, obs_yaw):
    """Per-cell heading (S,S,K) from the optical-flow velocity field; falls back
    to the observed yaw where the agent is ~stationary (|v|<0.5, e.g. parked) so
    the box stays correctly oriented instead of using a noisy zero-velocity angle."""
    sp = np.hypot(vfield[..., 0], vfield[..., 1])
    th = np.arctan2(vfield[..., 1], vfield[..., 0])
    return np.where(sp < 0.5, obs_yaw, th).astype(np.float32)


def box_overlap_oriented(P, P_ego_2d, te_field, tj_field, dims_e, dims_j,
                         scale_t, nbin=8):
    """Per-cell oriented box-overlap collision probability (S,S,K):
       P_ego(g) * P( ego-box@g[heading te(g)] intersects agent-box[heading tj] ).
    Headings are per-cell (optical-flow); binned into nbin angle buckets so it
    stays a handful of convolutions instead of a per-cell kernel."""
    dx, dy = float(scale_t[0]) / S, float(scale_t[1]) / S
    reach = dims_e[0] + dims_j[0]
    kx, ky = max(int(np.ceil(reach / dx)), 1), max(int(np.ceil(reach / dy)), 1)
    ii, jj = np.mgrid[-kx:kx + 1, -ky:ky + 1]
    dxm, dym = ii * dx, jj * dy
    Pj = P.reshape(S, S, K).detach().cpu().numpy()
    edges = np.linspace(-np.pi, np.pi, nbin + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = np.zeros((S, S, K), np.float32)
    kern = {}
    for k in range(K):
        pj = Pj[:, :, k]; pe = P_ego_2d[:, :, k]
        if pj.max() < 1e-12 or pe.max() < 1e-12:
            continue
        be_idx = np.clip(np.digitize(te_field[:, :, k], edges) - 1, 0, nbin - 1)
        bj_idx = np.clip(np.digitize(tj_field[:, :, k], edges) - 1, 0, nbin - 1)
        for be in np.unique(be_idx[pe > pe.max() * 1e-3]):
            emask = (be_idx == be)
            for bj in np.unique(bj_idx[pj > pj.max() * 1e-3]):
                pj_b = pj * (bj_idx == bj)
                if pj_b.sum() < 1e-12:
                    continue
                key = (int(be), int(bj))
                m = kern.get(key)
                if m is None:
                    m = _sat_mask(dxm, dym, centers[be], dims_e[0], dims_e[1],
                                  centers[bj], dims_j[0], dims_j[1]).astype(np.float32)
                    kern[key] = m
                out[:, :, k] += emask * ndi_convolve(pj_b, m, mode="constant")
    return out * P_ego_2d


# RF_SCENE_IDX: pick a specific test-loader batch index (aligned with
# find_scene.py). If unset, fall back to the first scene with >=2 well-observed
# neighbours (history >= MIN_HIST) plus the ego.
SCENE_IDX = int(os.environ["RF_SCENE_IDX"]) if os.environ.get("RF_SCENE_IDX") else None
with torch.no_grad():
    chosen = None
    for i, batch in enumerate(site.test_loader):
        x = batch["input"].to(dev)
        feat = batch["feature"].to(dev)
        vt = batch["type"].to(dev)
        fut = batch["future"].to(dev)
        if SCENE_IDX is not None and i != SCENE_IDX:
            continue
        if torch.isnan(x[:, 0, -2:, :]).any() or torch.isnan(fut[0, 0]).any():
            if SCENE_IDX is not None:
                raise SystemExit(f"scene {SCENE_IDX} has invalid ego")
            continue
        # agents with a fully-valid observed position AND ground-truth future
        valid = [a for a in range(x.shape[1])
                 if not torch.isnan(x[0, a, -1]).any()
                 and not torch.isnan(fut[0, a]).any()]
        well_obs = [a for a in valid if a != 0
                    and int((~torch.isnan(x[0, a, :, 0])).sum()) >= MIN_HIST]
        if SCENE_IDX is not None or len(well_obs) >= 2:
            chosen = (x, feat, vt, fut, int(batch["locationId"].view(-1)[0]), valid, i)
            break
    assert chosen is not None, "no suitable scene found"
    x, feat, vt, fut, loc, valid, scene_i = chosen
    print(f"scene idx={scene_i} loc={loc} agents={valid}")

    bx = boundaries_for_location(loc)
    xlo, xhi = float(bx[0, 0]), float(bx[0, 1])
    ylo, yhi = float(bx[1, 0]), float(bx[1, 1])
    scale = torch.tensor([xhi - xlo, yhi - ylo], device=dev)
    lo = torch.tensor([xlo, ylo], device=dev)

    # ===== Forecast risk EVOLUTION over the K-step (~5s) horizon. We build the
    # combined GT timeline (history+future) so the agents can advance along their
    # GT future while the single forecast's risk field plays out step by step.
    # One forecast = P_ego * P(box overlap) * 1/2 mu |dv|^2 per horizon step.
    Th = x.shape[2]                                            # history length
    Mtot = Th + K                                             # combined timeline
    contributors = [a for a in valid if a != 0
                    and int((~torch.isnan(x[0, a, :, 0])).sum()) >= MIN_HIST]
    pos_comb = torch.cat([x[0], fut[0]], dim=1)               # (N,Mtot,2) normalized
    posm_all = (pos_comb * scale + lo).cpu().numpy()          # (N,Mtot,2) metres
    velm_all = np.gradient(posm_all, DT, axis=1)              # (N,Mtot,2) m/s
    accm_all = np.gradient(velm_all, DT, axis=1)
    head_all = np.degrees(np.arctan2(velm_all[..., 1], velm_all[..., 0])) % 360.0
    feat5 = np.stack([head_all, velm_all[..., 0], velm_all[..., 1],
                      accm_all[..., 0], accm_all[..., 1]], axis=-1)  # (N,Mtot,5)
    fb = feature_boundaries
    feat5n = (feat5 - fb[:, 0]) / (fb[:, 1] - fb[:, 0])      # normalized 0..4
    Cf = feat.shape[-1]                                       # full feature channels (6)
    ch_extra = feat[0, :, -1, 5:Cf].cpu().numpy()            # (N, Cf-5) static attr(s)
    ch_extra = np.repeat(ch_extra[:, None, :], Mtot, axis=1)  # (N,Mtot,Cf-5) held
    featnorm = np.concatenate([feat5n, ch_extra], axis=-1)    # (N,Mtot,Cf)
    featnorm[:, :Th, :] = feat[0].cpu().numpy()              # history = original (exact)
    POS = pos_comb.unsqueeze(0)                               # (1,N,Mtot,2)
    FEATn = torch.tensor(featnorm, dtype=torch.float32, device=dev).unsqueeze(0)
    velm0 = velm_all[0]                                       # (Mtot,2) ego velocity

    def run_forecast(p0):
        """At present p0: K-step risk field (S,S,K) + each agent's predicted
        position-probability density P_a(g,k) -> {a: (S,S,K)}."""
        xw = POS[:, :, p0 - Th + 1:p0 + 1, :].contiguous()
        fw = FEATn[:, :, p0 - Th + 1:p0 + 1, :].contiguous()
        v_ego_fc = torch.tensor(
            np.stack([velm0[min(p0 + 1 + k, Mtot - 1)] for k in range(K)]),
            dtype=torch.float32, device=dev)
        P_ego = agent_density(xw, fw, vt, 0)
        P_ego_2d = P_ego.reshape(S, S, K).cpu().numpy()
        dens = {0: P_ego_2d}                                 # ego position density
        dims_e = tuple(d / 2 for d in VEH_LW.get(agent_class(vt, 0), (4.5, 1.9)))
        v_ego_field = optical_flow_vel(P_ego, scale)         # (S,S,K,2) ego per-cell vel
        te_field = of_heading_field(v_ego_field, float(fw[0, 0, -1, 0]) * 2 * np.pi)
        rf = np.zeros((S, S, K), dtype=np.float32)
        for j in contributors:
            if torch.isnan(xw[0, j]).any():
                continue
            P = agent_density(xw, fw, vt, j)
            dens[j] = P.reshape(S, S, K).cpu().numpy()       # neighbour position density
            dims_j = tuple(d / 2 for d in VEH_LW.get(agent_class(vt, j), (4.5, 1.9)))
            vfield = optical_flow_vel(P, scale)              # (S,S,K,2) m/s per cell
            tj_field = of_heading_field(vfield, float(fw[0, j, -1, 0]) * 2 * np.pi)
            # per-cell oriented box overlap (heading from the OF field per cell)
            pcoll = box_overlap_oriented(P, P_ego_2d, te_field, tj_field,
                                         dims_e, dims_j, scale)   # (S,S,K)
            # location-dependent energy: both velocities per-cell OF fields ->
            # C_j(g,k) = 1/2 mu ||v_ego(g,k) - v_j(g,k)||^2.
            dv = vfield - v_ego_field                        # (S,S,K,2) per-cell
            Cfield = 0.5 * M_R * (dv ** 2).sum(-1)           # (S,S,K) J
            rf += pcoll * Cfield
        for k in range(K):
            rf[:, :, k] = gaussian_filter(rf[:, :, k], DILATE_SIGMA)
        return rf, dens

    p_start, p_end = Th - 1, Mtot - 1
    NEAR = int(os.environ.get("RF_NEAR", "5"))               # GT-advance per cycle
    NCYC = int(os.environ.get("RF_CYCLES", "3"))             # number of replan cycles
    # cycles: freeze at p0 (GT) -> show forecast (transparent predicted paths + risk
    # evolution) -> agents advance NEAR GT frames -> next cycle at p0+NEAR.
    cycles = []                                              # (p0, rf, dens)
    p0 = p_start
    while len(cycles) < NCYC and p0 + NEAR <= p_end:
        rf, dens = run_forecast(p0)
        cycles.append((p0, rf, dens))
        peakk = int(rf.reshape(-1, K).sum(0).argmax())
        print(f"RESULT cycle{len(cycles)} present={p0} (t={(p0-p_start)*DT:.2f}s) "
              f"peak-risk step t+{(peakk+1)*DT:.2f}s  E={float(rf[:,:,peakk].sum()):.3g} J")
        p0 += NEAR
    cloud_agents = [0] + [a for a in contributors]           # agents with a density

# ---- background (official drone-dataset-tools registration) -------------
rec = InD.LOCATION_RECORDINGS[loc][0]
bg, Wm, Hm = None, None, None
try:
    o_raw = float(pd.read_csv(f"data/{rec}_recordingMeta.csv").at[0, "orthoPxToMeter"])
    o = o_raw * SCALE_DOWN                                  # metres per display pixel
    bg = mpimg.imread(f"data/{rec}_background.png")
    Hm, Wm = bg.shape[0] * o, bg.shape[1] * o
    print(f"RESULT bg {bg.shape[1]}x{bg.shape[0]}px  span {Wm:.0f}x{Hm:.0f}m")
except Exception as e:
    print(f"RESULT background unavailable: {e}")

# crop tight to all agents' trajectories, clipped to the image -> no outliers
allpts = posm_all[valid].reshape(-1, 2)
allpts = allpts[~np.isnan(allpts[:, 0])]
pad = 14.0
cx0, cx1 = allpts[:, 0].min() - pad, allpts[:, 0].max() + pad
cy0, cy1 = allpts[:, 1].min() - pad, allpts[:, 1].max() + pad
if bg is not None:
    cx0, cx1 = max(cx0, 0.0), min(cx1, Wm)
    cy0, cy1 = max(cy0, -Hm), min(cy1, 0.0)


def rgba(fr):
    # Map per-cell risk (J) to colour against VMAX = T_critical (per-cell p99.9).
    # Cells >= T_critical saturate (top colour = "critical"); benign scenes stay
    # dim. Same scale as the colorbar -> colour reads directly as criticality.
    d = np.clip(fr / VMAX, 0, 1) ** GAMMA_DISP
    img = cmap(d)
    img[..., 3] = np.clip(d * 1.2, 0, 0.55)   # lower alpha so position clouds show
    return img


AG_PALETTE = [(0.0, 0.85, 1.0), (1.0, 0.55, 0.0), (0.25, 0.95, 0.35),
              (0.95, 0.3, 0.9), (1.0, 0.9, 0.2), (0.6, 0.6, 1.0),
              (1.0, 0.45, 0.45), (0.4, 1.0, 0.85)]   # distinct per-agent hues


def dens_rgba(P2d, rgb, maxa=0.55):
    """Render an agent's position-probability field as a translucent single-hue
    layer: alpha grows with probability (per-frame normalized), colour = agent."""
    m = float(P2d.max())
    d = (P2d / m) ** 0.55 if m > 0 else P2d
    img = np.zeros((P2d.shape[0], P2d.shape[1], 4), dtype=np.float32)
    img[..., 0], img[..., 1], img[..., 2] = rgb
    img[..., 3] = np.clip(d, 0, 1) * maxa
    return img


def box_xy(cx, cy, theta, L, W):
    """4 corners (4,2) of an oriented L x W vehicle box centred at (cx,cy),
    rotated to heading `theta` -- same boxes used by the collision test."""
    hl, hw = L / 2.0, W / 2.0
    c, s = np.cos(theta), np.sin(theta)
    cor = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    rot = np.stack([cor[:, 0] * c - cor[:, 1] * s,
                    cor[:, 0] * s + cor[:, 1] * c], axis=1)
    return rot + np.array([cx, cy])


# per-agent box dims (L,W) and observed-heading fallback (for ~static agents)
veh_dims = {a: VEH_LW.get(agent_class(vt, a), (4.5, 1.9)) for a in valid}
hd_obs = {a: float(feat[0, a, -1, 0]) * 2 * np.pi for a in valid}


def draw_heading(a, p):
    """Heading (rad) of agent a at combined index p: GT velocity dir, else obs."""
    v = velm_all[a, p]
    return float(np.arctan2(v[1], v[0])) if np.hypot(v[0], v[1]) > 0.5 else hd_obs[a]


zero = np.zeros((S, S), dtype=np.float32)
# Cycle plan: freeze at present (2 frames) -> forecast viz (K steps: transparent
# predicted ghosts + risk evolution) -> advance NEAR GT frames -> next cycle.
PLAN = []                                                    # (cycle, phase, step)
for _ci in range(len(cycles)):
    PLAN += [(_ci, "now", 0)] * 2
    PLAN += [(_ci, "fc", _k) for _k in range(K)]
    PLAN += [(_ci, "adv", _d) for _d in range(1, NEAR + 1)]

fig, ax = plt.subplots(figsize=(7.6, 6.8))
if bg is not None:
    ax.imshow(bg, extent=[0, Wm, -Hm, 0], origin="upper", zorder=0)
else:
    ax.set_facecolor("black")
from matplotlib.patches import Polygon  # noqa: E402
AG_RGB = {a: AG_PALETTE[i % len(AG_PALETTE)] for i, a in enumerate(valid)}
# per-agent predicted position-probability layers (translucent, distinct colour)
dens_im = {}
for a in cloud_agents:
    dens_im[a] = ax.imshow(dens_rgba(zero, AG_RGB[a]), extent=[xlo, xhi, ylo, yhi],
                           origin="lower", zorder=2, animated=True)
# risk field on top of the position-probability clouds
im = ax.imshow(rgba(zero.T), extent=[xlo, xhi, ylo, yhi], origin="lower",
               zorder=3, animated=True)
dots, boxes = {}, {}                                         # solid "real" agents
for a in valid:
    col = AG_RGB[a]
    dt, = ax.plot([], [], ("o" if a == 0 else "s"), color=col,
                  ms=(9 if a == 0 else 6), mec="white", alpha=0.95, zorder=6)
    _p = posm_all[a, p_start]
    bx = Polygon(box_xy(_p[0], _p[1], draw_heading(a, p_start), *veh_dims[a]),
                 closed=True, fill=False, edgecolor=col, lw=1.6, alpha=0.9, zorder=5)
    ax.add_patch(bx)
    dots[a] = dt; boxes[a] = bx
ttl = ax.set_title("", fontsize=11)
ax.set_xlim(cx0, cx1); ax.set_ylim(cy0, cy1)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

# Colorbar: same scale as the field (0 -> VMAX = T_critical), gamma-matched, so
# colour reads as criticality. Mark the calibrated benign/high/critical levels.
import matplotlib.colors as mcolors  # noqa: E402


class _GammaNorm(mcolors.Normalize):
    def __call__(self, value, clip=None):
        return np.ma.masked_array(np.clip(value / VMAX, 0, 1) ** GAMMA_DISP)


sm = plt.cm.ScalarMappable(norm=_GammaNorm(0, VMAX), cmap=cmap)
cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
cbar.set_label("expected collision energy  (J)", fontsize=9)
# value ticks (robust to the ~90% zero-mass cells); top = T_critical = "critical"
_ticks = [0.0, 0.25 * VMAX, 0.5 * VMAX, VMAX]
_lab = [f"{t:.0f}" for t in _ticks[:-1]] + [f"CRITICAL\n{VMAX:.0f} J"]
cbar.set_ticks(_ticks); cbar.set_ticklabels(_lab)
cbar.ax.tick_params(labelsize=7)
fig.tight_layout()


def update(f):
    ci, phase, step = PLAN[f]
    p0, rf, dens = cycles[ci]
    present = p0 + step if phase == "adv" else p0            # real-agent GT index
    # risk field + per-agent position-probability clouds only while forecasting
    im.set_data(rgba(rf[:, :, step].T) if phase == "fc" else rgba(zero.T))
    for a in cloud_agents:
        if phase == "fc" and a in dens:
            dens_im[a].set_data(dens_rgba(dens[a][:, :, step].T, AG_RGB[a]))
        else:
            dens_im[a].set_data(dens_rgba(zero, AG_RGB[a]))
    # solid "real" agents at the (frozen during forecast) GT present position
    for a in valid:
        pos = posm_all[a, present]
        if np.isnan(pos[0]):
            dots[a].set_data([], [])
            boxes[a].set_xy(np.full((4, 2), np.nan))
        else:
            dots[a].set_data([pos[0]], [pos[1]])
            boxes[a].set_xy(box_xy(pos[0], pos[1], draw_heading(a, present),
                                   *veh_dims[a]))
    tnow = (present - p_start) * DT
    if phase == "now":
        ttl.set_text(f"Cycle {ci+1}: observe at t = {tnow:4.2f}s -- forecast follows "
                     f"(per-agent position probability in colour).")
    elif phase == "fc":
        ttl.set_text(f"Cycle {ci+1} forecast @ t={(p0-p_start)*DT:.2f}s -- position "
                     f"prob. + risk at horizon t+{(step+1)*DT:4.2f}s/{K*DT:.1f}s.  "
                     f"E = {float(rf[:, :, step].sum()):.2e} J  (vmax = {VMAX:.3g} J).")
    else:
        ttl.set_text(f"Cycle {ci+1}: agents advance on ground truth ... t = {tnow:4.2f}s")
    return ([im, ttl] + list(dens_im.values()) + list(dots.values())
            + list(boxes.values()))


anim = FuncAnimation(fig, update, frames=len(PLAN), interval=1000 // FPS, blit=False)
out = os.environ.get("RF_OUT", "riskfield_horizon.gif")
anim.save(out, writer=PillowWriter(fps=FPS))
print(f"RESULT saved {out}  ({len(PLAN)} frames, {S}x{S}, loc={loc}, "
      f"agents={len(valid)})")
