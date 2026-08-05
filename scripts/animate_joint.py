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
Two-model field: ego marginal P_ego from ind_8 (single-target + map); the OTHER
agents from ind_7's joint ego-conditioned density p(Y_j|Y_<j, scene, a_ego)
(scene-level AR + map), evaluated exactly on the grid via an AR-MAP rollout.
Env: RF_CKPT_EGO (default ind_8), RF_CKPT_JOINT (default ind_7),
     RF_GRID (64), RF_FPS (9), RF_SCENE_IDX, RF_CYCLES, RF_NEAR.
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

from datasets.registry import get_dataset  # noqa: E402
from model.RiskFlow import RiskFlow  # noqa: E402
from riskflow_config import default_dict  # noqa: E402

_reg = get_dataset()                              # RF_DATASET env (ind|ad4che)
boundaries_for_location = _reg["boundaries_for_location"]
feature_boundaries = _reg["feature_boundaries"]

ckpt_ego = os.environ.get("RF_CKPT_EGO", "serialized/riskflow_ind_8.pt")
ckpt_joint = os.environ.get("RF_CKPT_JOINT", "serialized/riskflow_ind_7.pt")
A_SCALE = 100.0            # ego action proxy scale -- MUST match training
S = int(os.environ.get("RF_GRID", "64"))
FPS = int(os.environ.get("RF_FPS", "9"))
DT = float(os.environ.get("RF_DT", "0.08"))   # InD 0.08; AD4CHE 30fps*step2=0.0667
M_R = 1500.0 / 2.0         # reduced mass of two 1500 kg vehicles (kg) -> C in J
SG_WIN, SG_POLY = 11, 2    # Savitzky-Golay window/order for smoothed velocity
SCALE_DOWN = float(_reg.get("bg_scale_down", 12.0))   # bg-PNG downscale (drone-dataset-tools: InD 12, rounD 10)
DILATE_SIGMA, GAMMA_DISP = 2.0, 0.55
# Min observed-history frames for a neighbour to contribute risk. Below ~30/50
# the model is not robust to missing history (k=0 error ~tens of m; see
# hist_error_scan.py), so its prediction is unreliable and excluded.
MIN_HIST = int(os.environ.get("RF_MIN_HIST", "30"))
VSCALE = float(os.environ.get("RF_VSCALE", "1.0"))   # counterfactual ego-speed scale
# (brake 0.7 / maintain 1.0 / accelerate 1.3) -> scales the closing-energy severity;
# pair with a FIXED colorbar (RF_LOG) to compare maneuvers on one scale.
# Vehicle bounding-box (length, width) in metres by class. Collision = the two
# ORIENTED boxes intersect (Separating-Axis Test), so a truck is long-but-narrow
# and a car passing alongside it does NOT collide -- fixes the parked-truck
# side-pass false positive that an isotropic disk produced.
VEH_LW = _reg["veh_lw"]                                # dataset-aware vehicle dims

# Colorbar scale: plain values in J. Priority: explicit RF_VMAX > RF_CALIB json
# (legacy) > DATA-DERIVED (peak over this animation's forecast cycles, set after
# the cycles are computed). No criticality semantics -- the scale is display-only.
BANDS = None
VMAX = None
_cal_note = "data-derived (peak over forecast cycles)"
if os.environ.get("RF_CALIB"):
    try:
        with open(os.environ["RF_CALIB"]) as _cf:
            _cal = json.load(_cf)
        VMAX = float(_cal["vmax_recommended"])
        _cal_note = f"RF_CALIB p99.9 over {_cal['scenes_used']} scenes (display scale)"
    except (FileNotFoundError, KeyError):
        pass
if os.environ.get("RF_VMAX"):                         # explicit override
    VMAX = float(os.environ["RF_VMAX"])
    _cal_note = "RF_VMAX override"
if VMAX is not None:
    print(f"RESULT vmax_global = {VMAX:.4g} J  [{_cal_note}]")

c = default_dict()
ind = _reg["LoaderClass"](
    root=_reg["root"], max_samples=c["maximum_samples"], train_ratio=c["train_ratio"],
    train_batch_size=c["train_batch_size"], test_batch_size=1,
    missing_rate=c["masked_data_ratio"], max_num_cars=c["max_num_cars"],
    max_empty_frames=c["max_empty_frames"], seq_len=c["seq_len"],
    moving_window=c["seq_len"] * 2, sampling_step=c["sampling_step"],
    should_shuffle=False, include_future=c["include_future"],
)
site = ind.observation_site_by_scope("all")
dev = "cuda" if torch.cuda.is_available() else "cpu"

def _build(scene_level):
    return RiskFlow(
        seq_len=c["seq_len"], input_dim=c["input_dim"], feature_dim=c["feature_dim"],
        embedding_dim=c["embedding_dim"], hidden_dim=c["hidden_dim"],
        max_num_cars=c["max_num_cars"], num_classes=c["num_classes"],
        gru_layers=c["gru_layers"], num_heads=c["num_heads"], dropout=c["dropout"],
        norm_rotation=c["norm_rotate"], flow_layers=c["flow_layers"],
        flow_hidden_dim=c["flow_hidden_dim"], coupling_layers=c["coupling_layers"],
        use_cnf=c["use_cnf"], use_cgmm=c["use_cgmm"], gmm_modes=c["gmm_modes"],
        use_world_model=True, wm_state_dim=c["wm_state_dim"], action_dim=c["action_dim"],
        scene_level=scene_level, use_map=True, map_size=c["map_size"],
        map_data_dir=_reg["map_data_dir"], map_dataset=_reg["map_dataset"],
        map_local=os.environ.get("RF_MAP_LOCAL", "0").lower() in ("1", "true"),
        map_crop_m=c.get("map_crop_m", 40.0), map_raster_res=c.get("map_raster_res", 192),
    ).to(dev)

# Two-model risk field: ind_8 = ego marginal P_ego (single-target + map);
# ind_7 = joint, ego-conditioned others p(Y_j | Y_<j, scene, a_ego) (scene-level
# AR + map). The map is load-bearing (~9-35 nats) so both pass location_id.
m_ego = _build(False)
m_ego.load_state_dict(torch.load(ckpt_ego, map_location=dev), strict=False)
m_ego.eval()
m_joint = _build(True)
m_joint.load_state_dict(torch.load(ckpt_joint, map_location=dev), strict=False)
m_joint.eval()
m = m_ego                  # alias: _cond_grid / agent_density serve the EGO
print(f"RESULT ego={ckpt_ego}  joint={ckpt_joint}")
K = c["seq_len"]

g1 = torch.linspace(0.05, 0.95, S)
GX, GY = torch.meshgrid(g1, g1, indexing="ij")
grid = torch.stack([GX.reshape(-1), GY.reshape(-1)], -1).to(dev)
G = grid.shape[0]
cmap = matplotlib.colormaps["inferno"]

# Single source of truth for the field math (shared with calibrate_joint.py).
from scripts.joint_field import JointRiskField  # noqa: E402
engine = JointRiskField(m_ego, m_joint, grid, S, K, dev, min_hist=MIN_HIST)


def base_logpx(z, det):
    d = z.shape[-1]
    return -0.5 * (z.pow(2).sum(-1) + d * np.log(2 * np.pi)) - det


def _cond_grid(x, feat, vt, a):
    """Per-frame EGO flow conditioning of agent a, broadcast to the grid (G,K,E).
    Adds the map embedding (load-bearing for ind_8) before the world-model roll."""
    xr = torch.roll(x, -a, 1); fr = torch.roll(feat, -a, 1); vr = torch.roll(vt, -a, 1)
    emb, _ = m.encoder(None, torch.cat([xr, fr], -1), vr, per_agent=False)
    memb = m._map_emb(LOC_T, dev)
    if memb is not None:
        emb = emb + memb
    cond = m._flow_condition(emb, K, None)
    if cond.dim() == 2:
        cond = cond.unsqueeze(1).expand(-1, K, -1)
    return cond.expand(G, K, cond.shape[-1]).contiguous()


def agent_density(x, feat, vt, a):
    """EGO per-agent marginal density on the grid (ind_8 + map)."""
    condG = _cond_grid(x, feat, vt, a)
    y = grid.view(G, 1, 2).expand(G, K, 2).contiguous()
    z, det = m.flow(y, condG, sampling_frequency=1)
    P = base_logpx(z, det).exp()
    return P / P.sum(0, keepdim=True).clamp(min=1e-9)


def _joint_cond_density(agent_emb, s_seq, y_fill, order, i):
    """ind_7 conditional density of agent i on the grid, given current y_fill
    (Y_<i filled with earlier agents' MAPs). Returns (G,K) normalized per step."""
    cond = m_joint.ar_decoder(agent_emb, s_seq, y_fill, order)   # (1,N,K,E)
    ci = cond[:, i]                                              # (1,K,E)
    condG = ci.expand(G, K, ci.shape[-1]).contiguous()
    yq = grid.view(G, 1, 2).expand(G, K, 2).contiguous()
    z, det = m_joint.flow(yq, condG, sampling_frequency=1)
    P = base_logpx(z, det).exp()
    return P / P.sum(0, keepdim=True).clamp(min=1e-9)


def joint_densities(xw, fw, P_ego):
    """ind_7 joint ego-conditioned densities for the contributors.

    a_ego is the action proxy of the ego's MAP path (from ind_8's P_ego); the AR
    chain is rolled with each earlier agent's per-frame MAP (ego first, then
    nearest-first neighbours). Returns {j: (G,K)} for j in `contributors`."""
    agent_emb, car_valid = m_joint.encoder(
        None, torch.cat([xw, fw], -1), vt, per_agent=True)       # (1,N,E),(1,N)
    memb = m_joint._map_emb(LOC_T, dev)
    if memb is not None:
        agent_emb = agent_emb + memb.unsqueeze(1)
    # ego MAP path (K,2 normalized) -> action proxy a_ego (1,K,2)
    ego_pos = grid[P_ego.argmax(0)]                              # (K,2)
    a = ego_pos[2:] - 2 * ego_pos[1:-1] + ego_pos[:-2]
    a_ego = torch.cat([a[:1], a, a[-1:]], 0)[None] * A_SCALE     # (1,K,2)
    s_seq = m_joint.world_model.forward_scene(
        agent_emb, car_valid, K, a_ego, return_dyn=False)
    order = m_joint._compute_ordering(xw, car_valid)             # (1,N)
    y_fill = torch.zeros(1, xw.shape[1], K, 2, device=dev)
    y_fill[:, 0] = ego_pos[None]                                 # ego slot = its MAP
    out = {}
    for i in order[0].tolist():
        if i == 0 or not bool(car_valid[0, i]):
            continue
        P = _joint_cond_density(agent_emb, s_seq, y_fill, order, i)
        y_fill[0, i] = grid[P.argmax(0)]                         # fill its MAP
        if i in contributors:
            out[i] = P
    return out


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
            chosen = (x, feat, vt, fut, int(batch["locationId"].view(-1)[0]), valid, i, batch)
            break
    assert chosen is not None, "no suitable scene found"
    x, feat, vt, fut, loc, valid, scene_i, _batch = chosen
    print(f"scene idx={scene_i} loc={loc} agents={valid}")
    # Recover each agent's RECORDED footprint (length,width) without touching the
    # cache (same matching as qualitative_map.py): class-default boxes (e.g. 12 m
    # trucks vs recorded ~7.7 m) make queued AD4CHE vehicles appear to overlap.
    dimsb = None; _site_match = None; _eid = None; _sf = None
    try:
        _eid = int(np.asarray(_batch["trackId"]).reshape(-1)[0])
        _sf = int(np.asarray(_batch["startFrame"]).reshape(-1)[0])
        _ref = np.nan_to_num(x[0, 0].cpu().numpy())
        for _st in getattr(ind, "LOCATION_RECORDINGS", {}).get(loc, []):
            try:
                _ss = ind.get_specific_sample(_st, _eid, _sf)
            except Exception:
                continue
            if np.allclose(np.nan_to_num(_ss["input"][0, 0].cpu().numpy()), _ref, atol=1e-3):
                dimsb = np.asarray(_ss["dims"][0]); _site_match = _st; break
    except Exception as e:  # noqa: BLE001
        print(f"RESULT dims recovery unavailable ({e}); class defaults")
    print(f"RESULT recorded dims {'FOUND' if dimsb is not None else 'NOT found'} "
          f"(site={_site_match})")
    # Raw recording for ALL-agent rendering: the cached window carries only the
    # 8 nearest agents (the model's conditioning limit), but the VIEW should show
    # every vehicle present. Draw all vehicles from the recorded tracks (recorded
    # per-frame heading + per-track dims); clouds/risk still come from the
    # modelled <=8 (that is the field's definition, not a rendering choice).
    RAW = None
    if _site_match is not None:
        try:
            _m2, _tracks2, _tmeta2 = ind._load_and_clean_data(_site_match)
            _lcol, _wcol = ind._dim_columns()
            RAW = dict(
                by_frame={int(f): g for f, g in _tracks2.groupby("frame")},
                dims={int(t): (float(l), float(w)) for t, l, w in
                      zip(_tmeta2["trackId"], _tmeta2[_lcol], _tmeta2[_wcol])},
            )
            print(f"RESULT raw tracks loaded: all vehicles will be rendered", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"RESULT raw tracks unavailable ({e}); cached agents only")

    bx = boundaries_for_location(loc)
    xlo, xhi = float(bx[0, 0]), float(bx[0, 1])
    ylo, yhi = float(bx[1, 0]), float(bx[1, 1])
    # imshow extent for the grid arrays: samples live at normalized 0.05..0.95,
    # NOT the full [0,1] box -- stretching them over [xlo,xhi] shifts rendered
    # mass outward by up to ~4% of the box (~6 m at the edges).
    _g1n = g1.numpy(); _h = (_g1n[1] - _g1n[0]) / 2.0
    GEXT = [xlo + (_g1n[0] - _h) * (xhi - xlo), xlo + (_g1n[-1] + _h) * (xhi - xlo),
            ylo + (_g1n[0] - _h) * (yhi - ylo), ylo + (_g1n[-1] + _h) * (yhi - ylo)]
    scale = torch.tensor([xhi - xlo, yhi - ylo], device=dev)
    lo = torch.tensor([xlo, ylo], device=dev)
    LOC_T = torch.tensor([loc], device=dev)                   # for the map embedding

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
        position-probability density P_a(g,k) -> {a: (S,S,K)}. Delegates the
        field math to the shared engine (identical to calibrate_joint.py).

        For advanced presents (p0 > p_start) the window is RE-FETCHED from the
        dataset (get_specific_sample at the shifted start frame) so every cycle
        conditions on a genuine window, exactly like training/eval. The previous
        spliced window (GT future positions + np.gradient features) made the
        predicted density lag progressively behind the ego (~13 m by cycle 6 --
        it read as the cloud trailing/'reversing'); genuine windows anchor the
        k=0 density at the ego for every cycle."""
        if p0 > Th - 1 and _site_match is not None:
            try:
                sf = _sf + (p0 - (Th - 1)) * int(getattr(ind, "sampling_step", 1))
                ss = ind.get_specific_sample(_site_match, _eid, sf)
                xw2 = ss["input"].to(dev); fw2 = ss["feature"].to(dev)
                vt2 = ss["type"].to(dev)
                contrib2 = [a for a in range(1, xw2.shape[1])
                            if not torch.isnan(xw2[0, a, -1]).any()
                            and int((~torch.isnan(xw2[0, a, :, 0])).sum()) >= MIN_HIST]
                return engine.field(xw2, fw2, vt2, contrib2, LOC_T, scale, return_dens=True,
                                    v_ego_scale=VSCALE, dims_m=np.asarray(ss["dims"][0]))
            except Exception as e:  # noqa: BLE001
                print(f"RESULT re-fetch failed at p0={p0} ({e}); spliced fallback", flush=True)
        xw = POS[:, :, p0 - Th + 1:p0 + 1, :].contiguous()
        fw = FEATn[:, :, p0 - Th + 1:p0 + 1, :].contiguous()
        return engine.field(xw, fw, vt, contributors, LOC_T, scale, return_dens=True,
                            v_ego_scale=VSCALE, dims_m=dimsb)

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
    if VMAX is None:                                         # data-derived display scale
        VMAX = max(max(float(_rf.max()) for _, _rf, _ in cycles), 1e-6)
        print(f"RESULT vmax_global = {VMAX:.4g} J  [{_cal_note}]")
    cloud_agents = [0] + [a for a in contributors]           # agents with a density

    # ---- DISPLAY de-skirt: suppress the diffuse low-probability tail of the
    # risk field. Risk = P_ego . M . C with the severity C ~1e4 J, so by the late
    # horizon (where the ego occupancy over-disperses, its peak prob dropping
    # several-fold) the near-zero P_ego tail still clears the log-display floor and
    # lights up a broad skirt, including grid-edge cells that fall on the off-road
    # border. We zero the displayed risk where the ego is essentially never going
    # to be (P_ego < tau * per-frame max). This is display-only: it does not touch
    # the field used for detection/calibration, and it preserves the risk peak
    # (the peak cell sits at high P_ego). Off by RF_PEGO_FLOOR=0.
    PEGO_FLOOR = float(os.environ.get("RF_PEGO_FLOOR", "0.02"))
    if PEGO_FLOOR > 0:
        for (_p0, _rf, _dens) in cycles:
            Pe = _dens[0]                                    # (S,S,K) ego occupancy
            thr = PEGO_FLOOR * Pe.reshape(-1, Pe.shape[-1]).max(0)   # (K,) per-frame
            _rf[Pe < thr[None, None, :]] = 0.0

    if os.environ.get("RF_DIAG"):
        # Diagnose spurious off-road / isolated risk cells in the display.
        import scipy.ndimage as _ndi
        _lr = os.environ.get("RF_LOG")
        floor = (float(_lr.split(",")[0]) if _lr else 1.0)   # display floor (J)
        for (p0, _rf, _dens) in cycles:
            pk = int(_rf.reshape(-1, K).sum(0).argmax())
            R = _rf[:, :, pk]; Pe = _dens[0][:, :, pk]
            shown = R >= floor                               # cells the display lights up
            ncell = int(shown.sum())
            # main blob = component containing the global-max cell
            lab, _ = _ndi.label(shown)
            mx = np.unravel_index(int(R.argmax()), R.shape)
            main = lab == lab[mx] if lab[mx] else np.zeros_like(shown)
            iso = shown & ~main                              # lit cells off the main blob
            # edge ring (outer 10%) ~ off-photo border
            S_ = R.shape[0]; b = max(1, S_ // 10)
            edge = np.zeros_like(shown); edge[:b] = edge[-b:] = edge[:, :b] = edge[:, -b:] = True
            print(f"DIAG p0={p0} peakk={pk} peakE={R.max():.1f}J shownCells={ncell} "
                  f"isolated={int(iso.sum())} edgeLit={int((shown&edge).sum())} "
                  f"Pego[max={Pe.max():.2e} cellsAtIso={Pe[iso].max() if iso.any() else 0:.2e}]")
            for tau in (0.02, 0.05, 0.1, 0.2, 0.3):
                keep = Pe >= tau * Pe.max()
                R2 = np.where(keep, R, 0.0); shown2 = R2 >= floor
                print(f"      Pego-floor tau={tau}: shownCells={int(shown2.sum())} "
                      f"edgeLit={int((shown2 & edge).sum())} "
                      f"peakKept={'Y' if R2[mx]>=floor else 'N'} peakE={R2.max():.1f}J")
            # --- agent position-probability CLOUDS (dens_rgba) visibility ---
            # alpha = (P/P.max())**0.55 * 0.55 ; "visible" ~ alpha >= 0.08
            S_ = R.shape[0]; bb = max(1, S_ // 10)
            edgeC = np.zeros((S_, S_), bool); edgeC[:bb] = edgeC[-bb:] = edgeC[:, :bb] = edgeC[:, -bb:] = True
            for a in cloud_agents:
                Pc = _dens[a][:, :, pk]; m = float(Pc.max())
                if m <= 0:
                    continue
                alpha = (Pc / m) ** 0.55 * 0.55
                vis = alpha >= 0.08                          # display-visible cells
                lab2, _ = _ndi.label(vis)
                mxc = np.unravel_index(int(Pc.argmax()), Pc.shape)
                mainc = lab2 == lab2[mxc] if lab2[mxc] else np.zeros_like(vis)
                isoc = vis & ~mainc
                print(f"      CLOUD a={a} visCells={int(vis.sum())} isolated={int(isoc.sum())} "
                      f"edgeVis={int((vis & edgeC).sum())} "
                      f"tailFrac@1%={float((Pc>=0.01*m).sum())/ (S_*S_):.3f}")
        raise SystemExit(0)

# ---- background -------------------------------------------------------
bg, Wm, Hm, bg_extent = None, None, None, None
try:
    if _reg["name"] == "ad4che":
        # per-scene map at its true metric registration (centre-origin):
        # x in [-W/2*s, W/2*s], y in [-H/2*s, H/2*s], s = metres/pixel.
        from datasets.AD4CHE import scene_scale
        bg = mpimg.imread(os.path.join(_reg["root"], "maps", f"{loc}.jpg"))
        Hp, Wp = bg.shape[0], bg.shape[1]
        sm = scene_scale(_reg["root"], loc)
        bg_extent = [-Wp / 2 * sm, Wp / 2 * sm, -Hp / 2 * sm, Hp / 2 * sm]
        print(f"RESULT bg ad4che scene {loc} {Wp}x{Hp}px scale={sm} extent={[round(e,1) for e in bg_extent]}")
    else:
        rec = _reg["LoaderClass"].LOCATION_RECORDINGS[loc][0]
        o_raw = float(pd.read_csv(os.path.join(_reg["root"], f"{rec}_recordingMeta.csv"))
                      .at[0, "orthoPxToMeter"])
        o = o_raw * SCALE_DOWN                              # metres per display pixel
        bg = mpimg.imread(os.path.join(_reg["root"], f"{rec}_background.png"))
        Hm, Wm = bg.shape[0] * o, bg.shape[1] * o
        bg_extent = [0, Wm, -Hm, 0]
        print(f"RESULT bg {bg.shape[1]}x{bg.shape[0]}px  span {Wm:.0f}x{Hm:.0f}m")
except Exception as e:
    print(f"RESULT background unavailable: {e}")

# crop tight to all agents' trajectories, clipped to the background extent
allpts = posm_all[valid].reshape(-1, 2)
allpts = allpts[~np.isnan(allpts[:, 0])]
pad = 14.0
cx0, cx1 = allpts[:, 0].min() - pad, allpts[:, 0].max() + pad
cy0, cy1 = allpts[:, 1].min() - pad, allpts[:, 1].max() + pad
if bg_extent is not None:
    cx0, cx1 = max(cx0, bg_extent[0]), min(cx1, bg_extent[1])
    cy0, cy1 = max(cy0, bg_extent[2]), min(cy1, bg_extent[3])
# Force a FIXED crop aspect ratio so every scene fills the figure the same way.
# With set_aspect("equal"), a wide crop yields a short axes (and a short
# colorbar), so videos of different scenes end up different content heights and
# do not align side by side. Expanding the deficient dimension to a constant
# width:height ratio makes the drawn frame (and colorbar) identical across all
# animations; off-photo expansion just shows the black map border. RF_CROP_ASPECT=0
# restores the old tight (variable) crop.
_ar = float(os.environ.get("RF_CROP_ASPECT", "1.18"))
if _ar > 0:
    cw, ch = cx1 - cx0, cy1 - cy0
    if cw / ch < _ar:                       # too tall -> widen
        nw = ch * _ar; cxc = 0.5 * (cx0 + cx1); cx0, cx1 = cxc - nw / 2, cxc + nw / 2
    else:                                   # too wide -> heighten
        nh = cw / _ar; cyc = 0.5 * (cy0 + cy1); cy0, cy1 = cyc - nh / 2, cyc + nh / 2

# ---- off-map display mask -------------------------------------------------
# The predicted occupancy over-disperses at the late horizon (its peak prob can
# drop several-fold), and since risk = P_ego . M . C with C ~1e4 J, that diffuse
# tail still clears the log-display floor and lights up a skirt -- including grid
# cells that fall on the off-road black border of the drone photo. ONMAP[i,j]
# (in the rf (x=i, y=j) convention) is True only where the background image has
# actual content (non-black), so the overlays render strictly on the photo.
# Display-only; does not touch the field used for detection/calibration.
ONMAP = np.ones((S, S), bool)
if bg is not None and os.environ.get("RF_MAPMASK", "1") != "0":
    _bg = bg.astype(np.float32)
    if _bg.max() > 1.5:
        _bg = _bg / 255.0
    _lum = _bg[..., :3].mean(-1) if _bg.ndim == 3 else _bg
    Hpx, Wpx = _lum.shape[0], _lum.shape[1]
    _gc = (_g1n if 'g1n' not in dir() else _g1n)             # cell-centre normals 0.05..0.95
    _gc = g1.numpy()
    mx = xlo + _gc * (xhi - xlo)                             # (S,) metric x per i
    my = ylo + _gc * (yhi - ylo)                             # (S,) metric y per j
    ex0, ex1, ey0, ey1 = bg_extent
    col = ((mx[:, None] - ex0) / (ex1 - ex0) * Wpx).astype(int)        # (S,1)->broadcast
    row = ((ey1 - my[None, :]) / (ey1 - ey0) * Hpx).astype(int)        # origin=upper
    col = np.broadcast_to(col, (S, S)); row = np.broadcast_to(row, (S, S))
    inb = (col >= 0) & (col < Wpx) & (row >= 0) & (row < Hpx)
    cc = np.clip(col, 0, Wpx - 1); rr = np.clip(row, 0, Hpx - 1)
    ONMAP = inb & (_lum[rr, cc] > 0.06)                      # non-black photo content
    print(f"RESULT off-map mask: {int((~ONMAP).sum())}/{S*S} cells masked off-photo", flush=True)
ONMAP_T = ONMAP.T                                           # display arrays are .T'd


LOGRNG = ([float(t) for t in os.environ["RF_LOG"].split(",")]
          if os.environ.get("RF_LOG") else None)   # unified cross-dataset log scale


def rgba(fr):
    # Map per-cell risk (J) to colour. RF_LOG="vmin,vmax" -> unified log scale
    # (same energy = same colour across all datasets); otherwise linear vs VMAX
    # (RF_VMAX or this animation's forecast peak).
    fr = np.nan_to_num(np.asarray(fr, dtype=np.float32), nan=0.0)  # NaN cell -> transparent
    if LOGRNG:
        lo, hi = LOGRNG
        d = np.clip((np.log10(np.clip(fr, lo, hi)) - np.log10(lo))
                    / (np.log10(hi) - np.log10(lo)), 0, 1)
        d[fr < lo] = 0.0
    else:
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
    # Hide the diffuse low-probability tail: when the occupancy over-disperses
    # (late horizon / fast circulating motion) the faint skirt otherwise bleeds
    # past the road. Show only the confident part of the distribution.
    cfloor = float(os.environ.get("RF_CLOUD_FLOOR", "0.25"))
    d = np.where(d < cfloor, 0.0, d)
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
# recorded per-track footprint when recovered; class default otherwise
veh_dims = {a: ((float(dimsb[a, 0]), float(dimsb[a, 1]))
                if (dimsb is not None and float(dimsb[a, 0]) > 0.5)
                else VEH_LW.get(agent_class(vt, a), (4.5, 1.9))) for a in valid}
hd_obs = {a: float(feat[0, a, -1, 0]) * 2 * np.pi for a in valid}


def draw_heading(a, p):
    """Heading (rad) of agent a at combined index p: GT velocity dir, else obs.
    AD4CHE: ALWAYS the recorded heading -- velocity direction is noisy for the
    slow congested vehicles and mis-orients the long truck boxes (the road is
    straight, so the recorded heading stays valid over the window)."""
    if _reg["name"] == "ad4che":
        return hd_obs[a]
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
    ax.imshow(bg, extent=bg_extent, origin="upper", zorder=0)
else:
    ax.set_facecolor("black")
from matplotlib.patches import Polygon  # noqa: E402
# ego (index 0) in cyan; ALL surrounding agents share one colour (orange),
# matching the qualitative figures' ego-vs-others scheme, not a per-agent palette.
AG_RGB = {a: (AG_PALETTE[0] if a == 0 else AG_PALETTE[1]) for a in valid}
# Per-agent predicted position-probability layers. One layer per SLOT (not per
# original-batch agent): re-fetched cycle windows re-order neighbours by
# distance, so dens keys can be any index 0..N-1. Colour by slot (0=ego cyan).
dens_im = {}
for a in range(x.shape[1]):
    dens_im[a] = ax.imshow(dens_rgba(zero, AG_PALETTE[0] if a == 0 else AG_PALETTE[1]),
                           extent=GEXT,
                           origin="lower", zorder=2, animated=True)
# risk field on top of the position-probability clouds
im = ax.imshow(rgba(zero.T), extent=GEXT, origin="lower",
               zorder=3, animated=True)
dots, boxes = {}, {}                                         # solid "real" agents
raw_artists = []                                             # per-frame ALL-vehicle artists
if RAW is None:                                              # fallback: cached <=8 agents only
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


_cbnorm = (mcolors.LogNorm(LOGRNG[0], LOGRNG[1]) if LOGRNG
           else _GammaNorm(0, VMAX))
sm = plt.cm.ScalarMappable(norm=_cbnorm, cmap=cmap)
# colorbar height locked to the image axes (make_axes_locatable tracks the
# aspect-equal axes box), instead of a free-floating fig.colorbar.
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402
_cax = make_axes_locatable(ax).append_axes("right", size="3.5%", pad=0.1)
cbar = fig.colorbar(sm, cax=_cax)
cbar.set_label("expected collision energy  (J)", fontsize=9)
# plain value ticks in J -- no criticality marker (display scale only)
if LOGRNG:
    _ticks = [10.0 ** e for e in range(int(np.ceil(np.log10(LOGRNG[0]))),
                                       int(np.floor(np.log10(LOGRNG[1]))) + 1)]
    _lab = [f"{t:g}" for t in _ticks]
else:
    _ticks = [0.0, 0.25 * VMAX, 0.5 * VMAX, VMAX]
    _lab = [f"{t:.0f}" for t in _ticks]
cbar.set_ticks(_ticks); cbar.set_ticklabels(_lab)
cbar.ax.tick_params(labelsize=7)
fig.tight_layout()


def update(f):
    ci, phase, step = PLAN[f]
    p0, rf, dens = cycles[ci]
    present = p0 + step if phase == "adv" else p0            # real-agent GT index
    # risk field + per-agent position-probability clouds only while forecasting
    im.set_data(rgba(np.where(ONMAP_T, rf[:, :, step].T, 0.0)) if phase == "fc"
                else rgba(zero.T))
    for a in dens_im:
        col = AG_PALETTE[0] if a == 0 else AG_PALETTE[1]
        if phase == "fc" and a in dens:
            dens_im[a].set_data(dens_rgba(np.where(ONMAP_T, dens[a][:, :, step].T, 0.0), col))
        else:
            dens_im[a].set_data(dens_rgba(zero, col))
    # solid "real" agents at the (frozen during forecast) GT present position.
    # With raw tracks available, draw EVERY vehicle recorded at this frame
    # (recorded heading + dims), not just the <=8 the model conditions on.
    if RAW is not None:
        for art in raw_artists:
            art.remove()
        raw_artists.clear()
        f_raw = _sf + present * int(getattr(ind, "sampling_step", 1))
        g = RAW["by_frame"].get(int(f_raw))
        if g is not None:
            for _, r in g.iterrows():
                xpos, ypos = float(r["xCenter"]), float(r["yCenter"])
                if not (cx0 - 12 < xpos < cx1 + 12 and cy0 - 12 < ypos < cy1 + 12):
                    continue
                tid = int(r["trackId"])
                col = AG_PALETTE[0] if tid == _eid else AG_PALETTE[1]
                th = np.radians(float(r["heading"]))
                L, W = RAW["dims"].get(tid, (4.5, 1.9))
                pol = Polygon(box_xy(xpos, ypos, th, L, W), closed=True, fill=False,
                              edgecolor=col, lw=1.6, alpha=0.9, zorder=5)
                ax.add_patch(pol); raw_artists.append(pol)
                dot, = ax.plot([xpos], [ypos], ("o" if tid == _eid else "s"), color=col,
                               ms=(9 if tid == _eid else 5), mec="white", alpha=0.95, zorder=6)
                raw_artists.append(dot)
    else:
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
    _TITLE = os.environ.get("RF_TITLE")          # fixed label (e.g. counterfactual maneuver)
    if _TITLE:
        ttl.set_text(_TITLE)
    elif phase == "now":
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


if os.environ.get("RF_SAVEFRAME"):
    # Visual-verify the off-map mask: render the peak-risk forecast frame of the
    # highest-energy cycle to a PNG and exit (no full animation).
    _bestci = int(np.argmax([float(_rf.reshape(-1, K).sum(0).max()) for _, _rf, _ in cycles]))
    _base = os.environ["RF_SAVEFRAME"]
    # save a strip across the horizon (catch off-map clouds at any frame), one
    # PNG per quartile step of the best cycle.
    for _st in sorted(set([0, K // 4, K // 2, 3 * K // 4, K - 1])):
        _f = next((fi for fi, (ci, ph, st) in enumerate(PLAN)
                   if ci == _bestci and ph == "fc" and st == _st), None)
        if _f is None:
            continue
        update(_f)
        _outp = _base.replace(".png", f"_k{_st}.png")
        # full canvas (NOT bbox_inches=tight) so the preview matches the video frame
        fig.savefig(_outp, dpi=120)
        print(f"RESULT saved {_outp} (cycle{_bestci} step{_st})")
    raise SystemExit(0)

anim = FuncAnimation(fig, update, frames=len(PLAN), interval=1000 // FPS, blit=False)
out = os.environ.get("RF_OUT", "riskfield_horizon.gif")
anim.save(out, writer=PillowWriter(fps=FPS))
print(f"RESULT saved {out}  ({len(PLAN)} frames, {S}x{S}, loc={loc}, "
      f"agents={len(valid)})")
