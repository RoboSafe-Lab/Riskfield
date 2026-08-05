"""Term-by-term breakdown of the meeting-probability x collision-energy risk.

Risk_k(g) = sum_j meet_j(g,k) * C_j(k)
  meet_j(g,k) = P_ego(g,k) * P_j(g,k)         prob. ego AND agent j meet at g
  C_j(k)      = 1/2 m_r ||v_ego(k) - v_j(k)||^2   collision energy IF they meet
  v_j(k)      = d/dt centroid(P_j)            model-predicted agent velocity

The energy is a per-step scalar (the two agents' relative predicted motion),
applied only where they meet. For a chosen scene this prints, at several query
cells g (in front of / at / behind the ego, risk argmax) and horizon steps k,
every factor: P_ego, P_j, the joint meeting probability, |dv|, C_j(k).

Env: RF_CKPT (default serialized/riskflow_ind_5.pt), RF_GRID (64).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.ndimage import convolve as ndi_convolve  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402

from datasets.InD import InD, boundaries_for_location  # noqa: E402
from model.RiskFlow import RiskFlow  # noqa: E402
from riskflow_config import default_dict  # noqa: E402

ckpt = os.environ.get("RF_CKPT", "serialized/riskflow_ind_5.pt")
S = int(os.environ.get("RF_GRID", "64"))
DT = 0.08
M_R = 1500.0 / 2.0                                    # reduced mass (kg) -> C in J
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
    condG = _cond_grid(x, feat, vt, a)
    y = grid.view(G, 1, 2).expand(G, K, 2).contiguous()
    z, det = m.flow(y, condG, sampling_frequency=1)
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


def nearest_cell(norm_xy):
    i = int(torch.argmin((g1 - float(norm_xy[0])).abs()))
    j = int(torch.argmin((g1 - float(norm_xy[1])).abs()))
    return i * S + j


CLASS_NAME = {0: "car", 1: "truck_bus", 2: "bicycle", 3: "other"}
SCENE_IDX = int(os.environ["RF_SCENE_IDX"]) if os.environ.get("RF_SCENE_IDX") else None
with torch.no_grad():
    for i, batch in enumerate(site.test_loader):
        x = batch["input"].to(dev)
        feat = batch["feature"].to(dev)
        vt = batch["type"].to(dev)
        ft = batch["future"].to(dev)
        if SCENE_IDX is not None and i != SCENE_IDX:
            continue
        if torch.isnan(x[:, 0, -2:, :]).any() or torch.isnan(ft[0, 0]).any():
            if SCENE_IDX is not None:
                raise SystemExit(f"scene {SCENE_IDX} has invalid ego")
            continue
        valid = [a for a in range(x.shape[1])
                 if not torch.isnan(x[0, a, -1]).any()
                 and not torch.isnan(ft[0, a]).any()]
        if SCENE_IDX is not None or len(valid) >= 3:
            break
    loc = int(batch["locationId"].view(-1)[0])
    print(f"scene loc={loc} agents={valid}")
    bx = boundaries_for_location(loc)
    xlo, xhi = float(bx[0, 0]), float(bx[0, 1])
    ylo, yhi = float(bx[1, 0]), float(bx[1, 1])
    scale = np.array([xhi - xlo, yhi - ylo])
    lo = np.array([xlo, ylo])
    scale_t = torch.tensor(scale, dtype=torch.float32, device=dev)

    lo_t = torch.tensor(lo, dtype=torch.float32, device=dev)
    Pa = {a: agent_density(x, feat, vt, a) for a in valid}        # each (G,K)
    P_ego = Pa[0]
    # ego scalar velocity (m/s) from ground-truth (planned) trajectory
    tr0 = ft[0, 0] * scale_t + lo_t
    v_ego = torch.zeros(K, 2, device=dev)
    v_ego[:-1] = (tr0[1:] - tr0[:-1]) / DT
    v_ego[-1] = v_ego[-2]
    # collision probability x collision energy (physical, Joules):
    #   pcoll_j(g,k) = P_ego(g,k) * (P_j * 1_disk(r_ego+r_j))(g,k)   probability
    #   C_j(k)       = 1/2 mu ||v_ego(k)-v_j(k)||^2                  Joules
    #   Risk_k(g)    = sum_j pcoll_j(g,k) * C_j(k)                   expected E (J)
    # only well-observed neighbours contribute (sparse history -> unreliable)
    contrib_agents = [j for j in valid if j != 0
                      and int((~torch.isnan(x[0, j, :, 0])).sum()) >= MIN_HIST]
    print(f"\n  risk contributors (hist>={MIN_HIST}): {contrib_agents}  "
          f"(excluded {[j for j in valid if j != 0 and j not in contrib_agents]})")
    dims_e = tuple(d / 2 for d in VEH_LW.get(agent_class(vt, 0), (4.5, 1.9)))
    v_ego_np = v_ego.cpu().numpy()
    th_e = heading_seq(v_ego_np, float(feat[0, 0, -1, 0]) * 2 * np.pi)
    Cj, meetj = {}, {}
    for j in contrib_agents:
        v_j = centroid_vel(Pa[j], scale_t, lo_t)                  # (K,2) smoothed
        Cj[j] = 0.5 * M_R * (v_ego - v_j).pow(2).sum(-1)          # (K,) J
        dims_j = tuple(d / 2 for d in VEH_LW.get(agent_class(vt, j), (4.5, 1.9)))
        th_j = heading_seq(v_j.cpu().numpy(), float(feat[0, j, -1, 0]) * 2 * np.pi)
        meetj[j] = P_ego * box_overlap(Pa[j], th_e, th_j, dims_e, dims_j, scale_t)
    risk = torch.zeros(G, K, device=dev)
    for j in contrib_agents:
        risk += meetj[j] * Cj[j].view(1, K)                       # expected E (J)

    # --- model-predicted motion vs ground truth (does the density translate?)
    print("\n" + "#" * 78)
    print("  MODEL CENTROID & MODE vs GROUND-TRUTH PATH (metres)")
    print(f"  location {loc} boundaries: x[{xlo:.0f},{xhi:.0f}] y[{ylo:.0f},{yhi:.0f}]")
    print("#" * 78)
    obs0 = x[0, 0, -1].cpu().numpy() * scale + lo
    for a in valid:
        cen = (torch.einsum("gd,gk->kd", grid, Pa[a]) * scale_t + lo_t).cpu().numpy()
        mode = (grid[Pa[a].argmax(dim=0)] * scale_t + lo_t).cpu().numpy()  # (K,2)
        gt = (ft[0, a].cpu().numpy() * scale + lo)                      # (K,2) GT
        obs_n = x[0, a, -1].cpu().numpy()                              # normalized
        obs = obs_n * scale + lo
        nhist = int((~torch.isnan(x[0, a, :, 0])).sum())              # valid hist steps
        Th = x.shape[2]
        mdisp = np.linalg.norm(cen[-1] - cen[0])
        gdisp = np.linalg.norm(gt[-1] - gt[0])
        mspd = np.linalg.norm(np.diff(cen, axis=0), axis=1).mean() / DT
        gspd = np.linalg.norm(np.diff(gt, axis=0), axis=1).mean() / DT
        cls = CLASS_NAME.get(agent_class(vt, a), "?")
        tag = ("EGO" if a == 0 else f"agent {a}") + f" [{cls}]"
        rel = obs - obs0
        # heading-frame check: observed motion direction (our metre frame) vs the
        # heading feature (feat * 360 deg). For moving agents they should match.
        hist_m = x[0, a].cpu().numpy() * scale + lo                   # (T,2) metres
        valid_h = ~np.isnan(hist_m[:, 0])
        if valid_h.sum() >= 2:
            d_disp = hist_m[valid_h][-1] - hist_m[valid_h][0]
            mot_deg = float(np.degrees(np.arctan2(d_disp[1], d_disp[0])) % 360)
        else:
            mot_deg = float("nan")
        feat_deg = float(feat[0, a, -1, 0]) * 360.0
        print(f"\n  {tag}  obs=({obs[0]:.0f},{obs[1]:.0f})m  "
              f"rel-to-ego=({rel[0]:+.0f},{rel[1]:+.0f})m  hist={nhist}/{Th}  "
              f"motion_dir={mot_deg:.0f}deg  feat_heading={feat_deg:.0f}deg")
        print(f"    MODEL centroid: net {mdisp:5.1f}m  speed {mspd:4.1f} m/s   |   "
              f"GT: net {gdisp:5.1f}m  speed {gspd:4.1f} m/s")
        for kk in (0, 10, 20, 30, 40, 49):
            print(f"      k={kk:2d}: cen=({cen[kk,0]:6.1f},{cen[kk,1]:6.1f})  "
                  f"mode=({mode[kk,0]:6.1f},{mode[kk,1]:6.1f})  "
                  f"GT=({gt[kk,0]:6.1f},{gt[kk,1]:6.1f})  "
                  f"|cen-GT|={np.linalg.norm(cen[kk]-gt[kk]):4.1f} "
                  f"|mode-GT|={np.linalg.norm(mode[kk]-gt[kk]):4.1f}")

    ego_now_m = x[0, 0, -1].cpu().numpy() * scale + lo
    ego_prev_m = x[0, 0, -2].cpu().numpy() * scale + lo
    dirv = ego_now_m - ego_prev_m
    dirv = dirv / (np.linalg.norm(dirv) + 1e-9)
    queries = {
        "ego_now": x[0, 0, -1].cpu().numpy(),
        "12m_in_front": ((ego_now_m + 12.0 * dirv) - lo) / scale,
        "12m_behind": ((ego_now_m - 12.0 * dirv) - lo) / scale,
    }

    for kq in (5, 15, 30):
        gmax = int(torch.argmax(risk[:, kq]))
        gx_m = grid[gmax].cpu().numpy() * scale + lo
        print("\n" + "=" * 78)
        print(f"  HORIZON STEP k={kq}  ({kq*DT:.2f}s ahead)")
        print("=" * 78)
        # per-step relative speed (model-pred) for context
        spd = {j: float((v_ego[kq] - centroid_vel(Pa[j], scale_t, lo_t)[kq])
                        .pow(2).sum().sqrt()) for j in valid if j != 0}
        qs = dict(queries)
        qs[f"risk_argmax@({gx_m[0]:.0f},{gx_m[1]:.0f})m"] = grid[gmax].cpu().numpy()
        for name, qn in qs.items():
            gi = nearest_cell(qn)
            pe = float(P_ego[gi, kq])
            print(f"\n  cell '{name}'   P_ego = {pe:.3e}")
            tot = 0.0
            for j in valid:
                if j == 0:
                    continue
                pj = float(Pa[j][gi, kq])
                if j not in meetj:                               # excluded (sparse hist)
                    print(f"    agent {j}: [EXCL] P_j={pj:.3e}  (history<{MIN_HIST})")
                    continue
                pcoll = float(meetj[j][gi, kq])                  # collision prob
                cj = float(Cj[j][kq])                            # collision energy J
                term = pcoll * cj
                tot += term
                meets = "MEET " if pcoll > 0 else "no   "
                print(f"    agent {j}: [{meets}] P_j={pj:.3e}  P_coll="
                      f"{pcoll:.3e}  |dv|={spd[j]:4.1f}m/s  C_j={cj:8.0f}J  "
                      f"E = {term:.4e} J")
            print(f"    --> Risk_k(g) = {tot:.4e} J")
