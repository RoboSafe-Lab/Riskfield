"""Rank test scenes for the risk-field demo.

For each of the first RF_SCENES test-loader batches, build the gated risk field
(same formula as animate_horizon.py: Risk_k(g) = sum_j P_ego*P_j*C_j(k), only
neighbours with history >= MIN_HIST contribute) and report the scenes with the
largest total risk, so we can pick a demo scene that (a) has >=2 well-observed
neighbours and (b) actually contains a genuine ego-neighbour interaction.

Prints the top scenes by total risk with their batch index (use as
RF_SCENE_IDX in animate_horizon.py), location, and per-neighbour history/speed.
Env: RF_CKPT, RF_GRID (64), RF_SCENES (600), RF_MIN_HIST (30).
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
SCENES = int(os.environ.get("RF_SCENES", "600"))
MIN_HIST = int(os.environ.get("RF_MIN_HIST", "30"))
DT, M_R = 0.08, 1500.0 / 2.0
SG_WIN, SG_POLY = 11, 2
VEH_LW = {0: (4.5, 1.9), 1: (10.0, 2.6), 2: (1.8, 0.6), 3: (0.7, 0.7)}

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


def density(x, feat, vt, a):
    z, det = m.flow(Y_GRID, _cond_grid(x, feat, vt, a), sampling_frequency=1)
    P = base_logpx(z, det).exp()
    return P / P.sum(0, keepdim=True).clamp(min=1e-9)


def agent_class(vt, a):
    return int(vt[0, a].reshape(-1)[0])


def centroid_vel(P, scale_t, lo_t):
    cen = (torch.einsum("gd,gk->kd", grid, P) * scale_t + lo_t)
    c = cen.detach().cpu().numpy()
    w = min(SG_WIN, c.shape[0] if c.shape[0] % 2 else c.shape[0] - 1)
    if w >= SG_POLY + 2:
        v = savgol_filter(c, window_length=w, polyorder=SG_POLY, deriv=1,
                          delta=DT, axis=0)
    else:
        v = np.gradient(c, DT, axis=0)
    return torch.tensor(v, dtype=torch.float32, device=P.device)


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
        msk = cache.get(key)
        if msk is None:
            msk = _sat_mask(dxm, dym, th_e[k], dims_e[0], dims_e[1],
                            th_j[k], dims_j[0], dims_j[1]).astype(np.float32)
            cache[key] = msk
        out[:, :, k] = ndi_convolve(arr[:, :, k], msk, mode="constant")
    return torch.tensor(out.reshape(G, K), device=P.device)


results = []  # (total_risk, idx, loc, info)
with torch.no_grad():
    for i, batch in enumerate(site.test_loader):
        if i >= SCENES:
            break
        x = batch["input"].to(dev)
        feat = batch["feature"].to(dev)
        vt = batch["type"].to(dev)
        ft = batch["future"].to(dev)
        if torch.isnan(x[:, 0, -2:, :]).any() or torch.isnan(ft[0, 0]).any():
            continue
        loc = int(batch["locationId"].view(-1)[0])
        bx = boundaries_for_location(loc)
        scale = torch.tensor([float(bx[0, 1] - bx[0, 0]),
                              float(bx[1, 1] - bx[1, 0])], device=dev)
        lo = torch.tensor([float(bx[0, 0]), float(bx[1, 0])], device=dev)
        neigh = [a for a in range(1, x.shape[1])
                 if not torch.isnan(x[0, a, -1]).any()
                 and not torch.isnan(ft[0, a]).any()
                 and int((~torch.isnan(x[0, a, :, 0])).sum()) >= MIN_HIST]
        if len(neigh) < 2:
            continue
        P_ego = density(x, feat, vt, 0)
        tr0 = ft[0, 0] * scale + lo
        v_ego = torch.zeros(K, 2, device=dev)
        v_ego[:-1] = (tr0[1:] - tr0[:-1]) / DT
        v_ego[-1] = v_ego[-2]
        dims_e = tuple(d / 2 for d in VEH_LW.get(agent_class(vt, 0), (4.5, 1.9)))
        th_e = heading_seq(v_ego.cpu().numpy(), float(feat[0, 0, -1, 0]) * 2 * np.pi)
        risk = torch.zeros(G, K, device=dev)
        info = []
        for j in neigh:
            P = density(x, feat, vt, j)
            v_j = centroid_vel(P, scale, lo)
            dims_j = tuple(d / 2 for d in VEH_LW.get(agent_class(vt, j), (4.5, 1.9)))
            th_j = heading_seq(v_j.cpu().numpy(), float(feat[0, j, -1, 0]) * 2 * np.pi)
            C = 0.5 * M_R * (v_ego - v_j).pow(2).sum(-1)
            pcoll = P_ego * box_overlap(P, th_e, th_j, dims_e, dims_j, scale)
            risk += pcoll * C.view(1, K)
            gspd = float((ft[0, j] * scale + lo).diff(dim=0).pow(2)
                         .sum(-1).sqrt().mean() / DT)
            info.append(f"a{j}(spd{gspd:.0f})")
        tot = float(risk.sum())
        pk = float(risk.max())
        results.append((tot, i, loc, len(neigh), pk, " ".join(info)))

results.sort(reverse=True)
print(f"\nRESULT scanned {SCENES} scenes; {len(results)} with >=2 well-observed "
      f"neighbours\n")
print("  rank  idx   loc  #ngh   total_risk   peak_cell   neighbours(GTspeed)")
for r, (tot, i, loc, nn, pk, info) in enumerate(results[:25]):
    print(f"  {r+1:3d}  {i:4d}  {loc:3d}  {nn:4d}   {tot:10.1f}  {pk:9.2f}   {info}")
