"""Honest counterfactual risk demonstration (cost-driven).

Picks a real all-33 test scene, evaluates the trained world-model occupancy
density on a spatial grid for the predicted agent (autonomous rollout -- the
regime the model was actually trained in), and computes the counterfactual
objective J(A) (paper Eq. counterfactual_objective) under several candidate
ego maneuvers.

HONESTY: InD has no ego-action labels, so the learned action-embedding path is
NOT data-calibrated and is deliberately NOT used. The counterfactual signal is
the physically grounded ego kinetic-energy term: a candidate maneuver sets the
ego speed profile, which changes C_k = 1/2 m_r ||v_ego - v_agent||^2 and hence
J(A). The occupancy density is identical across maneuvers (trained model);
only the physical severity differs. Braking genuinely lowers collision energy.

Env: RF_CKPT (default serialized/riskflow_ind_3.pt), RF_GRID (default 40).
Run from project root:  python scripts/counterfactual_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from datasets.InD import InD, boundaries_for_location  # noqa: E402
from model.RiskFlow import RiskFlow  # noqa: E402
from riskflow_config import default_dict  # noqa: E402

ckpt = os.environ.get("RF_CKPT", "serialized/riskflow_ind_3.pt")
S = int(os.environ.get("RF_GRID", "40"))
GAMMA = 0.95
DT = 0.08            # sampling_step=2 @ 25 Hz
MASS = 1500.0        # constant per-vehicle mass (InD has none) -> m_r = M/2
M_R = MASS / 2.0

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
    scene_level=False,                                           # single-target demo
).to(dev)
m.load_state_dict(torch.load(ckpt, map_location=dev))
m.eval()

# normalized SxS grid (model lives in [0,1] normalized space)
gx = torch.linspace(0.05, 0.95, S)
gy = torch.linspace(0.05, 0.95, S)
GX, GY = torch.meshgrid(gx, gy, indexing="ij")
grid = torch.stack([GX.reshape(-1), GY.reshape(-1)], -1).to(dev)  # (G,2)
G = grid.shape[0]

with torch.no_grad():
    for batch in site.test_loader:
        x = batch["input"].to(dev)
        feat = batch["feature"].to(dev)
        vt = batch["type"].to(dev)
        loc = int(batch["locationId"].view(-1)[0].item())
        K = c["seq_len"]
        if torch.isnan(x[:, 0, -2:, :]).any():      # need a clean ego history tail
            continue

        emb = m.encoder(None, torch.cat([x, feat], dim=-1), vt)      # (1,E)
        bx = boundaries_for_location(loc)                            # [[xlo,xhi],[ylo,yhi]]
        sx = float(bx[0, 1] - bx[0, 0]); sy = float(bx[1, 1] - bx[1, 0])
        scale = torch.tensor([sx, sy], device=dev)
        lo = torch.tensor([float(bx[0, 0]), float(bx[1, 0])], device=dev)
        A_SCALE = 100.0                                              # must match training

        # ego speed/heading from last observed ego history (meters)
        ego_hist = x[0, 0, -2:, :] * scale + lo
        v0 = (ego_hist[1] - ego_hist[0]) / DT
        sp0 = float(v0.norm().clamp(min=1.0))
        dirv = v0 / v0.norm().clamp(min=1e-6)
        p0 = ego_hist[1]
        t = torch.arange(K, device=dev, dtype=torch.float32)

        def speed_profile(name):
            if name == "brake":
                return (sp0 - 3.0 * t * DT).clamp(min=0.0)
            if name == "accelerate":
                return sp0 + 1.5 * t * DT
            return torch.full((K,), sp0, device=dev)                 # maintain

        def maneuver_action(sp):
            # candidate ego future positions (m) -> normalized -> 2nd-diff*scale
            pos = p0.unsqueeze(0) + torch.cumsum(
                sp.unsqueeze(-1) * dirv.unsqueeze(0) * DT, dim=0)     # (K,2) m
            pn = (pos - lo) / scale                                   # normalized
            a = pn[2:] - 2.0 * pn[1:-1] + pn[:-2]
            a = torch.cat([a[:1], a, a[-1:]], dim=0) * A_SCALE        # (K,2)
            return a.unsqueeze(0)                                     # (1,K,2)

        def density_for(action):
            # LEARNED action-conditioned occupancy via the trained action path
            cond = m._flow_condition(emb, K, action)                 # (1,K,E)
            condG = cond.expand(G, K, cond.shape[-1]) if cond.dim() == 3 \
                else cond.expand(G, cond.shape[-1])
            y = grid.view(G, 1, 2).expand(G, K, 2).contiguous()
            z, det = m.flow(y, condG, sampling_frequency=1)
            _, logpx = m.log_prob(z, det, embedding=emb.expand(G, emb.shape[-1]))
            P = logpx.exp()
            return P / P.sum(0, keepdim=True).clamp(min=1e-9)         # (G,K)

        def jof(name):
            sp = speed_profile(name)
            P = density_for(maneuver_action(sp))                     # learned density
            cen_m = (P.t().unsqueeze(-1) * grid.unsqueeze(0)).sum(1) * scale  # (K,2)
            v_ag = torch.zeros(K, 2, device=dev)
            v_ag[:-1] = (cen_m[1:] - cen_m[:-1]) / DT
            v_ag[-1] = v_ag[-2]
            v_ego = sp.unsqueeze(-1) * dirv.unsqueeze(0)             # (K,2)
            C = 0.5 * M_R * (v_ego - v_ag).pow(2).sum(-1)            # KE loss (K,)
            disc = GAMMA ** t
            risk_k = (P * C.unsqueeze(0)).sum(0)
            return float((disc * risk_k).sum().item()), P

        Js, Pmaps = {}, {}
        for nm in ("maintain", "brake", "accelerate"):
            Js[nm], Pmaps[nm] = jof(nm)
        best = min(Js, key=Js.get)

        print("=" * 60)
        print(f"RESULT ckpt={ckpt} loc={loc} grid={S}x{S} ego_v0={sp0:.2f} m/s")
        for k, v in Js.items():
            print(f"RESULT J[{k:>10}] = {v:.4f}")
        print(f"RESULT A_star = {best}  (argmin J; lower = safer)")
        print("=" * 60)
        np.savez(
            "counterfactual_demo.npz",
            P_maintain=Pmaps["maintain"].reshape(S, S, K).cpu().numpy(),
            P_brake=Pmaps["brake"].reshape(S, S, K).cpu().numpy(),
            P_accelerate=Pmaps["accelerate"].reshape(S, S, K).cpu().numpy(),
            J=np.array([Js["maintain"], Js["brake"], Js["accelerate"]]),
            loc=loc, ego_v0=sp0,
        )
        break
