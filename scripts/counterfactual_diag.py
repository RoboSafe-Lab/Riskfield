"""Diagnostic counterfactual demo on the action-conditioned (option 2) model.

For each of N test scenes we evaluate the LEARNED action-conditioned density
P_A under candidate ego maneuvers A in {maintain, brake, accelerate}, and
decompose the counterfactual objective J(A) into three protocols:

  J_full (P_A, v_A)   -- full counterfactual (density AND cost react to A)
  J_dens (P_A, v_mt)  -- density-only: only the learned density changes
  J_phys (P_mt, v_A)  -- cost-only: physics, no learned signal (the ind_3 view)

This tells us *how much* of the J(A) spread comes from the learned action path
vs. the physical cost term. We also report DENSITY SHIFTS as the per-step L1
distance between P_A and P_maintain. If density shifts are ~0, option 2's
action path is not doing meaningful work; if they are large, but J_dens is
flat across A, the learned changes are not risk-discriminative.

The ACCELERATION SWEEP answers a separate question: how far outside the realistic
band must the ego action go before the learned density moves at all? Both P are
normalized over the grid, so the L1 distance lives in [0, 2] and TV = L1/2 is the
fraction of probability mass that relocates.

Env: RF_CKPT (default serialized/riskflow_ind_4.pt), RF_GRID (40),
     RF_N_SCENES (20), RF_ACC_SWEEP (comma-separated constant accelerations,
     m/s^2, applied over the horizon).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from datasets.InD import InD, boundaries_for_location  # noqa: E402
from model.RiskFlow import RiskFlow  # noqa: E402
from riskflow_config import default_dict  # noqa: E402

ckpt = os.environ.get("RF_CKPT", "serialized/riskflow_ind_4.pt")
S = int(os.environ.get("RF_GRID", "40"))
N_SCENES = int(os.environ.get("RF_N_SCENES", "20"))
GAMMA = 0.95
DT = 0.08
MASS = 1500.0
M_R = MASS / 2.0
A_SCALE = 100.0           # must match training
# Constant ego accelerations to sweep, m/s^2. Comfortable braking is ~3, emergency
# ~8; beyond roughly 10 the request exceeds tyre friction and is off-distribution
# for anything the model saw in training.
ACC_SWEEP = [float(v) for v in os.environ.get(
    "RF_ACC_SWEEP", "-3,-1.5,1.5,3,-8,8,-15,15,-30,30").split(",")]

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
    scene_level=False,                                           # legacy diag on ind_4
).to(dev)
m.load_state_dict(torch.load(ckpt, map_location=dev))
m.eval()

gx = torch.linspace(0.05, 0.95, S)
gy = torch.linspace(0.05, 0.95, S)
GX, GY = torch.meshgrid(gx, gy, indexing="ij")
grid = torch.stack([GX.reshape(-1), GY.reshape(-1)], -1).to(dev)
G = grid.shape[0]
K = c["seq_len"]


def density_for(emb, action):
    cond = m._flow_condition(emb, K, action)
    condG = cond.expand(G, K, cond.shape[-1]) if cond.dim() == 3 \
        else cond.expand(G, cond.shape[-1])
    y = grid.view(G, 1, 2).expand(G, K, 2).contiguous()
    z, det = m.flow(y, condG, sampling_frequency=1)
    _, logpx = m.log_prob(z, det, embedding=emb.expand(G, emb.shape[-1]))
    P = logpx.exp()
    return P / P.sum(0, keepdim=True).clamp(min=1e-9)


def J_value(P, sp, dirv, scale):
    cen_m = (P.t().unsqueeze(-1) * grid.unsqueeze(0)).sum(1) * scale
    v_ag = torch.zeros(K, 2, device=dev)
    v_ag[:-1] = (cen_m[1:] - cen_m[:-1]) / DT
    v_ag[-1] = v_ag[-2]
    v_ego = sp.unsqueeze(-1) * dirv.unsqueeze(0)
    C = 0.5 * M_R * (v_ego - v_ag).pow(2).sum(-1)
    disc = (GAMMA ** torch.arange(K, device=dev, dtype=torch.float32))
    return float((disc * (P * C.unsqueeze(0)).sum(0)).sum().item())


names = ("maintain", "brake", "accelerate")
agg = {f"J_full_{n}": [] for n in names}
agg.update({f"J_dens_{n}": [] for n in names})
agg.update({f"J_phys_{n}": [] for n in names})
agg["dL1_brake_vs_maintain"] = []
agg["dL1_accel_vs_maintain"] = []
for _a in ACC_SWEEP:
    agg[f"sweep_{_a:+.1f}"] = []
done = 0
with torch.no_grad():
    for batch in site.test_loader:
        if done >= N_SCENES:
            break
        x = batch["input"].to(dev)
        feat = batch["feature"].to(dev)
        vt = batch["type"].to(dev)
        loc = int(batch["locationId"].view(-1)[0].item())
        if torch.isnan(x[:, 0, -2:, :]).any():
            continue

        emb = m.encoder(None, torch.cat([x, feat], dim=-1), vt)
        bx = boundaries_for_location(loc)
        sx = float(bx[0, 1] - bx[0, 0]); sy = float(bx[1, 1] - bx[1, 0])
        scale = torch.tensor([sx, sy], device=dev)
        lo = torch.tensor([float(bx[0, 0]), float(bx[1, 0])], device=dev)

        ego_hist = x[0, 0, -2:, :] * scale + lo
        v0 = (ego_hist[1] - ego_hist[0]) / DT
        sp0 = float(v0.norm().clamp(min=1.0))
        dirv = v0 / v0.norm().clamp(min=1e-6)
        p0 = ego_hist[1]
        t = torch.arange(K, device=dev, dtype=torch.float32)

        def sp(name):
            if name == "brake":
                return (sp0 - 3.0 * t * DT).clamp(min=0.0)
            if name == "accelerate":
                return sp0 + 1.5 * t * DT
            return torch.full((K,), sp0, device=dev)

        def action_for(spv):
            pos = p0.unsqueeze(0) + torch.cumsum(
                spv.unsqueeze(-1) * dirv.unsqueeze(0) * DT, dim=0)
            pn = (pos - lo) / scale
            a = pn[2:] - 2.0 * pn[1:-1] + pn[:-2]
            a = torch.cat([a[:1], a, a[-1:]], dim=0) * A_SCALE
            return a.unsqueeze(0)

        speeds = {n: sp(n) for n in names}
        actions = {n: action_for(speeds[n]) for n in names}
        Ps = {n: density_for(emb, actions[n]) for n in names}

        # density shift (per-step L1 distance, mean over K) -- diagnostic of
        # whether the learned action path actually moves the density.
        agg["dL1_brake_vs_maintain"].append(
            (Ps["brake"] - Ps["maintain"]).abs().sum(0).mean().item())
        agg["dL1_accel_vs_maintain"].append(
            (Ps["accelerate"] - Ps["maintain"]).abs().sum(0).mean().item())

        # Same L1, but as a function of how hard the ego is asked to accelerate.
        # The realistic arms above move the density by ~0; this locates the
        # magnitude at which the learned action path finally does something.
        for _acc in ACC_SWEEP:
            _spv = (sp0 + _acc * t * DT).clamp(min=0.0)
            _Pa = density_for(emb, action_for(_spv))
            agg[f"sweep_{_acc:+.1f}"].append(
                (_Pa - Ps["maintain"]).abs().sum(0).mean().item())

        for n in names:
            agg[f"J_full_{n}"].append(J_value(Ps[n], speeds[n], dirv, scale))
            agg[f"J_dens_{n}"].append(J_value(Ps[n], speeds["maintain"], dirv, scale))
            agg[f"J_phys_{n}"].append(J_value(Ps["maintain"], speeds[n], dirv, scale))
        done += 1

print("=" * 70)
print(f"RESULT ckpt={ckpt} scenes={done} grid={S}x{S} action_scale={A_SCALE}")
def mean(k): return float(np.mean(agg[k]))
print(f"RESULT density-shift L1  brake-vs-maintain  = {mean('dL1_brake_vs_maintain'):.5f}")
print(f"RESULT density-shift L1  accel-vs-maintain  = {mean('dL1_accel_vs_maintain'):.5f}")
print("RESULT --- density-shift L1 vs maintain, by constant ego acceleration ---")
print("RESULT    a (m/s^2)      L1       TV   regime")
for _a in sorted(ACC_SWEEP, key=abs):
    _v = mean(f"sweep_{_a:+.1f}")
    _r = ("realistic" if abs(_a) <= 4.0 else
          "hard but physical" if abs(_a) <= 8.0 else "beyond friction limit (OOD)")
    print(f"RESULT   {_a:+8.1f}   {_v:7.4f}  {_v/2:7.4f}   {_r}")
print("RESULT J_full   (density+cost react)  : "
      f"mt={mean('J_full_maintain'):.1f}  br={mean('J_full_brake'):.1f}  ac={mean('J_full_accelerate'):.1f}")
print("RESULT J_dens   (density only, cost fixed): "
      f"mt={mean('J_dens_maintain'):.1f}  br={mean('J_dens_brake'):.1f}  ac={mean('J_dens_accelerate'):.1f}")
print("RESULT J_phys   (cost only, density fixed): "
      f"mt={mean('J_phys_maintain'):.1f}  br={mean('J_phys_brake'):.1f}  ac={mean('J_phys_accelerate'):.1f}")
print("=" * 70)
np.savez("counterfactual_diag.npz",
         **{k: np.array(v) for k, v in agg.items()})
