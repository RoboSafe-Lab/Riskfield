"""Verdict for the scene-level AR model (ind_6).

Part A -- accuracy: NLL of ground-truth futures under the scene-level forward,
reported for the ego (index 0, apples-to-apples with prior single-target NLL)
and averaged over all non-ego agents (the scene-level objective).

Part B -- learned counterfactual diagnostic: for each test scene and each
candidate ego maneuver, run the AR chain (decode neighbours in order, feeding
each one's MAP centroid to the next), then decompose the objective J(A):

  J_full (P_A, v_A)   density AND cost react to the ego action
  J_dens (P_A, v_mt)  density only  -> isolates the LEARNED counterfactual
  J_phys (P_mt, v_A)  cost only     -> physics, no learned signal

plus the per-agent density shift L1 vs. maintain.

The ACCELERATION SWEEP asks how hard the ego must be driven before the neighbour
densities move at all. Each per-agent P is normalized over the grid, so the L1
distance lies in [0, 2] and TV = L1/2 is the fraction of probability mass that
relocates.

Env: RF_CKPT (default serialized/riskflow_ind_6.pt), RF_GRID (40),
     RF_NLL_SCENES (4000), RF_CF_SCENES (20),
     RF_ACC_SWEEP (comma-separated constant accelerations, m/s^2).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from datasets.InD import InD, boundaries_for_location  # noqa: E402
from model.RiskFlow import RiskFlow  # noqa: E402
from riskflow_config import default_dict  # noqa: E402

ckpt = os.environ.get("RF_CKPT", "serialized/riskflow_ind_6.pt")
S = int(os.environ.get("RF_GRID", "40"))
NLL_SCENES = int(os.environ.get("RF_NLL_SCENES", "4000"))
CF_SCENES = int(os.environ.get("RF_CF_SCENES", "20"))
GAMMA, DT, A_SCALE = 0.95, 0.08, 100.0
M_R = 1500.0 / 2.0

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
    scene_level=True,
).to(dev)
miss, unexp = m.load_state_dict(torch.load(ckpt, map_location=dev), strict=False)
print(f"RESULT load: missing={len(miss)} unexpected={len(unexp)}")
m.eval()
K = c["seq_len"]


def base_logpx(z, det):
    d = z.shape[-1]
    logpz = -0.5 * (z.pow(2).sum(-1) + d * np.log(2 * np.pi))
    return logpz - det


# ------------------------------------------------------------------ Part A
ego_nll, all_nll, n_ego, n_all = 0.0, 0.0, 0, 0
with torch.no_grad():
    for i, batch in enumerate(site.test_loader):
        if i >= NLL_SCENES:
            break
        x = batch["input"].to(dev)
        feat = batch["feature"].to(dev)
        vt = batch["type"].to(dev)
        fut = batch["future"].to(dev)                       # (1,N,K,2)
        z, det, _ = m(x, fut, feat, vt)                     # (1,N,K,2),(1,N,K)
        lpx = base_logpx(z, det)                            # (1,N,K)
        valid = ~torch.isnan(fut).any(dim=(2, 3))           # (1,N)
        if valid[0, 0]:
            ego_nll += -lpx[0, 0].mean().item(); n_ego += 1
        for a in range(1, fut.shape[1]):
            if valid[0, a]:
                all_nll += -lpx[0, a].mean().item(); n_all += 1
print(f"RESULT accuracy scenes={min(NLL_SCENES, i)}  "
      f"ego_NLL={ego_nll / max(n_ego,1):.4f}  "
      f"neighbour_NLL={all_nll / max(n_all,1):.4f}")


# ------------------------------------------------------------------ Part B
gx = torch.linspace(0.05, 0.95, S)
GX, GY = torch.meshgrid(gx, gx, indexing="ij")
grid = torch.stack([GX.reshape(-1), GY.reshape(-1)], -1).to(dev)   # (G,2)
G = grid.shape[0]


def agent_density(cond_a):
    """cond_a: (K,E) -> per-step density on the grid (G,K), normalized."""
    y = grid.view(G, 1, 2).expand(G, K, 2).contiguous()
    cg = cond_a.unsqueeze(0).expand(G, K, cond_a.shape[-1])
    z, det = m.flow(y, cg, sampling_frequency=1)
    P = base_logpx(z, det).exp()
    return P / P.sum(0, keepdim=True).clamp(min=1e-9)


names = ("maintain", "brake", "accelerate")
# Constant ego accelerations to sweep. Comfortable braking ~3 m/s^2, emergency ~8;
# beyond ~10 the request exceeds tyre friction and is off-distribution.
ACC_SWEEP = [float(v) for v in os.environ.get(
    "RF_ACC_SWEEP", "-3,-1.5,1.5,3,-8,8,-15,15,-30,30").split(",")]
agg = {f"J_{k}_{n}": [] for k in ("full", "dens", "phys") for n in names}
agg["dL1_brake"] = []
for _a in ACC_SWEEP:
    agg[f"sweep_{_a:+.1f}"] = []
agg["dL1_accel"] = []
done = 0
with torch.no_grad():
    for batch in site.test_loader:
        if done >= CF_SCENES:
            break
        x = batch["input"].to(dev)
        feat = batch["feature"].to(dev)
        vt = batch["type"].to(dev)
        if torch.isnan(x[:, 0, -2:, :]).any():
            continue
        loc = int(batch["locationId"].view(-1)[0].item())
        bx = boundaries_for_location(loc)
        sx, sy = float(bx[0, 1] - bx[0, 0]), float(bx[1, 1] - bx[1, 0])
        scale = torch.tensor([sx, sy], device=dev)
        lo = torch.tensor([float(bx[0, 0]), float(bx[1, 0])], device=dev)

        agent_emb, car_valid = m.encoder(
            None, torch.cat([x, feat], -1), vt, per_agent=True)
        order = m._compute_ordering(x, car_valid)
        N = car_valid.shape[1]
        neigh = [int(order[0, i]) for i in range(1, N) if car_valid[0, int(order[0, i])]]
        if not neigh:
            continue

        ego_hist = x[0, 0, -2:, :] * scale + lo
        v0 = (ego_hist[1] - ego_hist[0]) / DT
        sp0 = float(v0.norm().clamp(min=1.0))
        dirv = v0 / v0.norm().clamp(min=1e-6)
        p0 = ego_hist[1]
        t = torch.arange(K, device=dev, dtype=torch.float32)

        def speed(nm):
            if nm == "brake":
                return (sp0 - 3.0 * t * DT).clamp(min=0.0)
            if nm == "accelerate":
                return sp0 + 1.5 * t * DT
            return torch.full((K,), sp0, device=dev)

        def action(sp):
            pos = p0.unsqueeze(0) + torch.cumsum(
                sp.unsqueeze(-1) * dirv.unsqueeze(0) * DT, dim=0)
            pn = (pos - lo) / scale
            a = pn[2:] - 2 * pn[1:-1] + pn[:-2]
            return torch.cat([a[:1], a, a[-1:]], 0).unsqueeze(0) * A_SCALE

        # AR chain decode per maneuver -> per-neighbour density + MAP centroid.
        Pmap, Vmap = {}, {}
        for nm in names:
            sp = speed(nm)
            s_seq = m.world_model.forward_scene(agent_emb, car_valid, K, action(sp))
            Y = torch.zeros(1, N, K, 2, device=dev)
            Ps, Vs = {}, {}
            for ag in neigh:
                cond = m.ar_decoder(agent_emb, s_seq, Y, order)      # (1,N,K,E)
                P = agent_density(cond[0, ag])                       # (G,K)
                cen = (P.t().unsqueeze(-1) * grid.unsqueeze(0)).sum(1)  # (K,2) norm
                Y[0, ag] = cen
                Ps[ag] = P
                v = torch.zeros(K, 2, device=dev)
                cen_m = cen * scale
                v[:-1] = (cen_m[1:] - cen_m[:-1]) / DT
                v[-1] = v[-2]
                Vs[ag] = v
            Pmap[nm] = Ps
            Vmap[nm] = Vs

        def J(P_src, v_src_speed):
            tot = 0.0
            disc = GAMMA ** t
            v_ego = v_src_speed.unsqueeze(-1) * dirv.unsqueeze(0)
            for ag in neigh:
                P = Pmap[P_src][ag]
                C = 0.5 * M_R * (v_ego - Vmap[P_src][ag]).pow(2).sum(-1)
                tot = tot + (disc * (P * C.unsqueeze(0)).sum(0)).sum()
            return float(tot.item())

        for nm in names:
            agg[f"J_full_{nm}"].append(J(nm, speed(nm)))
            agg[f"J_dens_{nm}"].append(J(nm, speed("maintain")))
            agg[f"J_phys_{nm}"].append(J("maintain", speed(nm)))
        dl1b = np.mean([(Pmap["brake"][a] - Pmap["maintain"][a]).abs().sum(0).mean().item()
                        for a in neigh])
        dl1a = np.mean([(Pmap["accelerate"][a] - Pmap["maintain"][a]).abs().sum(0).mean().item()
                        for a in neigh])
        # Same per-agent L1, as a function of how hard the ego is driven. This is
        # the measurement behind the claim that the learned reaction only moves at
        # out-of-distribution ego actions.
        for _acc in ACC_SWEEP:
            _sp = (sp0 + _acc * t * DT).clamp(min=0.0)
            _s_seq = m.world_model.forward_scene(agent_emb, car_valid, K, action(_sp))
            _Y = torch.zeros(1, N, K, 2, device=dev); _Ps = {}
            for ag in neigh:
                _cond = m.ar_decoder(agent_emb, _s_seq, _Y, order)
                _P = agent_density(_cond[0, ag])
                _Y[0, ag] = (_P.t().unsqueeze(-1) * grid.unsqueeze(0)).sum(1)
                _Ps[ag] = _P
            agg[f"sweep_{_acc:+.1f}"].append(np.mean(
                [(_Ps[a] - Pmap["maintain"][a]).abs().sum(0).mean().item() for a in neigh]))
        agg["dL1_brake"].append(dl1b)
        agg["dL1_accel"].append(dl1a)
        done += 1


def mn(k):
    return float(np.mean(agg[k])) if agg[k] else float("nan")


print(f"RESULT cf_scenes={done}")
print(f"RESULT density-shift L1  brake={mn('dL1_brake'):.5f}  accel={mn('dL1_accel'):.5f}")
print("RESULT --- per-agent density-shift L1 vs maintain, by ego acceleration ---")
print("RESULT    a (m/s^2)      L1       TV   regime")
for _a in sorted(ACC_SWEEP, key=abs):
    _v = mn(f"sweep_{_a:+.1f}")
    _r = ("realistic" if abs(_a) <= 4.0 else
          "hard but physical" if abs(_a) <= 8.0 else "beyond friction limit (OOD)")
    print(f"RESULT   {_a:+8.1f}   {_v:7.4f}  {_v/2:7.4f}   {_r}")
print(f"RESULT J_full  mt={mn('J_full_maintain'):.1f}  br={mn('J_full_brake'):.1f}  ac={mn('J_full_accelerate'):.1f}")
print(f"RESULT J_dens  mt={mn('J_dens_maintain'):.1f}  br={mn('J_dens_brake'):.1f}  ac={mn('J_dens_accelerate'):.1f}")
print(f"RESULT J_phys  mt={mn('J_phys_maintain'):.1f}  br={mn('J_phys_brake'):.1f}  ac={mn('J_phys_accelerate'):.1f}")
