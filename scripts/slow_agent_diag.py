"""Diagnose model quality vs agent speed. For each test agent (with GT future),
binned by observed speed, report: predicted-density radius at mid-horizon (sharpness,
m), GT final-displacement error of the density centroid (FDE, m), centroid-velocity
RMS error vs GT (m/s), and the GT speed std over the horizon (is the future actually
determinate?). If slow bins have LARGE density radius but SMALL GT speed std, the
model is over-diffuse for slow agents whose future is determinate (a fixable deficit);
if GT speed std is large, the diffuseness is genuine intent uncertainty."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from datasets.registry import get_dataset
from model.RiskFlow import RiskFlow
from riskflow_config import default_dict
from scripts.joint_field import JointRiskField

reg = get_dataset(); c = default_dict(); dev = "cuda" if torch.cuda.is_available() else "cpu"
S = int(os.environ.get("RF_GRID", "64")); K = c["seq_len"]; DT = float(os.environ.get("RF_DT", "0.0667"))
MIN_HIST = 30; bf = reg["boundaries_for_location"]; N = int(os.environ.get("RF_N", "150"))

ind = reg["LoaderClass"](root=reg["root"], max_samples=c["maximum_samples"], train_ratio=c["train_ratio"],
    train_batch_size=c["train_batch_size"], test_batch_size=1, missing_rate=c["masked_data_ratio"],
    max_num_cars=c["max_num_cars"], max_empty_frames=c["max_empty_frames"], seq_len=c["seq_len"],
    moving_window=c["seq_len"] * 2, sampling_step=c["sampling_step"], should_shuffle=False, include_future=c["include_future"])
site = ind.observation_site_by_scope("all")


def build(sl):
    return RiskFlow(seq_len=c["seq_len"], input_dim=c["input_dim"], feature_dim=c["feature_dim"],
        embedding_dim=c["embedding_dim"], hidden_dim=c["hidden_dim"], max_num_cars=c["max_num_cars"],
        num_classes=c["num_classes"], gru_layers=c["gru_layers"], num_heads=c["num_heads"], dropout=c["dropout"],
        norm_rotation=c["norm_rotate"], flow_layers=c["flow_layers"], flow_hidden_dim=c["flow_hidden_dim"],
        coupling_layers=c["coupling_layers"], use_cnf=c["use_cnf"], use_cgmm=c["use_cgmm"], gmm_modes=c["gmm_modes"],
        use_world_model=True, wm_state_dim=c["wm_state_dim"], action_dim=c["action_dim"], scene_level=sl,
        use_map=True, map_size=c["map_size"], map_data_dir=reg["map_data_dir"], map_dataset=reg["map_dataset"]).to(dev).eval()


me = build(False); me.load_state_dict(torch.load(os.environ["RF_CKPT_EGO"], map_location=dev), strict=False)
mj = build(True);  mj.load_state_dict(torch.load(os.environ["RF_CKPT_JOINT"], map_location=dev), strict=False)
g1 = torch.linspace(0.05, 0.95, S); GX, GY = torch.meshgrid(g1, g1, indexing="ij")
grid = torch.stack([GX.reshape(-1), GY.reshape(-1)], -1).to(dev)
eng = JointRiskField(me, mj, grid, S, K, dev, min_hist=MIN_HIST)
gmn = grid.cpu().numpy()                                   # (G,2) normalized

bins = [(0, 1), (1, 3), (3, 6), (6, 10), (10, 99)]
acc = {b: {"rad": [], "fde": [], "velerr": [], "gtstd": [], "n": 0} for b in bins}

n_used = 0
for i, b in enumerate(site.test_loader):
    if n_used >= N:
        break
    x = b["input"].to(dev); feat = b["feature"].to(dev); vt = b["type"].to(dev); fut = b["future"].to(dev)
    if torch.isnan(x[:, 0, -2:, :]).any():
        continue
    n_used += 1
    loc = int(b["locationId"].view(-1)[0]); bx = bf(loc)
    xlo, xhi = float(bx[0, 0]), float(bx[0, 1]); ylo, yhi = float(bx[1, 0]), float(bx[1, 1])
    scale = torch.tensor([xhi - xlo, yhi - ylo], device=dev); LOC_T = torch.tensor([loc], device=dev)
    sca = np.array([xhi - xlo, yhi - ylo]); lon = np.array([xlo, ylo])
    pos_m = gmn * sca + lon                                # (G,2) metres
    contrib = [a for a in range(1, x.shape[1]) if not torch.isnan(x[0, a, -1]).any()
               and int((~torch.isnan(x[0, a, :, 0])).sum()) >= MIN_HIST]
    learned = os.environ.get("RF_VMETHOD") == "learned"
    with torch.no_grad():
        P_ego = eng.ego_density(x, feat, vt, 0, LOC_T)
        if learned:
            vf_ego = eng.ego_velocity_field(x, feat, vt, 0, LOC_T, scale)   # (S,S,K,2) m/s
            Pj, Vj = eng.joint_densities(x, feat, vt, P_ego, contrib, LOC_T, scale=scale)
        else:
            Pj = eng.joint_densities(x, feat, vt, P_ego, contrib, LOC_T); Vj = {}
    for a in [0] + contrib:
        if torch.isnan(fut[0, a]).any():
            continue
        P = P_ego if a == 0 else Pj.get(a)
        if P is None:
            continue
        vobs = float(eng._obs_speed(x, a, scale))
        Pn = P.reshape(S * S, K).cpu().numpy()
        pm = Pn[:, K // 2]; pm = pm / (pm.sum() + 1e-12)
        cen = (pos_m * pm[:, None]).sum(0)
        rad = float(np.sqrt((((pos_m - cen) ** 2).sum(1) * pm).sum()))     # density radius (m) mid-horizon
        if learned:
            vfield = vf_ego if a == 0 else Vj.get(a)
            if vfield is None:
                continue
            P2 = P.reshape(S, S, K).cpu().numpy()
            w = P2 / (P2.sum((0, 1), keepdims=True) + 1e-12)
            pv = (vfield * w[..., None]).sum((0, 1))               # (K,2) P-weighted learned velocity
            psp = np.hypot(pv[:, 0], pv[:, 1])
        else:
            cv = eng.centroid_velocity(P, scale); psp = np.hypot(cv[:, 0], cv[:, 1])
        gt = fut[0, a].cpu().numpy() * sca + lon; gv = np.gradient(gt, DT, axis=0); gsp = np.hypot(gv[:, 0], gv[:, 1])
        # FDE: density centroid final position vs GT final position
        pf = Pn[:, -1]; pf = pf / (pf.sum() + 1e-12); cenf = (pos_m * pf[:, None]).sum(0)
        fde = float(np.hypot(*(cenf - gt[-1])))
        velerr = float(np.sqrt(np.mean((psp - gsp) ** 2)))
        gtstd = float(np.std(gsp))
        for bb in bins:
            if bb[0] <= vobs < bb[1]:
                acc[bb]["rad"].append(rad); acc[bb]["fde"].append(fde)
                acc[bb]["velerr"].append(velerr); acc[bb]["gtstd"].append(gtstd); acc[bb]["n"] += 1
                break

print(f"=== model quality vs observed speed ({reg['name']}, {n_used} scenes) ===")
print(f"{'speed bin':12s} {'n':>5s} {'dens_radius(m)':>14s} {'FDE(m)':>8s} {'velerr(m/s)':>11s} {'GT_speed_std':>12s}")
for bb in bins:
    d = acc[bb]
    if d["n"] == 0:
        continue
    print(f"[{bb[0]:>2d},{bb[1]:>2d}) m/s  {d['n']:>5d} {np.mean(d['rad']):>14.2f} {np.mean(d['fde']):>8.2f} "
          f"{np.mean(d['velerr']):>11.2f} {np.mean(d['gtstd']):>12.2f}")
