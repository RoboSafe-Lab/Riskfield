"""Counterfactual objective J(A) on the TWO-MODEL field (regenerates tab:cf).

For candidate ego maneuvers A in {brake, maintain, accelerate} we evaluate
J(A) = sum_k gamma^k sum_g Risk_k(g | A), where the maneuver scales the ego speed
in the kinetic-energy severity term (the physically-grounded counterfactual: braking
lowers collision energy). Reports mean J(A) over scenes, % vs maintain, and the
safest maneuver A* = argmin. Output: RESULT lines.

Env: RF_GRID (48), RF_N_SCENES (20), RF_STRIDE (50), RF_GAMMA (0.95),
     RF_BRAKE (0.7), RF_ACCEL (1.3).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from datasets.InD import InD, boundaries_for_location
from model.RiskFlow import RiskFlow
from riskflow_config import default_dict
from scripts.joint_field import JointRiskField
SDIMS = (JointRiskField.load_scene_dims(os.environ["RF_DIMS"])
         if os.environ.get("RF_DIMS") else None)   # recorded per-agent dims sidecar

S = int(os.environ.get("RF_GRID", "48")); N_SCENES = int(os.environ.get("RF_N_SCENES", "20"))
STRIDE = int(os.environ.get("RF_STRIDE", "50")); GAMMA = float(os.environ.get("RF_GAMMA", "0.95"))
BRAKE = float(os.environ.get("RF_BRAKE", "0.7")); ACCEL = float(os.environ.get("RF_ACCEL", "1.3"))

def build(sl, c):
    return RiskFlow(seq_len=c["seq_len"], input_dim=c["input_dim"], feature_dim=c["feature_dim"],
        embedding_dim=c["embedding_dim"], hidden_dim=c["hidden_dim"], max_num_cars=c["max_num_cars"],
        num_classes=c["num_classes"], gru_layers=c["gru_layers"], num_heads=c["num_heads"], dropout=c["dropout"],
        norm_rotation=c["norm_rotate"], flow_layers=c["flow_layers"], flow_hidden_dim=c["flow_hidden_dim"],
        coupling_layers=c["coupling_layers"], use_cnf=c["use_cnf"], use_cgmm=c["use_cgmm"], gmm_modes=c["gmm_modes"],
        use_world_model=True, wm_state_dim=c["wm_state_dim"], action_dim=c["action_dim"], scene_level=sl,
        use_map=True, map_size=c["map_size"], map_data_dir="data",
        map_local=os.environ.get("RF_MAP_LOCAL", "0").lower() in ("1", "true"),
        map_crop_m=c.get("map_crop_m", 40.0), map_raster_res=c.get("map_raster_res", 192))

def main():
    c = default_dict(); dev = "cuda" if torch.cuda.is_available() else "cpu"; K = c["seq_len"]
    ind = InD(root="data", max_samples=c["maximum_samples"], train_ratio=c["train_ratio"],
              train_batch_size=c["train_batch_size"], test_batch_size=1, missing_rate=c["masked_data_ratio"],
              max_num_cars=c["max_num_cars"], max_empty_frames=c["max_empty_frames"], seq_len=c["seq_len"],
              moving_window=c["seq_len"]*2, sampling_step=c["sampling_step"], should_shuffle=False,
              include_future=c["include_future"])
    site = ind.observation_site_by_scope("all")
    me = build(False, c).to(dev).eval(); me.load_state_dict(torch.load(os.environ.get("RF_CKPT_EGO", "serialized/riskflow_ind_8.pt"), map_location=dev), strict=False)
    mj = build(True, c).to(dev).eval();  mj.load_state_dict(torch.load(os.environ.get("RF_CKPT_JOINT", "serialized/riskflow_ind_7.pt"), map_location=dev), strict=False)
    g1 = torch.linspace(0.05, 0.95, S); GX, GY = torch.meshgrid(g1, g1, indexing="ij")
    grid = torch.stack([GX.reshape(-1), GY.reshape(-1)], -1).to(dev)
    eng = JointRiskField(me, mj, grid, S, K, dev, min_hist=30)
    disc = (GAMMA ** np.arange(K)).astype(np.float32)
    maneuvers = {"brake": BRAKE, "maintain": 1.0, "accelerate": ACCEL}
    J = {m: [] for m in maneuvers}
    n = 0
    with torch.no_grad():
        for i, b in enumerate(site.test_loader):
            if STRIDE > 1 and (i % STRIDE) != 0: continue
            x = b["input"].to(dev); f = b["feature"].to(dev); vt = b["type"].to(dev); fut = b["future"].to(dev)
            if torch.isnan(x[:,0,-2:,:]).any() or torch.isnan(fut[0,0]).any(): continue
            loc = int(b["locationId"].view(-1)[0]); loc_t = torch.tensor([loc], device=dev)
            bx = boundaries_for_location(loc)
            scale = torch.tensor([float(bx[0,1]-bx[0,0]), float(bx[1,1]-bx[1,0])], device=dev)
            neigh = [a for a in range(1, x.shape[1]) if not torch.isnan(x[0,a,-1]).any()
                     and int((~torch.isnan(x[0,a,:,0])).sum()) >= 30]
            if not neigh: continue
            for m, s in maneuvers.items():
                rf = eng.field(x, f, vt, neigh, loc_t, scale, v_ego_scale=s,
                               dims_m=(SDIMS.get(i) if SDIMS else None))   # (S,S,K)
                J[m].append(float((rf.reshape(-1, K).sum(0) * disc).sum()))
            n += 1
            if n >= N_SCENES: break
    base = np.mean(J["maintain"])
    print(f"=== Counterfactual J(A), two-model field, {n} scenes (gamma={GAMMA}) ===")
    order = sorted(maneuvers, key=lambda m: np.mean(J[m]))
    star = order[0]
    for m in ["maintain", "brake", "accelerate"]:
        jm = np.mean(J[m]); pct = (jm - base) / base * 100
        tag = " (A*)" if m == star else ""
        print(f"RESULT {m:11s} J={jm:.4g}  vs_maintain={pct:+.1f}%{tag}")

if __name__ == "__main__":
    main()
