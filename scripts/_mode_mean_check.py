"""Ego occupancy: mode vs mean per estrip frame (is the box ahead of the cloud?)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from datasets.registry import get_dataset
from model.RiskFlow import RiskFlow
from riskflow_config import default_dict
from scripts.joint_field import JointRiskField

reg = get_dataset(); c = default_dict(); dev = "cuda" if torch.cuda.is_available() else "cpu"
S = 64; K = c["seq_len"]
SC = int(os.environ.get("RF_SCENE_IDX", "530"))
FRAMES = [int(s) for s in os.environ.get("RF_FRAMES", "0,7,25").split(",")]

ind = reg["LoaderClass"](root=reg["root"], max_samples=c["maximum_samples"], train_ratio=c["train_ratio"],
    train_batch_size=c["train_batch_size"], test_batch_size=1, missing_rate=c["masked_data_ratio"],
    max_num_cars=c["max_num_cars"], max_empty_frames=c["max_empty_frames"], seq_len=c["seq_len"],
    moving_window=c["seq_len"] * 2, sampling_step=c["sampling_step"], should_shuffle=False,
    include_future=c["include_future"])
site = ind.observation_site_by_scope("all")


def build(sl):
    return RiskFlow(seq_len=c["seq_len"], input_dim=c["input_dim"], feature_dim=c["feature_dim"],
        embedding_dim=c["embedding_dim"], hidden_dim=c["hidden_dim"], max_num_cars=c["max_num_cars"],
        num_classes=c["num_classes"], gru_layers=c["gru_layers"], num_heads=c["num_heads"], dropout=c["dropout"],
        norm_rotation=c["norm_rotate"], flow_layers=c["flow_layers"], flow_hidden_dim=c["flow_hidden_dim"],
        coupling_layers=c["coupling_layers"], use_cnf=c["use_cnf"], use_cgmm=c["use_cgmm"], gmm_modes=c["gmm_modes"],
        use_world_model=True, wm_state_dim=c["wm_state_dim"], action_dim=c["action_dim"], scene_level=sl,
        use_map=True, map_size=c["map_size"], map_data_dir=reg["map_data_dir"],
        map_dataset=reg["map_dataset"]).to(dev).eval()


me = build(False); me.load_state_dict(torch.load(os.environ["RF_CKPT_EGO"], map_location=dev), strict=False)
mj = build(True);  mj.load_state_dict(torch.load(os.environ["RF_CKPT_JOINT"], map_location=dev), strict=False)
g1 = torch.linspace(0.05, 0.95, S); GX, GY = torch.meshgrid(g1, g1, indexing="ij")
grid = torch.stack([GX.reshape(-1), GY.reshape(-1)], -1).to(dev)
eng = JointRiskField(me, mj, grid, S, K, dev, min_hist=30)

with torch.no_grad():
    for i, b in enumerate(site.test_loader):
        if i != SC:
            continue
        x = b["input"].to(dev); feat = b["feature"].to(dev); vt = b["type"].to(dev); fut = b["future"].to(dev)
        loc = int(b["locationId"].view(-1)[0]); loc_t = torch.tensor([loc], device=dev)
        bx = reg["boundaries_for_location"](loc)
        sca = np.array([float(bx[0, 1] - bx[0, 0]), float(bx[1, 1] - bx[1, 0])])
        lon = np.array([float(bx[0, 0]), float(bx[1, 0])])
        scale = torch.tensor(sca, device=dev, dtype=torch.float32)
        contrib = [a for a in range(1, x.shape[1]) if not torch.isnan(x[0, a, -1]).any()
                   and int((~torch.isnan(x[0, a, :, 0])).sum()) >= 30]
        rf, dens, _ = eng.field(x, feat, vt, contrib, loc_t, scale, return_dens=True, return_diag=True)
        g1n = g1.numpy(); Xg = g1n * sca[0] + lon[0]; Yg = g1n * sca[1] + lon[1]
        MX, MY = np.meshgrid(Xg, Yg, indexing="ij")
        cur = np.nan_to_num(x[0, 0, -1].cpu().numpy()) * sca + lon
        gt = fut[0, 0].cpu().numpy() * sca + lon
        print(f"scene {i}: ego current x={cur[0]:.1f}  GT x at k=7/25: "
              f"{gt[7,0]:.1f}/{gt[25,0]:.1f}")
        for kk in FRAMES:
            P = dens[0][:, :, kk]
            iidx = np.unravel_index(int(P.argmax()), P.shape)
            mode = (MX[iidx], MY[iidx])
            s = P.sum(); mean = ((MX * P).sum() / s, (MY * P).sum() / s)
            # x-marginal quantiles (mass along the lane)
            px = P.sum(axis=1) / s
            cdf = np.cumsum(px)
            q = [float(np.interp(t, cdf, Xg)) for t in (0.1, 0.5, 0.9)]
            print(f"  k={kk:2d} mode=({mode[0]:.1f},{mode[1]:.1f})  mean=({mean[0]:.1f},{mean[1]:.1f})  "
                  f"mean-mode_x={mean[0]-mode[0]:+.1f}m  x q10/50/90={q[0]:.1f}/{q[1]:.1f}/{q[2]:.1f}  "
                  f"gt_x={gt[min(kk,gt.shape[0]-1),0]:.1f}", flush=True)
        break
