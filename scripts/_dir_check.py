"""Does the predicted ego occupancy move WITH the ego (forward) or against it?
For each scene: cosine between the predicted-centroid displacement (k=0 -> k)
and the GT ego displacement over the same horizon. cos ~ +1 forward, -1 reverse.
Env: RF_SCENE_IDS, RF_CKPT_EGO/JOINT, RF_DATASET, RF_DT."""
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
IDS = [int(s) for s in os.environ.get("RF_SCENE_IDS", "78,270,189").split(",")]

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
        if i not in IDS:
            continue
        x = b["input"].to(dev); feat = b["feature"].to(dev); vt = b["type"].to(dev); fut = b["future"].to(dev)
        loc = int(b["locationId"].view(-1)[0]); loc_t = torch.tensor([loc], device=dev)
        bx = reg["boundaries_for_location"](loc)
        sca = np.array([float(bx[0, 1] - bx[0, 0]), float(bx[1, 1] - bx[1, 0])])
        lon = np.array([float(bx[0, 0]), float(bx[1, 0])])
        contrib = [a for a in range(1, x.shape[1]) if not torch.isnan(x[0, a, -1]).any()
                   and int((~torch.isnan(x[0, a, :, 0])).sum()) >= 30]
        scale = torch.tensor(sca, device=dev, dtype=torch.float32)
        rf, dens, _ = eng.field(x, feat, vt, contrib, loc_t, scale, return_dens=True, return_diag=True)
        gm = eng.grid.cpu().numpy() * sca + lon
        P = dens[0].reshape(-1, K)
        cen = np.stack([(P[:, k] / max(P[:, k].sum(), 1e-12)) @ gm for k in range(K)])
        gt = fut[0, 0].cpu().numpy() * sca + lon
        x0 = np.nan_to_num(x[0, 0, -1].cpu().numpy()) * sca + lon          # current pos
        print(f"scene {i}: cur_pos {np.round(x0,1)} cen_k0 {np.round(cen[0],1)}")
        for k in (10, 25, 49):
            dp = cen[k] - cen[0]; dg = gt[min(k, gt.shape[0]-1)] - gt[0]
            cosv = float(dp @ dg / (np.linalg.norm(dp) * np.linalg.norm(dg) + 1e-9))
            print(f"  k={k:2d} pred_disp {np.round(dp,1)} |{np.linalg.norm(dp):5.1f}m  "
                  f"gt_disp {np.round(dg,1)} |{np.linalg.norm(dg):5.1f}m  cos={cosv:+.2f}")
