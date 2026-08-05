"""Per-CYCLE direction check mirroring animate_joint's replan windows: for each
cycle present p0 (49, 49+NEAR, ...), rebuild the spliced GT window exactly as the
animation does and measure the ego predicted-centroid displacement vs GT.
Env: RF_SCENE_IDX, RF_CKPT_EGO/JOINT, RF_DATASET, RF_DT, RF_NEAR, RF_CYCLES."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from datasets.registry import get_dataset
from model.RiskFlow import RiskFlow
from riskflow_config import default_dict
from scripts.joint_field import JointRiskField

reg = get_dataset(); c = default_dict(); dev = "cuda" if torch.cuda.is_available() else "cpu"
S = 64; K = c["seq_len"]; DT = float(os.environ.get("RF_DT", "0.08"))
SC = int(os.environ["RF_SCENE_IDX"]); NEAR = int(os.environ.get("RF_NEAR", "6"))
NCYC = int(os.environ.get("RF_CYCLES", "6"))
feature_boundaries = reg["feature_boundaries"]

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
        lo = torch.tensor(lon, device=dev, dtype=torch.float32)
        contributors = [a for a in range(1, x.shape[1]) if not torch.isnan(x[0, a, -1]).any()
                        and int((~torch.isnan(x[0, a, :, 0])).sum()) >= 30]
        # ---- spliced GT timeline, EXACTLY as animate_joint builds it ----
        Th = x.shape[2]; Mtot = Th + K
        pos_comb = torch.cat([x[0], fut[0]], dim=1)
        posm_all = (pos_comb * scale + lo).cpu().numpy()
        velm_all = np.gradient(posm_all, DT, axis=1)
        accm_all = np.gradient(velm_all, DT, axis=1)
        head_all = np.degrees(np.arctan2(velm_all[..., 1], velm_all[..., 0])) % 360.0
        feat5 = np.stack([head_all, velm_all[..., 0], velm_all[..., 1],
                          accm_all[..., 0], accm_all[..., 1]], axis=-1)
        fb = feature_boundaries
        feat5n = (feat5 - fb[:, 0]) / (fb[:, 1] - fb[:, 0])
        Cf = feat.shape[-1]
        ch_extra = feat[0, :, -1, 5:Cf].cpu().numpy()
        ch_extra = np.repeat(ch_extra[:, None, :], Mtot, axis=1)
        featnorm = np.concatenate([feat5n, ch_extra], axis=-1)
        featnorm[:, :Th, :] = feat[0].cpu().numpy()
        POS = pos_comb.unsqueeze(0)
        FEATn = torch.tensor(featnorm, dtype=torch.float32, device=dev).unsqueeze(0)
        gm = eng.grid.cpu().numpy() * sca + lon
        for ci in range(NCYC):
            p0 = (Th - 1) + ci * NEAR
            if p0 + NEAR > Mtot - 1:
                break
            xw = POS[:, :, p0 - Th + 1:p0 + 1, :].contiguous()
            fw = FEATn[:, :, p0 - Th + 1:p0 + 1, :].contiguous()
            rf, dens = eng.field(xw, fw, vt, contributors, loc_t, scale, return_dens=True)
            P = dens[0].reshape(-1, K)
            cen = np.stack([(P[:, k] / max(P[:, k].sum(), 1e-12)) @ gm for k in range(K)])
            cur = posm_all[0, p0]
            kmax = min(25, Mtot - 1 - p0)
            dg = posm_all[0, min(p0 + 25, Mtot - 1)] - cur          # GT disp over same horizon
            dp = cen[25] - cen[0]
            cosv = float(dp @ dg / (np.linalg.norm(dp) * np.linalg.norm(dg) + 1e-9))
            off0 = np.linalg.norm(cen[0] - cur)
            print(f"cycle {ci+1} p0={p0}: cen_k0_offset={off0:5.1f}m  "
                  f"pred_disp(k25) {np.round(dp,1)} |{np.linalg.norm(dp):5.1f}m  "
                  f"gt_disp {np.round(dg,1)} |{np.linalg.norm(dg):5.1f}m  cos={cosv:+.2f}", flush=True)
        break
