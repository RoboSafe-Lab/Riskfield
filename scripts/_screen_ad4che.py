"""Screen AD4CHE PET-conflict scenes for prediction-direction sanity, to find a clean
high-energy hero. For each conflict scene, run the field, take the dominant partner,
and compare its PREDICTED centroid velocity to its OBSERVED velocity (cosine + speed
ratio). A scene is CLEAN if the prediction does not flip/over-speed the partner."""
import os, sys; sys.path.insert(0, ".")
import numpy as np, torch
from datasets.registry import get_dataset
from model.RiskFlow import RiskFlow
from riskflow_config import default_dict
from scripts.joint_field import JointRiskField
SDIMS = JointRiskField.load_scene_dims(os.environ["RF_DIMS"]) if os.environ.get("RF_DIMS") else None

DT = float(os.environ.get("RF_DT", "0.0667")); S = int(os.environ.get("RF_GRID", "48"))
STRIDE = int(os.environ.get("RF_STRIDE", "5"))
reg = get_dataset(); c = default_dict(); dev = "cuda" if torch.cuda.is_available() else "cpu"; K = c["seq_len"]
ind = reg["LoaderClass"](root=reg["root"], max_samples=c["maximum_samples"], train_ratio=c["train_ratio"],
    train_batch_size=c["train_batch_size"], test_batch_size=1, missing_rate=c["masked_data_ratio"],
    max_num_cars=c["max_num_cars"], max_empty_frames=c["max_empty_frames"], seq_len=c["seq_len"],
    moving_window=c["seq_len"]*2, sampling_step=c["sampling_step"], should_shuffle=False, include_future=c["include_future"])
site = ind.observation_site_by_scope("all"); bf = reg["boundaries_for_location"]
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
eng = JointRiskField(me, mj, grid, S, K, dev, min_hist=30)
_z = np.load(os.environ["RF_LABELS"], allow_pickle=True); L = _z["data"]
lab = {int(r[0]): int(r[2]) for r in L}
rows = []
with torch.no_grad():
    for i, b in enumerate(site.test_loader):
        if STRIDE > 1 and i % STRIDE != 0: continue
        if lab.get(i, 0) != 1: continue
        x = b["input"].to(dev); f = b["feature"].to(dev); vt = b["type"].to(dev)
        if torch.isnan(x[:, 0, -2:, :]).any(): continue
        loc = int(b["locationId"].view(-1)[0]); loc_t = torch.tensor([loc], device=dev); bx = bf(loc)
        scale = torch.tensor([float(bx[0, 1] - bx[0, 0]), float(bx[1, 1] - bx[1, 0])], device=dev); sca = scale.cpu().numpy()
        neigh = [a for a in range(1, x.shape[1]) if not torch.isnan(x[0, a, -1]).any()
                 and int((~torch.isnan(x[0, a, :, 0])).sum()) >= 30]
        if not neigh: continue
        try:
            rf, diag = eng.field(x, f, vt, neigh, loc_t, scale, return_diag=True,
                                 dims_m=(SDIMS.get(i) if SDIMS else None))
        except Exception:
            continue
        if not diag: continue
        d = max(diag, key=lambda z: z["energy"])
        j = int(d["j"]); vj = np.array(d["vj"]); peakE = float(rf.reshape(-1, K).max())
        xa = x[0, j].cpu().numpy(); vobs = (xa[-1] - xa[-2]) * sca / DT
        e0 = x[0, 0].cpu().numpy(); vego = (e0[-1] - e0[-2]) * sca / DT
        nvj = float(np.linalg.norm(vj)); nvo = float(np.linalg.norm(vobs)); nve = float(np.linalg.norm(vego))
        cos = float(vj @ vobs / (nvj * nvo + 1e-9)); ratio = nvj / (nvo + 1e-9)
        clean = (cos > 0.5) and (0.5 < ratio < 2.0) and (nvo > 2.0)
        rows.append((i, peakE, j, nve, nvo, nvj, cos, ratio, clean))
        print(f"RESULT scene={i} peakE={peakE:.0f}J j={j} ego_v={nve:.1f} part_vobs={nvo:.1f} "
              f"part_vpred={nvj:.1f} cos={cos:+.2f} ratio={ratio:.2f} {'CLEAN' if clean else 'BAD'}")
rows.sort(key=lambda r: -r[1])
print(f"=== {sum(1 for r in rows if r[8])}/{len(rows)} conflict scenes are direction-consistent (CLEAN) ===")
print("=== TOP CLEAN by genuine peak energy ===")
for r in [r for r in rows if r[8]][:10]:
    print(f"RESULT_CLEAN scene={r[0]} peakE={r[1]:.0f}J j={r[2]} ego_v={r[3]:.1f} "
          f"part_vobs={r[4]:.1f} part_vpred={r[5]:.1f} cos={r[6]:+.2f} ratio={r[7]:.2f}")
print("SCREEN_DONE")
