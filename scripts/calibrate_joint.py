"""Global risk-scale calibration for the TWO-MODEL field (ind_8 ego + ind_7 joint).

Uses the SAME field engine as the renderer (joint_field.JointRiskField) so the
calibrated T_critical matches exactly what animate_joint.py displays. Pools the
per-cell expected-collision-energy (J) over every processable test scene and
reports running p50/p90/p99/p99.9 every 250 scenes (saved each checkpoint, so a
time-kill keeps the latest estimate). T_critical = per-cell p99.9 = colorbar vmax.

Output: risk_calibration.json
Env: RF_CKPT_EGO (ind_8), RF_CKPT_JOINT (ind_7), RF_GRID (64), RF_PER_SCENE (2000).
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from datasets.registry import get_dataset  # noqa: E402
_reg = get_dataset()                          # RF_DATASET env (ind|ad4che)
boundaries_for_location = _reg["boundaries_for_location"]
from model.RiskFlow import RiskFlow  # noqa: E402
from riskflow_config import default_dict  # noqa: E402
from scripts.joint_field import JointRiskField
SDIMS = (JointRiskField.load_scene_dims(os.environ["RF_DIMS"])
         if os.environ.get("RF_DIMS") else None)   # recorded per-agent dims sidecar  # noqa: E402

ckpt_ego = os.environ.get("RF_CKPT_EGO", "serialized/riskflow_ind_8.pt")
ckpt_joint = os.environ.get("RF_CKPT_JOINT", "serialized/riskflow_ind_7.pt")
S = int(os.environ.get("RF_GRID", "64"))
PER_SCENE = int(os.environ.get("RF_PER_SCENE", "2000"))
MIN_HIST = int(os.environ.get("RF_MIN_HIST", "30"))
MAX_SCENES = int(os.environ.get("RF_MAX_SCENES", "800"))   # p99.9 converges well before this

c = default_dict()
ind = _reg["LoaderClass"](
    root=_reg["root"], max_samples=c["maximum_samples"], train_ratio=c["train_ratio"],
    train_batch_size=c["train_batch_size"], test_batch_size=1,
    missing_rate=c["masked_data_ratio"], max_num_cars=c["max_num_cars"],
    max_empty_frames=c["max_empty_frames"], seq_len=c["seq_len"],
    moving_window=c["seq_len"] * 2, sampling_step=c["sampling_step"],
    should_shuffle=False, include_future=c["include_future"],
)
site = ind.observation_site_by_scope("all")
dev = "cuda" if torch.cuda.is_available() else "cpu"
K = c["seq_len"]


def _build(scene_level):
    return RiskFlow(
        seq_len=c["seq_len"], input_dim=c["input_dim"], feature_dim=c["feature_dim"],
        embedding_dim=c["embedding_dim"], hidden_dim=c["hidden_dim"],
        max_num_cars=c["max_num_cars"], num_classes=c["num_classes"],
        gru_layers=c["gru_layers"], num_heads=c["num_heads"], dropout=c["dropout"],
        norm_rotation=c["norm_rotate"], flow_layers=c["flow_layers"],
        flow_hidden_dim=c["flow_hidden_dim"], coupling_layers=c["coupling_layers"],
        use_cnf=c["use_cnf"], use_cgmm=c["use_cgmm"], gmm_modes=c["gmm_modes"],
        use_world_model=True, wm_state_dim=c["wm_state_dim"], action_dim=c["action_dim"],
        scene_level=scene_level, use_map=True, map_size=c["map_size"],
        map_data_dir=_reg["map_data_dir"], map_dataset=_reg["map_dataset"],
        map_local=os.environ.get("RF_MAP_LOCAL", "0").lower() in ("1", "true"),
        map_crop_m=c.get("map_crop_m", 40.0), map_raster_res=c.get("map_raster_res", 192),
    ).to(dev)


m_ego = _build(False); m_ego.load_state_dict(torch.load(ckpt_ego, map_location=dev), strict=False); m_ego.eval()
m_joint = _build(True); m_joint.load_state_dict(torch.load(ckpt_joint, map_location=dev), strict=False); m_joint.eval()
print(f"RESULT ego={ckpt_ego}  joint={ckpt_joint}", flush=True)

g1 = torch.linspace(0.05, 0.95, S)
GX, GY = torch.meshgrid(g1, g1, indexing="ij")
grid = torch.stack([GX.reshape(-1), GY.reshape(-1)], -1).to(dev)
engine = JointRiskField(m_ego, m_joint, grid, S, K, dev, min_hist=MIN_HIST)


def save_calibration(samples, used, skipped, final):
    pool = np.concatenate(samples) if samples else np.zeros(1, np.float32)
    p = np.percentile(pool, [50.0, 90.0, 99.0, 99.9, 99.99])
    res = {
        "checkpoint_ego": ckpt_ego, "checkpoint_joint": ckpt_joint,
        "grid_S": S, "min_hist": MIN_HIST,
        "model": "two-model: ind_8 ego marginal (+map) x ind_7 joint ego-conditioned (+map)",
        "vel": "optical-flow transport per-cell velocity field (Horn-Schunck)",
        "scenes_used": used, "scenes_skipped": skipped, "complete": final,
        "pool_size": int(pool.size),
        "units": "joules (expected collision energy, per cell)",
        "cell_p50": float(p[0]), "cell_p90": float(p[1]), "cell_p99": float(p[2]),
        "cell_p99_9": float(p[3]), "cell_p99_99": float(p[4]),
        "cell_max": float(pool.max()),
        "T_critical": float(p[3]),
        "vmax_recommended": float(p[3]),
        "band_benign_max": float(p[1]),
        "band_high_min": float(p[2]),
        "formula": "sum_j P_ego*(P_j^joint*1_box) * 1/2 mu ||v_ego(g,k) - v_j(g,k)||^2; "
                   "P_ego=ind_8 marginal, P_j^joint=ind_7 AR ego-conditioned; per-cell OF velocity",
    }
    with open("risk_calibration.json", "w") as f:
        json.dump(res, f, indent=2)
    tag = "FINAL" if final else f"scenes={used}"
    print(f"RESULT {tag} (skip {skipped})  pool={pool.size}  per-cell[J] "
          f"p50={p[0]:.3g} p90={p[1]:.3g} p99={p[2]:.3g} "
          f"p99.9={p[3]:.3g}(=T_crit) p99.99={p[4]:.3g} max={float(pool.max()):.3g}",
          flush=True)


rng = np.random.default_rng(0)
samples = []
used = skipped = 0
with torch.no_grad():
    for _si, batch in enumerate(site.test_loader):
        x = batch["input"].to(dev)
        feat = batch["feature"].to(dev)
        vt = batch["type"].to(dev)
        ft = batch["future"].to(dev)
        loc = int(batch["locationId"].view(-1)[0])
        if torch.isnan(x[:, 0, -2:, :]).any() or torch.isnan(ft[0, 0]).any():
            skipped += 1
            continue
        neigh = [a for a in range(1, x.shape[1])
                 if not torch.isnan(x[0, a, -1]).any()
                 and int((~torch.isnan(x[0, a, :, 0])).sum()) >= MIN_HIST]
        if not neigh:
            skipped += 1
            continue

        bx = boundaries_for_location(loc)
        scale_t = torch.tensor([float(bx[0, 1] - bx[0, 0]),
                                float(bx[1, 1] - bx[1, 0])],
                               dtype=torch.float32, device=dev)
        loc_t = torch.tensor([loc], device=dev)

        rr = engine.field(x, feat, vt, neigh, loc_t, scale_t, return_dens=False,
                          dims_m=(SDIMS.get(_si) if SDIMS else None), scene_idx=_si)

        flat = rr.reshape(-1)
        idx = rng.choice(flat.shape[0], min(PER_SCENE, flat.shape[0]), replace=False)
        samples.append(flat[idx].astype(np.float32))
        used += 1
        if used % 250 == 0:
            save_calibration(samples, used, skipped, final=False)
        if used >= MAX_SCENES:
            break

save_calibration(samples, used, skipped, final=True)
