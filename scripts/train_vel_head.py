"""Train ONLY the velocity head with the rest of the model FROZEN, so the
occupancy density (and detection) are bit-identical. Velocity target = per-frame
normalized displacement of the GT future positions (already in the cache -> no
loader/cache change). Saves a checkpoint = frozen backbone + trained head.

Env:
  RF_VHEAD_MODE  ego|joint   (ego = single-target flow; joint = scene-level)
  RF_CKPT_IN     checkpoint to load (backbone)
  RF_CKPT_OUT    checkpoint to save (backbone + trained head)
  RF_VEPOCHS (8), RF_VLR (1e-3), RF_DATASET, RF_GRID unused here.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from datasets.registry import get_dataset
from model.RiskFlow import RiskFlow
from riskflow_config import default_dict

reg = get_dataset(); c = default_dict(); dev = "cuda" if torch.cuda.is_available() else "cpu"
MODE = os.environ.get("RF_VHEAD_MODE", "joint")
CKPT_IN = os.environ["RF_CKPT_IN"]; CKPT_OUT = os.environ["RF_CKPT_OUT"]
EPOCHS = int(os.environ.get("RF_VEPOCHS", "8")); LR = float(os.environ.get("RF_VLR", "1e-3"))
DT = float(os.environ.get("RF_DT", "0.0667"))
scene_level = (MODE == "joint")

ind = reg["LoaderClass"](root=reg["root"], max_samples=c["maximum_samples"], train_ratio=c["train_ratio"],
    train_batch_size=c["train_batch_size"], test_batch_size=1, missing_rate=c["masked_data_ratio"],
    max_num_cars=c["max_num_cars"], max_empty_frames=c["max_empty_frames"], seq_len=c["seq_len"],
    moving_window=c["seq_len"] * 2, sampling_step=c["sampling_step"], should_shuffle=True, include_future=c["include_future"])
site = ind.observation_site_by_scope("all")

model = RiskFlow(seq_len=c["seq_len"], input_dim=c["input_dim"], feature_dim=c["feature_dim"],
    embedding_dim=c["embedding_dim"], hidden_dim=c["hidden_dim"], max_num_cars=c["max_num_cars"],
    num_classes=c["num_classes"], gru_layers=c["gru_layers"], num_heads=c["num_heads"], dropout=c["dropout"],
    norm_rotation=c["norm_rotate"], flow_layers=c["flow_layers"], flow_hidden_dim=c["flow_hidden_dim"],
    coupling_layers=c["coupling_layers"], use_cnf=c["use_cnf"], use_cgmm=c["use_cgmm"], gmm_modes=c["gmm_modes"],
    use_world_model=True, wm_state_dim=c["wm_state_dim"], action_dim=c["action_dim"], scene_level=scene_level,
    use_map=True, map_size=c["map_size"], map_data_dir=reg["map_data_dir"], map_dataset=reg["map_dataset"]).to(dev)
missing, unexpected = model.load_state_dict(torch.load(CKPT_IN, map_location=dev), strict=False)
print(f"loaded {CKPT_IN}; fresh head params: {sum('velocity_head' in k for k in missing)}", flush=True)
model.eval()                                   # freeze dropout/BN in the backbone
for n, p in model.named_parameters():
    p.requires_grad = n.startswith("velocity_head")
opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)


def vel_residual_ms(fut, scl):                 # fut (...,K,2) normalized; scl (B,2) m -> Δ (m/s)
    v = torch.zeros_like(fut)
    v[..., :-1, :] = fut[..., 1:, :] - fut[..., :-1, :]
    v[..., -1, :] = v[..., -2, :]
    s = scl
    while s.dim() < v.dim():                    # (B,2)->(B,1,2) or (B,1,1,2)
        s = s.unsqueeze(1)
    v_ms = v * s / DT                           # per-frame velocity in m/s
    return v_ms - v_ms[..., 0:1, :]             # residual relative to the initial frame


def batch_scale(loc_t):                         # (B,) loc ids -> (B,2) box size in metres
    rows = []
    for l in loc_t.view(-1).tolist():
        bx = bf(int(l)); rows.append([float(bx[0, 1] - bx[0, 0]), float(bx[1, 1] - bx[1, 0])])
    return torch.tensor(rows, device=dev)


bf = reg["boundaries_for_location"]
for ep in range(EPOCHS):
    tot = 0.0; nb = 0
    for b in site.train_loader:
        x = b["input"].to(dev); feat = b["feature"].to(dev); vt = b["type"].to(dev)
        if "future" not in b:
            continue
        loc = b["locationId"].to(dev); scl = batch_scale(loc)
        if scene_level:
            y = b["future"].to(dev)            # (B,N,K,2)
        else:
            y = b["target"].to(dev)            # (B,K,2) ego
        out = model(x, y, feat, vt, return_vel=True, location_id=loc)
        vel_pred = out[-1]                      # learned m/s residual
        vg = vel_residual_ms(y, scl)           # target m/s residual
        mask = (~torch.isnan(y).any(dim=-1, keepdim=True)).float()
        vg = torch.nan_to_num(vg); vel_pred = torch.nan_to_num(vel_pred)
        loss = ((vel_pred - vg) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss.detach()); nb += 1
    print(f"epoch {ep} vel_mse(m/s^2)={tot / max(nb, 1):.4f}", flush=True)

torch.save(model.state_dict(), CKPT_OUT)
print(f"saved {CKPT_OUT}", flush=True)
