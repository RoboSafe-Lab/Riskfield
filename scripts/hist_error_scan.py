"""Population scan: model prediction error vs. observed-history completeness.

For every valid agent in the first RF_SCENES test scenes, compute the agent's
number of valid history frames and the model's centroid prediction error vs.
ground truth (at k=0 and mean over the horizon). Bin by history count and
report mean/median error per bin, so we can see whether sparse-history agents
are systematically mispredicted (and pick a gating threshold for the risk
field). Env: RF_CKPT (default serialized/riskflow_ind_5.pt), RF_GRID (64),
RF_SCENES (400).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from datasets.InD import InD, boundaries_for_location  # noqa: E402
from model.RiskFlow import RiskFlow  # noqa: E402
from riskflow_config import default_dict  # noqa: E402

ckpt = os.environ.get("RF_CKPT", "serialized/riskflow_ind_5.pt")
S = int(os.environ.get("RF_GRID", "64"))
SCENES = int(os.environ.get("RF_SCENES", "400"))
DT = 0.08

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
    scene_level=False,
).to(dev)
m.load_state_dict(torch.load(ckpt, map_location=dev), strict=False)
m.eval()
K = c["seq_len"]

g1 = torch.linspace(0.05, 0.95, S)
GX, GY = torch.meshgrid(g1, g1, indexing="ij")
grid = torch.stack([GX.reshape(-1), GY.reshape(-1)], -1).to(dev)
G = grid.shape[0]
Y_GRID = grid.view(G, 1, 2).expand(G, K, 2).contiguous()


def base_logpx(z, det):
    d = z.shape[-1]
    return -0.5 * (z.pow(2).sum(-1) + d * np.log(2 * np.pi)) - det


def _cond_grid(x, feat, vt, a):
    xr = torch.roll(x, -a, 1); fr = torch.roll(feat, -a, 1); vr = torch.roll(vt, -a, 1)
    emb, _ = m.encoder(None, torch.cat([xr, fr], -1), vr, per_agent=False)
    cond = m._flow_condition(emb, K, None)
    if cond.dim() == 2:
        cond = cond.unsqueeze(1).expand(-1, K, -1)
    return cond.expand(G, K, cond.shape[-1]).contiguous()


def centroid(x, feat, vt, a):
    z, det = m.flow(Y_GRID, _cond_grid(x, feat, vt, a), sampling_frequency=1)
    P = base_logpx(z, det).exp()
    P = P / P.sum(0, keepdim=True).clamp(min=1e-9)
    return torch.einsum("gd,gk->kd", grid, P)                  # (K,2) normalized


rows = []  # (hist_count, k0_err_m, mean_err_m, gt_speed)
seen = 0
with torch.no_grad():
    for batch in site.test_loader:
        if seen >= SCENES:
            break
        seen += 1
        x = batch["input"].to(dev)
        feat = batch["feature"].to(dev)
        vt = batch["type"].to(dev)
        ft = batch["future"].to(dev)
        loc = int(batch["locationId"].view(-1)[0])
        bx = boundaries_for_location(loc)
        scale = torch.tensor([float(bx[0, 1] - bx[0, 0]),
                              float(bx[1, 1] - bx[1, 0])], device=dev)
        lo = torch.tensor([float(bx[0, 0]), float(bx[1, 0])], device=dev)
        for a in range(x.shape[1]):
            if torch.isnan(x[0, a, -1]).any() or torch.isnan(ft[0, a]).any():
                continue
            nhist = int((~torch.isnan(x[0, a, :, 0])).sum())
            cen = centroid(x, feat, vt, a) * scale + lo            # (K,2) m
            gt = ft[0, a] * scale + lo                             # (K,2) m
            err = (cen - gt).pow(2).sum(-1).sqrt()                 # (K,)
            gspd = float((gt[1:] - gt[:-1]).pow(2).sum(-1).sqrt().mean() / DT)
            rows.append((nhist, float(err[0]), float(err.mean()), gspd))

rows = np.array(rows, dtype=np.float32)
print(f"RESULT agents={len(rows)} over {seen} scenes")
# bin by history count
edges = [0, 5, 10, 20, 30, 40, 49, 51]
print("\n hist-bin    n   k0_err(med/mean)   horizon_err(med/mean)   gt_speed(mean)")
for i in range(len(edges) - 1):
    lo_e, hi_e = edges[i], edges[i + 1]
    sel = rows[(rows[:, 0] >= lo_e) & (rows[:, 0] < hi_e)]
    if len(sel) == 0:
        continue
    print(f"  [{lo_e:2d},{hi_e:2d})  {len(sel):4d}   "
          f"{np.median(sel[:,1]):6.1f}/{sel[:,1].mean():6.1f} m      "
          f"{np.median(sel[:,2]):6.1f}/{sel[:,2].mean():6.1f} m       "
          f"{sel[:,3].mean():4.1f} m/s")
# correlation
if len(rows) > 2:
    cc = np.corrcoef(rows[:, 0], rows[:, 1])[0, 1]
    cs = np.corrcoef(rows[:, 3], rows[:, 1])[0, 1]
    print(f"\n corr(hist_count, k0_err)  = {cc:+.3f}")
    print(f" corr(gt_speed,   k0_err)  = {cs:+.3f}")
# well-observed subset
full = rows[rows[:, 0] >= 49]
print(f"\n full-history agents (>=49/50): n={len(full)}  "
      f"k0_err median={np.median(full[:,1]):.1f}m mean={full[:,1].mean():.1f}m  "
      f"p90={np.percentile(full[:,1],90):.1f}m")
