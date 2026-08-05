"""Animate the spatial-temporal risk field over the real InD intersection.

Picks a real all-33 test scene and renders a side-by-side animated GIF for
three candidate ego maneuvers (maintain / brake / accelerate). Each panel
shows, registered on the recording's drone background image:

  - the multi-agent risk field  Risk_k(g) = sum_j P_j(g,k) * C_j(k)
    overlaid as a translucent heatmap (footprint-dilated; paper Eq. dilation);
  - every agent in the scene, animated along its trajectory (ground truth for
    surrounding agents; the candidate maneuver path for the ego);
  - the ego's candidate path (cyan).

WITHIN a panel = temporal evolution; ACROSS panels = the physically grounded
counterfactual (the maneuver reshapes the kinetic-energy severity term C_j).

Env: RF_CKPT (default serialized/riskflow_ind_5.pt), RF_GRID (64), RF_FPS (10).
Output: riskfield_counterfactual.gif  (+ riskfield_tensors.npz)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from datasets.InD import InD, boundaries_for_location  # noqa: E402
from model.RiskFlow import RiskFlow  # noqa: E402
from riskflow_config import default_dict  # noqa: E402

ckpt = os.environ.get("RF_CKPT", "serialized/riskflow_ind_5.pt")
S = int(os.environ.get("RF_GRID", "64"))
FPS = int(os.environ.get("RF_FPS", "10"))
DT, M_R, GAMMA = 0.08, 1500.0 / 2.0, 0.95
DILATE_SIGMA, GAMMA_DISP = 2.0, 0.55

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


def base_logpx(z, det):
    d = z.shape[-1]
    return -0.5 * (z.pow(2).sum(-1) + d * np.log(2 * np.pi)) - det


def roll_to_front(t, j):
    return torch.roll(t, shifts=-j, dims=1)


def agent_density(x, feat, vt, j):
    xr, fr, vr = roll_to_front(x, j), roll_to_front(feat, j), roll_to_front(vt, j)
    emb, _ = m.encoder(None, torch.cat([xr, fr], -1), vr, per_agent=False)
    cond = m._flow_condition(emb, K, None)
    condG = cond.expand(G, K, cond.shape[-1]) if cond.dim() == 3 \
        else cond.expand(G, cond.shape[-1])
    y = grid.view(G, 1, 2).expand(G, K, 2).contiguous()
    z, det = m.flow(y, condG, sampling_frequency=1)
    P = base_logpx(z, det).exp()
    return P / P.sum(0, keepdim=True).clamp(min=1e-9)


with torch.no_grad():
    chosen = None
    for batch in site.test_loader:
        x = batch["input"].to(dev)
        feat = batch["feature"].to(dev)
        vt = batch["type"].to(dev)
        fut = batch["future"].to(dev)
        if torch.isnan(x[:, 0, -2:, :]).any():
            continue
        valid = [a for a in range(x.shape[1])
                 if not torch.isnan(x[0, a, -1]).any()]
        neigh = [a for a in valid if a != 0
                 and not torch.isnan(fut[0, a]).any()]
        if len(neigh) >= 2:
            chosen = (x, feat, vt, fut, int(batch["locationId"].view(-1)[0]),
                      valid, neigh)
            break
    assert chosen is not None, "no suitable scene found"
    x, feat, vt, fut, loc, valid, neigh = chosen
    print(f"scene loc={loc} valid_agents={valid} risk_neighbours={neigh}")

    bx = boundaries_for_location(loc)
    xlo, xhi = float(bx[0, 0]), float(bx[0, 1])
    ylo, yhi = float(bx[1, 0]), float(bx[1, 1])
    scale = torch.tensor([xhi - xlo, yhi - ylo], device=dev)
    lo = torch.tensor([xlo, ylo], device=dev)

    def to_m(norm):                                  # normalized -> metres
        return (norm * scale + lo).cpu().numpy()

    Pj, Vj = {}, {}
    for j in neigh:
        P = agent_density(x, feat, vt, j)
        Pj[j] = P
        cen = (P.t().unsqueeze(-1) * grid.unsqueeze(0)).sum(1) * scale
        v = torch.zeros(K, 2, device=dev)
        v[:-1] = (cen[1:] - cen[:-1]) / DT
        v[-1] = v[-2]
        Vj[j] = v

    ego_hist = x[0, 0, -2:, :] * scale + lo
    v0 = (ego_hist[1] - ego_hist[0]) / DT
    sp0 = float(v0.norm().clamp(min=1.0))
    dirv = v0 / v0.norm().clamp(min=1e-6)
    p0 = ego_hist[1]
    t = torch.arange(K, device=dev, dtype=torch.float32)
    speeds = {
        "Maintain": torch.full((K,), sp0, device=dev),
        "Brake": (sp0 - 3.0 * t * DT).clamp(min=0.0),
        "Accelerate": sp0 + 1.5 * t * DT,
    }
    risk, ego_m = {}, {}
    for nm, sp in speeds.items():
        v_ego = sp.unsqueeze(-1) * dirv.unsqueeze(0)
        R = torch.zeros(G, K, device=dev)
        for j in neigh:
            C = 0.5 * M_R * (v_ego - Vj[j]).pow(2).sum(-1)
            R = R + Pj[j] * C.unsqueeze(0)
        risk[nm] = R.reshape(S, S, K).cpu().numpy()
        ego_m[nm] = (p0.unsqueeze(0) + torch.cumsum(v_ego * DT, 0)).cpu().numpy()

    # agent trajectories in metres: ground-truth future for surrounding agents
    agent_fut = {a: to_m(fut[0, a]) for a in neigh}
    agent_hist = {a: to_m(x[0, a]) for a in valid}
    agent_type = {a: int(vt[0, a]) for a in valid}

names = list(speeds.keys())
for nm in names:
    for k in range(K):
        risk[nm][:, :, k] = gaussian_filter(risk[nm][:, :, k], DILATE_SIGMA)
vmax = max(float(risk[nm].max()) for nm in names)
disc = GAMMA ** np.arange(K)
Jcum = {nm: np.cumsum(disc * risk[nm].sum(axis=(0, 1))) for nm in names}
np.savez("riskfield_tensors.npz", **{n: risk[n] for n in names}, loc=loc)

# ---- background image + coordinate registration -------------------------
# InD's raw orthoPxToMeter underscales the background by ~11.5x relative to
# the track frame; the repo's interactive_field.py uses the same fudge.
FUDGE = 11.5
rec = InD.LOCATION_RECORDINGS[loc][0]
bg, bg_extent, Wm, Hm = None, None, None, None
try:
    o = float(pd.read_csv(f"data/{rec}_recordingMeta.csv").at[0, "orthoPxToMeter"]) * FUDGE
    bg = mpimg.imread(f"data/{rec}_background.png")
    H, W = bg.shape[0], bg.shape[1]
    Wm, Hm = W * o, H * o                          # image span in metres
    bg_extent = [0.0, Wm, -Hm, 0.0]                # px=x/o, py=-y/o (y negated)
    print(f"RESULT background {W}x{H}px  span {Wm:.0f}x{Hm:.0f}m")
except Exception as e:                              # graceful fallback
    print(f"RESULT background unavailable ({e}); using plain backdrop")

# crop view to the location box, clipped to the image so there are no margins
if bg is not None:
    vx0, vx1 = max(xlo, 0.0), min(xhi, Wm)
    vy0, vy1 = max(ylo, -Hm), min(yhi, 0.0)
else:
    vx0, vx1, vy0, vy1 = xlo, xhi, ylo, yhi

cmap = matplotlib.colormaps["inferno"]


def rgba(frame):
    d = np.clip(frame / max(vmax, 1e-9), 0, 1) ** GAMMA_DISP
    img = cmap(d)
    img[..., 3] = np.clip(d * 1.5, 0, 0.85)         # low risk -> transparent
    return img


# ---- render -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))
TYPE_C = {0: "#39FF14", 1: "#FF8C00", 2: "#FFFF33", 3: "#BBBBBB"}
ims, egodots, egolines, ttls = [], [], [], []
agdots, agtrails = [], []
for ax, nm in zip(axes, names):
    if bg is not None:
        ax.imshow(bg, extent=bg_extent, origin="upper", zorder=0)
    else:
        ax.set_facecolor("black")
    im = ax.imshow(rgba(risk[nm][:, :, 0].T), extent=[xlo, xhi, ylo, yhi],
                   origin="lower", zorder=1, animated=True)
    ax.plot(ego_m[nm][:, 0], ego_m[nm][:, 1], "-", color="cyan", lw=1.4,
            alpha=0.5, zorder=2)
    eline, = ax.plot([], [], "-", color="cyan", lw=2.4, zorder=3)
    edot, = ax.plot([], [], "o", color="cyan", ms=9, mec="white", zorder=5)
    # surrounding agents
    dots, trails = {}, {}
    for a in neigh:
        col = TYPE_C.get(agent_type[a], "#FF8C00")
        tr, = ax.plot([], [], "-", color=col, lw=1.6, alpha=0.7, zorder=3)
        dt, = ax.plot([], [], "s", color=col, ms=7, mec="white", zorder=5)
        trails[a] = tr; dots[a] = dt
    ttl = ax.set_title(nm, fontsize=12)
    ax.set_xlim(vx0, vx1); ax.set_ylim(vy0, vy1)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ims.append(im); egolines.append(eline); egodots.append(edot)
    agdots.append(dots); agtrails.append(trails); ttls.append(ttl)
sup = fig.suptitle("", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])


def update(k):
    arts = []
    for p, nm in enumerate(names):
        ims[p].set_data(rgba(risk[nm][:, :, k].T))
        egolines[p].set_data(ego_m[nm][:k + 1, 0], ego_m[nm][:k + 1, 1])
        egodots[p].set_data([ego_m[nm][k, 0]], [ego_m[nm][k, 1]])
        ttls[p].set_text(f"{nm}   (cumulative risk J = {Jcum[nm][k]:.2e})")
        for a in neigh:
            tr = agent_fut[a][:k + 1]
            agtrails[p][a].set_data(tr[:, 0], tr[:, 1])
            agdots[p][a].set_data([agent_fut[a][k, 0]], [agent_fut[a][k, 1]])
        arts += [ims[p], egolines[p], egodots[p], ttls[p]]
    sup.set_text(f"Spatial-temporal risk field on the InD intersection  --  "
                 f"step k = {k+1}/{K}  ($\\Delta t$={DT}s).  "
                 f"Cyan = ego (candidate plan); squares = surrounding agents.")
    return arts


anim = FuncAnimation(fig, update, frames=K, interval=1000 // FPS, blit=False)
out = "riskfield_counterfactual.gif"
anim.save(out, writer=PillowWriter(fps=FPS))
final = {nm: Jcum[nm][-1] for nm in names}
print(f"RESULT saved {out}  ({K} frames, {S}x{S}, loc={loc}, rec={rec})")
print("RESULT final J: " + "  ".join(f"{n}={final[n]:.3e}" for n in names) +
      f"   safest={min(final, key=final.get)}")
