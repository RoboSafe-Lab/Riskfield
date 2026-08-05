"""Receding-horizon (real-time re-forecast) risk-field animations.

For each example, slide a real-time window along one ego track: at every real
frame the model re-observes the updated history and re-forecasts the risk
field. Successive animation frames are therefore successive *re-predictions*
("another evolution each frame") -- the deployed-system view. No counterfactual.

Per real frame the displayed risk field is the discounted horizon sum
    Risk(g) = sum_k gamma^k * sum_j P_j(g,k) * C_j(k)
overlaid on the recording's drone background, with every agent drawn at its
current position.

Produces RF_N (default 4) GIFs: riskfield_evolution_{i}.gif.
Env: RF_CKPT (default serialized/riskflow_ind_5.pt), RF_GRID (64),
     RF_REAL_FRAMES (45), RF_FPS (8).
"""

import os
import sys
import functools

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
REAL_FRAMES = int(os.environ.get("RF_REAL_FRAMES", "45"))
FPS = int(os.environ.get("RF_FPS", "8"))
DT, M_R, GAMMA = 0.08, 1500.0 / 2.0, 0.95
DILATE_SIGMA, GAMMA_DISP, FUDGE = 2.0, 0.55, 11.5
# one recording per InD location for variety
SITES = ["08", "18", "30", "00"]

c = default_dict()
ind = InD(
    root="data", max_samples=c["maximum_samples"], train_ratio=c["train_ratio"],
    train_batch_size=c["train_batch_size"], test_batch_size=1,
    missing_rate=0.0, max_num_cars=c["max_num_cars"],
    max_empty_frames=c["max_empty_frames"], seq_len=c["seq_len"],
    moving_window=c["seq_len"] * 2, sampling_step=c["sampling_step"],
    should_shuffle=False, include_future=c["include_future"],
)
# cache the per-recording CSV load so consecutive get_specific_sample calls
# don't re-read from disk.
_orig_load = ind._load_and_clean_data
_cache = functools.lru_cache(maxsize=8)(lambda s: _orig_load(s))
ind._load_and_clean_data = _cache

dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = default_dict()
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
cmap = matplotlib.colormaps["inferno"]


def base_logpx(z, det):
    d = z.shape[-1]
    return -0.5 * (z.pow(2).sum(-1) + d * np.log(2 * np.pi)) - det


def agent_density(x, feat, vt, j):
    """Autonomous per-step density of agent j on the grid (index-rotated)."""
    xr = torch.roll(x, -j, 1); fr = torch.roll(feat, -j, 1); vr = torch.roll(vt, -j, 1)
    emb, _ = m.encoder(None, torch.cat([xr, fr], -1), vr, per_agent=False)
    cond = m._flow_condition(emb, K, None)
    condG = cond.expand(G, K, cond.shape[-1]) if cond.dim() == 3 \
        else cond.expand(G, cond.shape[-1])
    y = grid.view(G, 1, 2).expand(G, K, 2).contiguous()
    z, det = m.flow(y, condG, sampling_frequency=1)
    P = base_logpx(z, det).exp()
    return P / P.sum(0, keepdim=True).clamp(min=1e-9)


def pick_long_ego(site):
    """Longest target-class ego track in the recording."""
    _, _, tmeta = ind._load_and_clean_data(site)
    cand = tmeta[tmeta["class"].isin(ind.target_classes)]
    cand = cand[cand["numFrames"] >= (ind.moving_window + REAL_FRAMES) * ind.sampling_step]
    if len(cand) == 0:
        cand = tmeta[tmeta["class"].isin(ind.target_classes)]
    row = cand.sort_values("numFrames", ascending=False).iloc[0]
    return int(row["trackId"])


def render(site, ego, risk_seq, agents_seq, ego_seq, loc, out):
    bx = boundaries_for_location(loc)
    xlo, xhi = float(bx[0, 0]), float(bx[0, 1])
    ylo, yhi = float(bx[1, 0]), float(bx[1, 1])
    rec = InD.LOCATION_RECORDINGS[loc][0]
    bg, Wm, Hm = None, None, None
    try:
        o = float(pd.read_csv(f"data/{rec}_recordingMeta.csv")
                  .at[0, "orthoPxToMeter"]) * FUDGE
        bg = mpimg.imread(f"data/{rec}_background.png")[::2, ::2]   # 2x downsample
        Hm, Wm = bg.shape[0] * 2 * o, bg.shape[1] * 2 * o
    except Exception as e:
        print(f"RESULT bg unavailable for {site}: {e}")
    vx0, vx1 = (max(xlo, 0.0), min(xhi, Wm)) if bg is not None else (xlo, xhi)
    vy0, vy1 = (max(ylo, -Hm), min(yhi, 0.0)) if bg is not None else (ylo, yhi)
    vmax = max(float(r.max()) for r in risk_seq)

    def rgba(fr):
        d = np.clip(fr / max(vmax, 1e-9), 0, 1) ** GAMMA_DISP
        img = cmap(d)
        img[..., 3] = np.clip(d * 1.5, 0, 0.85)
        return img

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    if bg is not None:
        ax.imshow(bg, extent=[0, Wm, -Hm, 0], origin="upper", zorder=0)
    else:
        ax.set_facecolor("black")
    im = ax.imshow(rgba(risk_seq[0].T), extent=[xlo, xhi, ylo, yhi],
                   origin="lower", zorder=1, animated=True)
    egodot, = ax.plot([], [], "o", color="cyan", ms=10, mec="white", zorder=5)
    egotrail, = ax.plot([], [], "-", color="cyan", lw=1.6, alpha=0.6, zorder=4)
    othdots, = ax.plot([], [], "s", color="#FF8C00", ms=7, mec="white", zorder=5)
    ttl = ax.set_title("", fontsize=11)
    ax.set_xlim(vx0, vx1); ax.set_ylim(vy0, vy1)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()

    def update(f):
        im.set_data(rgba(risk_seq[f].T))
        ex, ey = zip(*ego_seq[:f + 1])
        egotrail.set_data(ex, ey)
        egodot.set_data([ego_seq[f][0]], [ego_seq[f][1]])
        ag = agents_seq[f]
        othdots.set_data(ag[:, 0] if len(ag) else [], ag[:, 1] if len(ag) else [])
        ttl.set_text(f"Receding-horizon risk field  |  site {site}, ego {ego}  |  "
                     f"real time t = {f+1}/{len(risk_seq)}  ({(f+1)*DT:.1f}s)")
        return [im, egodot, egotrail, othdots, ttl]

    anim = FuncAnimation(fig, update, frames=len(risk_seq),
                         interval=1000 // FPS, blit=False)
    anim.save(out, writer=PillowWriter(fps=FPS))
    plt.close(fig)


with torch.no_grad():
    for ex_i, site in enumerate(SITES):
        try:
            ego = pick_long_ego(site)
            _, _, tmeta = ind._load_and_clean_data(site)
            # consecutive start frames for this ego (sampling-stepped)
            _, tracks, _ = ind._load_and_clean_data(site)
            edf = tracks[tracks["trackId"] == ego].sort_values("frame")
            edf = edf.iloc[:: ind.sampling_step]
            frames = edf["frame"].tolist()
            n_real = min(REAL_FRAMES, len(frames) - ind.moving_window)
            if n_real < 8:
                print(f"RESULT site {site}: ego too short, skipped")
                continue

            loc = int(pd.read_csv(f"data/{site}_recordingMeta.csv")
                      .at[0, "locationId"])
            bx = boundaries_for_location(loc)
            scale = torch.tensor([float(bx[0, 1] - bx[0, 0]),
                                  float(bx[1, 1] - bx[1, 0])], device=dev)
            lo = torch.tensor([float(bx[0, 0]), float(bx[1, 0])], device=dev)

            risk_seq, agents_seq, ego_seq = [], [], []
            for f in range(n_real):
                s = ind.get_specific_sample(site, ego, frames[f])
                x = s["input"].to(dev)
                feat = s["feature"].to(dev)
                vt = s["type"].to(dev)

                valid = [a for a in range(x.shape[1])
                         if not torch.isnan(x[0, a, -1]).any()]
                neigh = [a for a in valid if a != 0]
                ego_last = x[0, 0, -1] * scale + lo
                v_ego = ((x[0, 0, -1] - x[0, 0, -2]) * scale) / DT
                R = torch.zeros(G, device=dev)
                disc = (GAMMA ** torch.arange(K, device=dev, dtype=torch.float32))
                for j in neigh:
                    P = agent_density(x, feat, vt, j)               # (G,K)
                    cen = (P.t().unsqueeze(-1) * grid.unsqueeze(0)).sum(1) * scale
                    vj = torch.zeros(K, 2, device=dev)
                    vj[:-1] = (cen[1:] - cen[:-1]) / DT
                    vj[-1] = vj[-2]
                    Cj = 0.5 * M_R * (v_ego.unsqueeze(0) - vj).pow(2).sum(-1)
                    R = R + (P * (disc * Cj).unsqueeze(0)).sum(1)
                rf = gaussian_filter(R.reshape(S, S).cpu().numpy(), DILATE_SIGMA)
                risk_seq.append(rf)
                ego_seq.append((float(ego_last[0]), float(ego_last[1])))
                oth = np.array([( float((x[0, a, -1, 0] * scale[0] + lo[0])),
                                  float((x[0, a, -1, 1] * scale[1] + lo[1])))
                                for a in neigh], dtype=float).reshape(-1, 2)
                agents_seq.append(oth)

            out = f"riskfield_evolution_{ex_i + 1}.gif"
            render(site, ego, risk_seq, agents_seq, ego_seq, loc, out)
            print(f"RESULT saved {out}  site={site} ego={ego} loc={loc} "
                  f"real_frames={len(risk_seq)}")
        except Exception as e:
            import traceback
            print(f"RESULT site {site} FAILED: {e}")
            traceback.print_exc()
