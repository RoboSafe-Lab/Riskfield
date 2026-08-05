"""Ground-truth-only animation: play the RAW recorded tracks of the scene that a
cached test-loader index points at -- every agent at its recorded position, with
its recorded per-frame heading and recorded per-track dimensions. No model, no
risk overlay. Purpose: fidelity check (e.g. AD4CHE queue 'overlaps' in the model
animation come from class-default 12 m truck boxes; the GT has no overlap).

Env: RF_DATASET (ad4che), RF_SCENE_IDX (cached test index, e.g. 530),
     RF_OUT (gt_<idx>.gif), RF_FPS (12), RF_SPAN (raw frames, 300), RF_PAD (20).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation, PillowWriter

from datasets.registry import get_dataset
from riskflow_config import default_dict

reg = get_dataset(); c = default_dict()
SCENE_IDX = int(os.environ["RF_SCENE_IDX"])
OUT = os.environ.get("RF_OUT", f"gt_{SCENE_IDX}.gif")
FPS = int(os.environ.get("RF_FPS", "12"))
SPAN = int(os.environ.get("RF_SPAN", "300"))
PAD = float(os.environ.get("RF_PAD", "20"))
AR = 1.5

ind = reg["LoaderClass"](root=reg["root"], max_samples=c["maximum_samples"], train_ratio=c["train_ratio"],
    train_batch_size=c["train_batch_size"], test_batch_size=1, missing_rate=c["masked_data_ratio"],
    max_num_cars=c["max_num_cars"], max_empty_frames=c["max_empty_frames"], seq_len=c["seq_len"],
    moving_window=c["seq_len"] * 2, sampling_step=c["sampling_step"], should_shuffle=False,
    include_future=c["include_future"])
site = ind.observation_site_by_scope("all")

# locate the cached scene -> (recording site, ego id, start frame)
b = None
for i, bb in enumerate(site.test_loader):
    if i == SCENE_IDX:
        b = bb; break
assert b is not None, f"scene {SCENE_IDX} not found"
ego_id = int(np.asarray(b["trackId"]).reshape(-1)[0])
f0 = int(np.asarray(b["startFrame"]).reshape(-1)[0])
loc = int(b["locationId"].view(-1)[0])
ego_ref = np.nan_to_num(b["input"][0, 0].numpy())
match = None
for st in getattr(ind, "LOCATION_RECORDINGS", {}).get(loc, []):
    try:
        ss = ind.get_specific_sample(st, ego_id, f0)
    except Exception:
        continue
    if np.allclose(np.nan_to_num(ss["input"][0, 0].cpu().numpy()), ego_ref, atol=1e-3):
        match = st; break
assert match is not None, "no matching recording"
print(f"RESULT scene={SCENE_IDX} site={match} loc={loc} ego={ego_id} f0={f0}", flush=True)

# raw tracks of the matched recording (recorded pos/heading per frame, dims per track)
meta, tracks, tracks_meta = ind._load_and_clean_data(match)
fps_rec = float(meta.at[0, "frameRate"]) if "frameRate" in meta.columns else 30.0
lcol, wcol = ind._dim_columns()
dims = {int(t): (float(l), float(w)) for t, l, w in
        zip(tracks_meta["trackId"], tracks_meta[lcol], tracks_meta[wcol])}
cls = dict(zip(tracks_meta["trackId"], tracks_meta["class"]))
f1 = min(f0 + SPAN, int(tracks["frame"].max()))
win = tracks[(tracks["frame"] >= f0) & (tracks["frame"] <= f1)]
frames = sorted(win["frame"].unique())[:: max(1, c["sampling_step"])]
by_frame = {f: g for f, g in win.groupby("frame")}

# background (AD4CHE: per-scene map, centre-origin registration)
if reg["name"] == "ad4che":
    from datasets.AD4CHE import scene_scale
    bg = mpimg.imread(os.path.join(reg["root"], "maps", f"{loc}.jpg"))
    Hp, Wp = bg.shape[0], bg.shape[1]; sm = scene_scale(reg["root"], loc)
    bge = [-Wp / 2 * sm, Wp / 2 * sm, -Hp / 2 * sm, Hp / 2 * sm]
else:
    rec = match
    o = float(pd.read_csv(os.path.join(reg["root"], f"{rec}_recordingMeta.csv"))
              .at[0, "orthoPxToMeter"]) * float(reg.get("bg_scale_down", 12.0))
    bg = mpimg.imread(os.path.join(reg["root"], f"{rec}_background.png"))
    bge = [0, bg.shape[1] * o, -bg.shape[0] * o, 0]

# crop around the ego's window trajectory
ew = win[win["trackId"] == ego_id]
cx0, cx1 = ew["xCenter"].min() - PAD, ew["xCenter"].max() + PAD
cy0, cy1 = ew["yCenter"].min() - PAD, ew["yCenter"].max() + PAD
w, h = cx1 - cx0, cy1 - cy0
if w / h < AR: e = (AR * h - w) / 2; cx0, cx1 = cx0 - e, cx1 + e
else:         e = (w / AR - h) / 2; cy0, cy1 = cy0 - e, cy1 + e
cx0, cx1 = max(cx0, bge[0]), min(cx1, bge[1]); cy0, cy1 = max(cy0, bge[2]), min(cy1, bge[3])

EGO_C, OTH_C = (0.0, 0.85, 1.0), (1.0, 0.55, 0.0)


def box_xy(cx, cy, th, L, W):
    hl, hw = L / 2, W / 2; cs, sn = np.cos(th), np.sin(th)
    cor = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    rot = np.stack([cor[:, 0] * cs - cor[:, 1] * sn, cor[:, 0] * sn + cor[:, 1] * cs], 1)
    return rot + np.array([cx, cy])


fig, ax = plt.subplots(figsize=(7.6, 5.4))
ax.imshow(bg, extent=bge, origin="upper", zorder=0)
ax.set_xlim(cx0, cx1); ax.set_ylim(cy0, cy1); ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
ttl = ax.set_title("", fontsize=11)
patches = []


def update(fi):
    global patches
    for p in patches: p.remove()
    patches = []
    f = frames[fi]; g = by_frame.get(f)
    if g is None: return []
    for _, r in g.iterrows():
        tid = int(r["trackId"])
        x, y = float(r["xCenter"]), float(r["yCenter"])
        if not (cx0 - 15 < x < cx1 + 15 and cy0 - 15 < y < cy1 + 15): continue
        th = np.radians(float(r["heading"]))                 # recorded per-frame heading
        L, W = dims.get(tid, (4.5, 1.9))                     # recorded per-track footprint
        col = EGO_C if tid == ego_id else OTH_C
        p = Polygon(box_xy(x, y, th, L, W), closed=True, fill=True,
                    facecolor=(*col, 0.25), edgecolor=col, lw=1.8, zorder=5)
        ax.add_patch(p); patches.append(p)
    ttl.set_text(f"GROUND TRUTH playback -- {reg['name']} site {loc}, scene {SCENE_IDX}  "
                 f"(t = {(f - f0) / fps_rec:.2f} s; recorded pos + heading + true dims)")
    return patches


anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
anim.save(OUT, writer=PillowWriter(fps=FPS))
print(f"RESULT saved {OUT} ({len(frames)} frames, site {match}, agents drawn with recorded dims/heading)", flush=True)
