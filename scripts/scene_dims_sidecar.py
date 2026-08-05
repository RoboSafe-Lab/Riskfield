"""Build a sidecar of RECORDED per-agent dims for every cached test scene, by
matching each scene back to its raw recording (ego trackId at the first future
frame + position) and identifying every agent slot by position. Output npz:
sidx (M,), dims (M, max_cars, 2) [L,W] m with NaN where unmatched.

The cached batches carry only normalized positions (no neighbour identities or
dims), and the caches cannot be rebuilt (unseeded splits), so this sidecar is
the bridge that lets the FIELD's meeting-probability convolution use recorded
footprints instead of class defaults (joint_field.field(dims_m=...)).

Env: RF_DATASET, RF_OUT (scene_dims_<dataset>.npz), RF_STRIDE (1).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
import numpy as np
import torch
from datasets.registry import get_dataset
from riskflow_config import default_dict

reg = get_dataset(); c = default_dict()
STRIDE = int(os.environ.get("RF_STRIDE", "1"))
OUT = os.environ.get("RF_OUT", f"scene_dims_{reg['name']}.npz")

ind = reg["LoaderClass"](root=reg["root"], max_samples=c["maximum_samples"], train_ratio=c["train_ratio"],
    train_batch_size=c["train_batch_size"], test_batch_size=1, missing_rate=c["masked_data_ratio"],
    max_num_cars=c["max_num_cars"], max_empty_frames=c["max_empty_frames"], seq_len=c["seq_len"],
    moving_window=c["seq_len"] * 2, sampling_step=c["sampling_step"], should_shuffle=False,
    include_future=c["include_future"])
site = ind.observation_site_by_scope("all")
Th = c["seq_len"]; sstep = int(c["sampling_step"]); NC = c["max_num_cars"]
bf = reg["boundaries_for_location"]; lcol, wcol = ind._dim_columns()

scenes = []
for i, b in enumerate(site.test_loader):
    if STRIDE > 1 and (i % STRIDE) != 0:
        continue
    x = b["input"]; fut = b["future"]
    loc = int(b["locationId"].view(-1)[0])
    if torch.isnan(fut[0, 0, 0]).any():
        continue
    bx = bf(loc)
    sca = np.array([float(bx[0, 1] - bx[0, 0]), float(bx[1, 1] - bx[1, 0])])
    lon = np.array([float(bx[0, 0]), float(bx[1, 0])])
    F = fut[0].numpy()
    p0 = F[:, 0, :] * sca + lon                            # (N,2) at first future frame
    present = [a for a in range(F.shape[0]) if not np.isnan(F[a, 0]).any()]
    scenes.append(dict(i=i, loc=loc, p0=p0, present=present,
                       ego_id=int(np.asarray(b["trackId"]).reshape(-1)[0]),
                       sf=int(np.asarray(b["startFrame"]).reshape(-1)[0])))
print(f"RESULT collected {len(scenes)} test scenes", flush=True)

by_loc = defaultdict(list)
for s in scenes:
    by_loc[s["loc"]].append(s)
for loc, lst in by_loc.items():
    pending = list(lst)
    for site_name in getattr(ind, "LOCATION_RECORDINGS", {}).get(loc, []):
        if not pending:
            break
        try:
            _m, tracks, tmeta = ind._load_and_clean_data(site_name)
        except Exception:
            continue
        dl = {int(t): (float(l), float(w)) for t, l, w in
              zip(tmeta["trackId"], tmeta[lcol], tmeta[wcol])}
        byf = {int(f): g[["trackId", "xCenter", "yCenter"]].to_numpy()
               for f, g in tracks.groupby("frame")}
        still = []
        for s in pending:
            px = byf.get(int(s["sf"] + Th * sstep))
            ok = False
            if px is not None:
                er = px[px[:, 0] == s["ego_id"]]
                if len(er) and np.hypot(er[0, 1] - s["p0"][0, 0], er[0, 2] - s["p0"][0, 1]) < 0.6:
                    ok = True
            if not ok:
                still.append(s)
                continue
            dm = np.full((NC, 2), np.nan)
            for a in s["present"]:
                d = np.hypot(px[:, 1] - s["p0"][a, 0], px[:, 2] - s["p0"][a, 1])
                jx = int(d.argmin())
                if d[jx] > 0.8:
                    continue
                L, W = dl.get(int(px[jx, 0]), (np.nan, np.nan))
                dm[a] = (L, W)
            s["dims"] = dm
        pending = still
    print(f"RESULT loc {loc}: matched {sum(1 for s in lst if 'dims' in s)}/{len(lst)}", flush=True)

sidx = np.array([s["i"] for s in scenes if "dims" in s], np.int64)
dims = np.stack([s["dims"] for s in scenes if "dims" in s]) if len(sidx) else np.zeros((0, NC, 2))
np.savez(OUT, sidx=sidx, dims=dims)
print(f"RESULT saved {OUT}: {len(sidx)} scenes with recorded dims", flush=True)
