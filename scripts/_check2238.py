"""Direction check for AD4CHE scene 2238 partner j=2: observed vs GT-future vs
predicted velocity, to see if the head-on (174 deg) is real or a prediction flip."""
import os, sys; sys.path.insert(0, ".")
import numpy as np, torch
from datasets.registry import get_dataset
from riskflow_config import default_dict

DT = float(os.environ.get("RF_DT", "0.0667"))
reg = get_dataset(); c = default_dict()
ind = reg["LoaderClass"](root=reg["root"], max_samples=c["maximum_samples"], train_ratio=c["train_ratio"],
    train_batch_size=c["train_batch_size"], test_batch_size=1, missing_rate=c["masked_data_ratio"],
    max_num_cars=c["max_num_cars"], max_empty_frames=c["max_empty_frames"], seq_len=c["seq_len"],
    moving_window=c["seq_len"]*2, sampling_step=c["sampling_step"], should_shuffle=False, include_future=c["include_future"])
site = ind.observation_site_by_scope("all"); bf = reg["boundaries_for_location"]
TARGET = int(os.environ.get("RF_SCENE_IDS", "2238"))
for i, b in enumerate(site.test_loader):
    if i != TARGET:
        continue
    x = b["input"]; fut = b["future"]
    loc = int(b["locationId"].view(-1)[0]); bx = bf(loc)
    sca = np.array([float(bx[0, 1] - bx[0, 0]), float(bx[1, 1] - bx[1, 0])])
    lo = np.array([float(bx[0, 0]), float(bx[1, 0])])
    print(f"RESULT scene={TARGET} loc={loc}")
    for a in [0, 2, 6]:
        xa = x[0, a].cpu().numpy()
        vobs = (xa[-1] - xa[-2]) * sca / DT
        fa = fut[0, a].cpu().numpy()
        if np.isnan(fa).any():
            vgt = np.array([np.nan, np.nan]); nfut = int((~np.isnan(fa[:, 0])).sum())
        else:
            fw = fa * sca + lo; vgt = np.gradient(fw, DT, axis=0).mean(0); nfut = fa.shape[0]
        print(f"RESULT agent {a}: v_obs=({vobs[0]:+.2f},{vobs[1]:+.2f})|{np.hypot(*vobs):.2f}  "
              f"v_GTfuture=({vgt[0]:+.2f},{vgt[1]:+.2f})|{np.hypot(*vgt):.2f}  (nfut={nfut})")
    break
print("CHECK_DONE")
