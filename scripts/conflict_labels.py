"""Surrogate conflict labels on InD GT future trajectories (ego-centric).

For each test scene we label whether the EGO (agent 0) experiences a conflict with
any surrounding agent within the K-step future horizon, using two standard surrogate
safety measures on the GROUND-TRUTH metric trajectories:

  TTC : at some future frame the ego and agent j are on a closing course whose
        time-to-collision < tau_TTC and whose projected closest gap < d_safe.
  PET : the ego and agent j paths pass within d_safe of each other with a
        post-encroachment time |t_ego - t_j| < tau_PET (crossing conflicts).

A scene is a conflict iff (min-TTC < tau_TTC) OR (PET < tau_PET) for any neighbour.
We also record the conflict ONSET frame (earliest future frame the criterion fires)
for early-warning lead-time, and the partner agent. Output: conflict_labels.npz
(per-scene: scene_idx, loc, label, onset_k, min_ttc, min_pet, partner, n_neigh).

Env: RF_TAU_TTC (1.5 s), RF_TAU_PET (1.0 s), RF_DSAFE_SCALE (1.0), RF_MAX_SCENES (0=all).
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from datasets.registry import get_dataset
from riskflow_config import default_dict
_reg = get_dataset()                          # RF_DATASET env (ind|ad4che)
boundaries_for_location = _reg["boundaries_for_location"]

DT = float(os.environ.get("RF_DT", "0.08"))   # InD 0.08; AD4CHE 30fps*step2=0.0667
if os.environ.get("RF_DATASET", "ind").lower() == "ad4che":
    VEH_LW = {0: (4.5, 1.9), 1: (12.0, 2.6)}                  # car, truck/bus (highway)
else:
    VEH_LW = {0: (4.5, 1.9), 1: (10.0, 2.6), 2: (1.8, 0.6), 3: (0.7, 0.7)}  # L,W (m)
TAU_TTC = float(os.environ.get("RF_TAU_TTC", "1.5"))   # min instantaneous TTC for a closing conflict (s)
TAU_PET = float(os.environ.get("RF_TAU_PET", "1.0"))   # max post-encroachment time (s)
MARGIN = float(os.environ.get("RF_MARGIN", "0.5"))     # near-miss box-inflation clearance (m)
V_MIN = float(os.environ.get("RF_VMIN", "2.0"))        # min relative speed for a conflict (m/s)
MAX_SCENES = int(os.environ.get("RF_MAX_SCENES", "0"))
STRIDE = int(os.environ.get("RF_STRIDE", "1"))         # process every STRIDE-th scene
MODE = os.environ.get("RF_MODE", "motion")             # "motion" (OBB+vmin) | "dist" (TTC-independent)
D_HARD = float(os.environ.get("RF_DHARD", "3.0"))      # min centre-distance threshold (m) for "dist"
D_SAFE = float(os.environ.get("RF_DSAFE", "3.0"))      # motion-gated near-miss centre dist (m) for primary
OUT = os.environ.get("RF_OUT", "conflict_labels.npz")

def half_dims(t):  # (half_length, half_width) m
    L, W = VEH_LW.get(int(t), (4.5, 1.9))
    return 0.5 * L, 0.5 * W

def _headings(P):
    """Per-frame heading (rad) from finite-diff velocity; hold last where stationary."""
    v = np.gradient(P, DT, axis=0)
    sp = np.hypot(v[:, 0], v[:, 1]); th = np.arctan2(v[:, 1], v[:, 0])
    last = 0.0
    for k in range(len(th)):
        if sp[k] < 0.3: th[k] = last
        else: last = th[k]
    return th

def _extend(P, n_ext):
    """Constant-velocity extrapolation of trajectory P (K,2) by n_ext frames."""
    if n_ext <= 0: return P
    v = P[-1] - P[-2]
    tail = P[-1] + np.outer(np.arange(1, n_ext + 1), v)
    return np.concatenate([P, tail], 0)

def _obb_overlap(de, dn, the, hel, hew, thj, hjl, hjw):
    """SAT: do ego box (half hel,hew, heading the) at origin and agent box
    (hjl,hjw, heading thj) at offset (de,dn) intersect? scalars."""
    ue = (np.cos(the), np.sin(the)); ve = (-np.sin(the), np.cos(the))
    uj = (np.cos(thj), np.sin(thj)); vj = (-np.sin(thj), np.cos(thj))
    for n in (ue, ve, uj, vj):
        re = hel * abs(ue[0]*n[0]+ue[1]*n[1]) + hew * abs(ve[0]*n[0]+ve[1]*n[1])
        rj = hjl * abs(uj[0]*n[0]+uj[1]*n[1]) + hjw * abs(vj[0]*n[0]+vj[1]*n[1])
        if abs(de*n[0]+dn*n[1]) > re + rj:
            return False
    return True

def pair_conflict(p0, th0, hd0, pj, thj, hdj, margin, v_min):
    """HONEST near-miss from the ACTUAL recorded trajectories only -- NO constant-
    velocity extrapolation. Two assumption-free surrogates plus an instantaneous TTC:
      * PET (crossing): min |t_ego-t_j| over recorded oriented-box overlaps that are
        genuinely in relative motion (|v_ego[ki]-v_j[kj]|>v_min excludes static
        car-following). Cross-frame: ego at ki and j at kj occupy the same box.
      * min_gap_mot: closest same-frame centre distance the pair reaches WHILE in
        genuine relative motion (|dv|>v_min) -- a real spatial near-miss, not parking.
      * ttc (diagnostic only, NOT used for the label): smallest instantaneous
        time-to-closest-approach per frame (-|d|^2/(d.dv) when closing). This is a
        per-frame constant-velocity projection, so it is reported but never labels.
      * min_gap: closest same-frame centre distance (ungated -- for the distance label).
    Returns (pet, ttc, min_gap, min_gap_mot, onset_k)."""
    K = p0.shape[0]
    Pe, The = p0, th0
    Pj, Thj = pj, thj
    Ve = np.gradient(Pe, DT, axis=0); Vj = np.gradient(Pj, DT, axis=0)
    hel, hew = hd0[0] + margin, hd0[1] + margin
    hjl, hjw = hdj[0] + margin, hdj[1] + margin
    reach = hel + hjl                                  # max centre dist for overlap
    # same-frame instantaneous TTC (diagnostic) + min centre gap (ungated & motion-gated)
    relp = Pj - Pe; relv = Vj - Ve                     # (K,2)
    dvdot = (relp * relv).sum(-1)                       # d . dv  (<0 => closing)
    dsame = np.hypot(relp[:, 0], relp[:, 1])           # (K,)
    min_gap = float(dsame.min())
    moving = np.hypot(relv[:, 0], relv[:, 1]) > v_min   # genuine relative motion per frame
    if moving.any():
        mk = int(np.where(moving)[0][np.argmin(dsame[moving])])
        min_gap_mot = float(dsame[mk]); gap_mot_k = mk
    else:
        min_gap_mot = np.inf; gap_mot_k = K
    ttc = np.inf; ttc_k = K
    for k in range(K):
        if dvdot[k] < -1e-6:                           # closing at this frame
            t = -float(relp[k] @ relp[k]) / float(dvdot[k])
            if 0.0 < t < ttc:
                ttc = t; ttc_k = k
    # cross-frame PET from real oriented-box overlaps
    dm = np.linalg.norm(Pe[:, None, :] - Pj[None, :, :], axis=-1)   # (K,K)
    cand = np.argwhere(dm < reach)
    pet = np.inf; pet_on = K
    for ki, kj in cand:
        vrel = float(np.linalg.norm(Ve[ki] - Vj[kj]))
        if vrel <= v_min:                              # static proximity -> not a conflict
            continue
        d = Pj[kj] - Pe[ki]
        if _obb_overlap(d[0], d[1], The[ki], hel, hew, Thj[kj], hjl, hjw):
            t = abs(int(ki) - int(kj)) * DT
            if t < pet:
                pet = t; pet_on = int(min(min(ki, kj), K - 1))
    cands = ([pet_on] if np.isfinite(pet) else []) + ([gap_mot_k] if np.isfinite(min_gap_mot) else [])
    onset = min(cands) if cands else K
    return pet, ttc, min_gap, min_gap_mot, onset

def _extend_h(th, n_ext):
    if n_ext <= 0: return th
    return np.concatenate([th, np.full(n_ext, th[-1])], 0)

GT_GEOM = os.environ.get("RF_GT_GEOM", "0").lower() in ("1", "true")


def _attach_gt_geom(scenes, ind, Th, step, K):
    """Attach RECORDED geometry to each scene: match the cached scene back to its
    raw recording (ego trackId at the first future frame + position), then for
    every valid agent pull the recorded per-track footprint (length, width) and
    the recorded per-frame heading over the future window. Adds s['geom'] =
    {agent: (half_len, half_wid, th_K)}; unmatched agents keep the class-default
    + finite-difference fallback."""
    from collections import defaultdict
    lcol, wcol = ind._dim_columns()
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
            dims = {int(t): (float(l), float(w)) for t, l, w in
                    zip(tmeta["trackId"], tmeta[lcol], tmeta[wcol])}
            byf = {int(f): g for f, g in tracks.groupby("frame")}
            tby = {}
            still = []
            for s in pending:
                f0 = int(s["sf"] + Th * step)              # raw frame of future k=0
                g = byf.get(f0)
                ok = False
                if g is not None:
                    er = g[g["trackId"] == s["ego_id"]]
                    if len(er):
                        p = er.iloc[0]
                        if np.hypot(p["xCenter"] - s["Pm"][0, 0, 0],
                                    p["yCenter"] - s["Pm"][0, 0, 1]) < 0.6:
                            ok = True
                if not ok:
                    still.append(s)
                    continue
                px = g[["trackId", "xCenter", "yCenter"]].to_numpy()
                geom = {}; tids = {}
                for a in s["valid"]:
                    ca = s["Pm"][a, 0]
                    d = np.hypot(px[:, 1] - ca[0], px[:, 2] - ca[1])
                    j = int(d.argmin())
                    if d[j] > 0.8:
                        continue
                    tid = int(px[j, 0])
                    L, W = dims.get(tid, (np.nan, np.nan))
                    if not np.isfinite(L) or L < 0.5:
                        continue
                    if tid not in tby:
                        tby[tid] = tracks[tracks["trackId"] == tid].set_index("frame")
                    tr = tby[tid]
                    tha = np.full(K, np.nan)
                    for k in range(K):
                        fk = int(s["sf"] + (Th + k) * step)
                        if fk in tr.index:
                            tha[k] = np.radians(float(tr.at[fk, "heading"]))
                    fin = np.flatnonzero(~np.isnan(tha))
                    if len(fin) == 0:
                        continue
                    last = tha[fin[0]]
                    for k in range(K):                      # hold-last fill
                        if np.isnan(tha[k]):
                            tha[k] = last
                        else:
                            last = tha[k]
                    geom[a] = (0.5 * L, 0.5 * W, tha)
                    tids[a] = tid
                s["geom"] = geom
                s["tids"] = tids
                s["site"] = site_name
            pending = still
        matched = sum(1 for s in lst if "geom" in s)
        print(f"RESULT gt-geom loc {loc}: matched {matched}/{len(lst)} scenes", flush=True)


def main():
    c = default_dict()
    ind = _reg["LoaderClass"](root=_reg["root"], max_samples=c["maximum_samples"], train_ratio=c["train_ratio"],
              train_batch_size=c["train_batch_size"], test_batch_size=1, missing_rate=c["masked_data_ratio"],
              max_num_cars=c["max_num_cars"], max_empty_frames=c["max_empty_frames"], seq_len=c["seq_len"],
              moving_window=c["seq_len"]*2, sampling_step=c["sampling_step"], should_shuffle=False,
              include_future=c["include_future"])
    site = ind.observation_site_by_scope("all")
    Th = c["seq_len"]; sstep = int(c["sampling_step"])
    scenes = []
    n = 0
    for i, b in enumerate(site.test_loader):
        if STRIDE > 1 and (i % STRIDE) != 0:
            continue
        x = b["input"]; fut = b["future"]; types = b["type"]
        loc = int(b["locationId"].view(-1)[0])
        if torch.isnan(x[:, 0, -2:, :]).any() or torch.isnan(fut[0, 0]).any():
            continue
        bx = boundaries_for_location(loc)
        scale = np.array([float(bx[0,1]-bx[0,0]), float(bx[1,1]-bx[1,0])], np.float64)
        lo = np.array([float(bx[0,0]), float(bx[1,0])], np.float64)
        F = fut[0].numpy()                                  # (N,K,2) normalized
        valid = [a for a in range(F.shape[0])
                 if not np.isnan(F[a]).any()
                 and int((~torch.isnan(x[0,a,:,0])).sum()) >= 30]
        neigh = [a for a in valid if a != 0]
        if not neigh:
            continue
        scenes.append(dict(i=i, loc=loc, valid=valid, neigh=neigh,
                           Pm=F * scale + lo,
                           types=types[0].numpy().reshape(-1),
                           ego_id=int(np.asarray(b["trackId"]).reshape(-1)[0]),
                           sf=int(np.asarray(b["startFrame"]).reshape(-1)[0])))
        n += 1
        if MAX_SCENES and n >= MAX_SCENES:
            break
    if GT_GEOM and MODE != "dist" and scenes:
        _attach_gt_geom(scenes, ind, Th, sstep, scenes[0]["Pm"].shape[1])
    rows = []; sites_out = []
    for s in scenes:
        i, loc, valid, neigh, Pm, types = s["i"], s["loc"], s["valid"], s["neigh"], s["Pm"], s["types"]
        geom = s.get("geom", {})
        TH = np.stack([_headings(Pm[a]) for a in valid])    # fallback headings
        th = {a: TH[ix] for ix, a in enumerate(valid)}
        for a in geom:                                      # recorded heading overrides
            th[a] = geom[a][2]
        hd0 = geom[0][:2] if 0 in geom else half_dims(types[0])
        best_pet, best_ttc, best_gap, best_gap_mot, best_on, best_j = np.inf, np.inf, np.inf, np.inf, Pm.shape[1], -1
        if MODE == "dist":
            # TTC-independent: pure simultaneous min centre-distance < D_HARD
            for j in neigh:
                dseq = np.linalg.norm(Pm[0] - Pm[j], axis=-1)   # (K,)
                gp = float(dseq.min()); best_gap = min(best_gap, gp)
                if gp < D_HARD:
                    on = int(dseq.argmin())
                    if on < best_on: best_on, best_j = on, j
                    best_pet = 0.0
            label = int(best_gap < D_HARD)
            best_pet = best_gap                                 # store min-dist in pet slot
            on_out = best_on if label else Pm.shape[1]
        else:
            for j in neigh:
                hdj = geom[j][:2] if j in geom else half_dims(types[j])
                pe, tt, gp, gpm, on = pair_conflict(Pm[0], th[0], hd0, Pm[j], th[j], hdj, MARGIN, V_MIN)
                best_gap = min(best_gap, gp); best_ttc = min(best_ttc, tt)
                best_gap_mot = min(best_gap_mot, gpm)
                trig = (pe < TAU_PET) or (gpm < D_SAFE)          # crossing OR real near-miss
                if trig and on < best_on:
                    best_on, best_j = on, j
                best_pet = min(best_pet, pe)
            # HONEST conflict: a real crossing (PET) OR an actual spatial near-miss
            # in genuine relative motion (min-distance), straight from the recorded
            # trajectories -- NO constant-velocity extrapolation, TTC not used to label.
            label = int(best_pet < TAU_PET or best_gap_mot < D_SAFE)
            on_out = best_on if label else Pm.shape[1]
        tids = s.get("tids", {})
        rows.append((i, loc, label, on_out, best_pet, best_ttc, best_gap, best_gap_mot, best_j,
                     len(neigh), tids.get(0, -1), tids.get(best_j, -1)))
        sites_out.append(s.get("site", ""))
    arr = np.array([(r[0],r[1],r[2],r[3], min(r[4],1e6), min(r[5],1e6), min(r[6],1e6),
                     min(r[7],1e6), r[8], r[9], r[10], r[11]) for r in rows], np.float64)
    np.savez(OUT, data=arr,
             cols=np.array(["scene","loc","label","onset_k","pet","ttc","min_gap","min_gap_mot",
                            "partner","n_neigh","ego_tid","partner_tid"]),
             sites=np.array(sites_out),
             tau_pet=TAU_PET, tau_ttc=TAU_TTC, d_safe=D_SAFE, margin=MARGIN, n_ext=0, dt=DT,
             mode=MODE, d_hard=D_HARD, gt_geom=int(GT_GEOM))
    lab = arr[:,2]
    pet_trig = (arr[:,4] < TAU_PET).mean()*100 if MODE != "dist" else float("nan")
    gap_trig = (arr[:,7] < D_SAFE).mean()*100 if MODE != "dist" else float("nan")
    print(f"RESULT [{MODE}] scenes={len(arr)}  conflict_rate={lab.mean()*100:.2f}%  "
          f"(PET<{TAU_PET}s OR min-dist<{D_SAFE}m, motion-gated; margin={MARGIN}m vmin={V_MIN}; "
          f"dist-label dhard={D_HARD}m; NO-CV, TTC not used to label)  "
          f"[PET-trig={pet_trig:.1f}% dist-trig={gap_trig:.1f}%]  conflicts={int(lab.sum())} -> {OUT}")
    on = arr[lab>0,3]
    if len(on):
        print(f"RESULT onset_k median={np.median(on):.0f} ({np.median(on)*DT:.2f}s)  "
              f"pet med(conflict)={np.median(arr[lab>0,4]):.2f}s  "
              f"ttc med(conflict)={np.median(arr[lab>0,5]):.2f}s  "
              f"min_gap_mot med(conflict)={np.median(arr[lab>0,7]):.2f}m  "
              f"min_gap med(safe)={np.median(arr[lab<1,6]):.2f}m")
    print("saved conflict_labels.npz")

if __name__ == "__main__":
    main()
