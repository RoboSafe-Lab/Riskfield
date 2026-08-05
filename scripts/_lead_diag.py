"""Decompose the early-warning lead time for ours_prob vs pora, per dataset.
Replicates eval_conflict's FAR10 lead exactly, but also prints the FAR10
threshold, the mean onset frame, and the mean first-alarm frame, so the
sign of the lead (PORA fires at/after onset) is explained by where its bar sits."""
import numpy as np

CFG = [
    ("InD",    "conflict_scores_ind_s20_g48.npz",   "conflict_labels_pet.npz"),
    ("AD4CHE", "conflict_scores_ad4che_s5_g48.npz", "conflict_labels_ad4che_pet.npz"),
    ("rounD",  "conflict_scores_round_s2_g48.npz",  "conflict_labels_round_pet.npz"),
]
METHODS = ["ours_prob", "pora"]

for name, cf, lf in CFG:
    Z = np.load(cf, allow_pickle=True)
    sidx = Z["sidx"]
    L = np.load(lf, allow_pickle=True)
    D = L["data"]
    dt = float(L["dt"]) if "dt" in L.files else 0.08
    lab = {int(r[0]): (int(r[2]), int(r[3])) for r in D}
    keep = np.array([k for k, s in enumerate(sidx) if int(s) in lab])
    y = np.array([lab[int(sidx[k])][0] for k in keep])
    onk = np.array([lab[int(sidx[k])][1] for k in keep])
    print("\n===== %s  (n=%d, conflicts=%d, dt=%.4f) =====" % (name, len(y), int(y.sum()), dt))
    for m in METHODS:
        P = Z["prof_" + m][keep]                       # (n, K) risk-vs-time
        K = P.shape[1]
        safe = P[y == 0]
        thr = float(np.percentile(safe.max(1), 90)) if len(safe) else 0.0
        ci = np.where(y == 1)[0]
        first = []; fired_on = []
        for k in ci:
            w = np.where(P[k] > thr)[0]
            if len(w):
                first.append(int(w[0])); fired_on.append(int(onk[k]))
        first = np.array(first); fired_on = np.array(fired_on)
        nfire = len(first)
        lead = float(np.mean((fired_on - first) * dt)) if nfire else float("nan")
        # background separation: median safe peak vs median conflict peak (scale check)
        safe_pk = np.median(safe.max(1)) if len(safe) else float("nan")
        conf_pk = np.median(P[y == 1].max(1))
        print("  %-9s FAR10-thr=%.4g | safe-peak med=%.4g  conflict-peak med=%.4g (signal/bar=%.2f)"
              % (m, thr, safe_pk, conf_pk, (conf_pk / thr if thr else float('nan'))))
        if nfire:
            print("             detect@FAR10=%d/%d=%.0f%% | mean onset=%.1f fr  mean first-alarm=%.1f fr  -> lead=%+.2f s"
                  % (nfire, len(ci), 100 * nfire / len(ci), fired_on.mean(), first.mean(), lead))
