"""For the two InD hero scenes, pull the risk-vs-time profile from the eval cache
and locate (a) the FAR10 warning level, (b) the frame the field's risk first
crosses it, (c) the PET onset frame, (d) the peak frame -- to design the
early-warning strip and confirm the field flags the conflict before PET onset."""
import numpy as np
DT = 0.08
Z = np.load("conflict_scores_ind_s20_g48.npz", allow_pickle=True)
sidx = Z["sidx"]
L = np.load("conflict_labels_pet.npz", allow_pickle=True)["data"]
lab = {int(r[0]): (int(r[2]), int(r[3])) for r in L}
y = np.array([lab[int(s)][0] if int(s) in lab else 0 for s in sidx])

for mk in ["ours_prob", "ours"]:
    P = Z["prof_" + mk]
    safe = P[y == 0]
    thr = float(np.percentile(safe.max(1), 90))
    print("\n=== method %s   FAR10 warning thr=%.4g ===" % (mk, thr))
    for sc in [44040, 56660]:
        w = np.where(sidx == sc)[0]
        if not len(w):
            print("  scene", sc, "not in cache"); continue
        prof = P[w[0]]
        K = len(prof)
        onset = lab[sc][1]
        kstar = int(prof.argmax())
        cross = np.where(prof > thr)[0]
        kwarn = int(cross[0]) if len(cross) else None
        lead = (onset - kwarn) * DT if kwarn is not None else None
        print("  scene %d: onset=%d (%.2fs) kstar=%d (%.2fs) k_warn=%s lead=%s" % (
            sc, onset, onset * DT, kstar, kstar * DT,
            kwarn, ("%.2fs" % lead if lead is not None else "never crosses")))
        # print profile normalized so warning=1.0
        norm = prof / thr if thr > 0 else prof
        print("     prof/thr per frame:", " ".join("%.1f" % x for x in norm))
