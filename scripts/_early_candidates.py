"""Find InD conflict scenes with strong EARLY WARNING: the risk (energy) field
crosses the FAR10 warning level well before the PET onset frame, and has a
visible peak. Candidates to replace Scene A in fig:qualitative."""
import numpy as np
DT = 0.08
Z = np.load("conflict_scores_ind_s20_g48.npz", allow_pickle=True)
sidx = Z["sidx"]
L = np.load("conflict_labels_pet.npz", allow_pickle=True)["data"]
lab = {int(r[0]): (int(r[2]), int(r[3])) for r in L}
y = np.array([lab[int(s)][0] if int(s) in lab else 0 for s in sidx])
Pe = Z["prof_ours"]; Pp = Z["prof_ours_prob"]
thr_e = float(np.percentile(Pe[y == 0].max(1), 90))
thr_p = float(np.percentile(Pp[y == 0].max(1), 90))
print("FAR10 thr: energy=%.1f prob=%.3f" % (thr_e, thr_p))

cand = []
for k in np.where(y == 1)[0]:
    sc = int(sidx[k]); onset = lab[sc][1]
    pe = Pe[k]; peak = float(pe.max()); kstar = int(pe.argmax())
    cr = np.where(pe > thr_e)[0]
    if not len(cr):
        continue
    kwarn = int(cr[0]); lead = (onset - kwarn) * DT
    # crossing must be before onset and peak must be visible
    if lead >= 0.24 and peak >= 120 and 8 <= onset <= 32:
        cand.append((sc, onset, kwarn, lead, peak, kstar))
cand.sort(key=lambda x: (-x[3], -x[4]))
print("scene  onset  k_warn  lead(s)  peakE(J)  kstar")
for sc, onset, kwarn, lead, peak, kstar in cand[:18]:
    print("%6d  %4d   %4d   %+.2f    %7.0f   %4d" % (sc, onset, kwarn, lead, peak, kstar))
