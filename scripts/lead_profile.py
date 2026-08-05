"""Early-warning timing: cumulative fraction of conflict scenes whose risk has
crossed the FAR10 alarm threshold, as a function of time relative to the labelled
conflict onset (t=0). Ours (prob.) fires BEFORE onset (curve mass left of 0);
PORA-style only clears its bar at/after onset (mass at/right of 0), so its lead
is non-positive. Curves saturate at detect@FAR10 (non-firing conflicts never cross).
Reads the eval score caches. Output: lead_profile.png."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CFG = [("InD",    "conflict_scores_ind_s20_g48.npz",   "conflict_labels_pet.npz",       0.08),
       ("AD4CHE", "conflict_scores_ad4che_s5_g48.npz", "conflict_labels_ad4che_pet.npz", 0.0667),
       ("rounD",  "conflict_scores_round_s2_g48.npz",  "conflict_labels_round_pet.npz",  0.08)]
METH = [("ours_prob", "Ours (prob.)", "#0072B2", "-"),
        ("pora",      "PORA-style",   "#E69F00", "--")]
XS = np.linspace(-1.6, 1.0, 240)

plt.rcParams.update({"font.size": 12})
fig, axs = plt.subplots(1, 3, figsize=(13.0, 3.7), sharey=True)
for col, (name, cf, lf, dt) in enumerate(CFG):
    Z = np.load(cf, allow_pickle=True); sidx = Z["sidx"]
    L = np.load(lf, allow_pickle=True); D = L["data"]
    lab = {int(r[0]): (int(r[2]), int(r[3])) for r in D}
    keep = np.array([k for k, s in enumerate(sidx) if int(s) in lab])
    y = np.array([lab[int(sidx[k])][0] for k in keep])
    onk = np.array([lab[int(sidx[k])][1] for k in keep])
    ax = axs[col]
    for mk, ml, c, ls in METH:
        P = Z["prof_" + mk][keep]; K = P.shape[1]
        safe = P[y == 0]; thr = float(np.percentile(safe.max(1), 90)) if len(safe) else 0.0
        ci = np.where(y == 1)[0]; nconf = len(ci)
        rel = []
        for k in ci:
            w = np.where(P[k] > thr)[0]
            rel.append((int(w[0]) - int(onk[k])) * dt if len(w) else np.inf)
        rel = np.array(rel)
        cdf = np.array([np.mean(rel <= x) for x in XS])
        ax.plot(XS, 100 * cdf, ls, color=c, lw=2.4, label=ml)
    ax.axvline(0.0, ls=":", color="r", lw=1.3, alpha=0.8)
    ax.text(0.02, 0.96, "conflict\nonset", transform=ax.transAxes, color="r",
            fontsize=8, va="top", ha="left")
    ax.set_title(name, fontsize=13)
    ax.set_xlabel("time relative to conflict onset [s]", fontsize=11)
    if col == 0:
        ax.set_ylabel("% conflicts alarmed\n(at FAR10)", fontsize=11)
    ax.grid(alpha=0.25); ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
fig.tight_layout()
fig.savefig("lead_profile.png", dpi=150, bbox_inches="tight")
print("saved lead_profile.png")
