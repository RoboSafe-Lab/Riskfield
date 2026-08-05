"""Detection robustness across the PET threshold (strict 0.5 / primary 1.0 / loose 1.5 s),
for each dataset and method. Reads sweep_{ind,ad4che,round}.json. Output: pet_sweep.png.
Shows the honest story compactly: TTC dominates InD at every threshold; ours-prob
dominates AD4CHE and rounD at every threshold."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DS = [("InD", "sweep_ind.json", "ind"),
      ("AD4CHE", "sweep_ad4che.json", "ad4che"),
      ("rounD", "sweep_round.json", "round")]
THR = [0.5, 1.0, 1.5]
TSUF = {0.5: "_petstrict", 1.0: "_pet", 1.5: "_petloose"}
# Okabe-Ito colourblind-safe palette; distinct markers keep it legible in grayscale too.
METHODS = [("ours_prob", "Ours (prob.)", "#0072B2", "o", 2.4),
           ("ours", "Ours (energy)", "#56B4E9", "s", 1.6),
           ("ttc", "TTC", "#D55E00", "^", 1.6),
           ("dsf", "DSF", "#000000", "v", 1.6),
           ("pora", "PORA-style", "#E69F00", "D", 1.6)]

def lblfile(suf, tau):
    base = "conflict_labels" + ("" if suf == "ind" else "_" + suf)
    return base + TSUF[tau] + ".npz"

plt.rcParams.update({"font.size": 12})
fig, axs = plt.subplots(2, 3, figsize=(13.0, 6.2), sharex=True)
for col, (name, f, suf) in enumerate(DS):
    d = json.load(open(f))["labels"]
    for ri, metric in enumerate(["AUROC", "AP"]):
        ax = axs[ri][col]
        for mkey, mlab, c, mk, lw in METHODS:
            ys = [d[lblfile(suf, t)][mkey][metric] for t in THR]
            ax.plot(THR, ys, marker=mk, color=c, label=mlab, lw=lw, ms=6,
                    zorder=(5 if mkey == "ours_prob" else 3))
        if metric == "AUROC":
            ax.axhline(0.5, ls=":", color="k", lw=1, alpha=0.5)
        if ri == 0:
            ax.set_title(f"{name}", fontsize=13)
        if col == 0:
            ax.set_ylabel(metric, fontsize=13)
        if ri == 1:
            ax.set_xlabel("PET threshold $\\tau_{\\mathrm{PET}}$ [s]", fontsize=12)
        ax.set_xticks(THR); ax.grid(alpha=0.25)
axs[0][2].legend(fontsize=9, loc="best", framealpha=0.9)
fig.tight_layout()
fig.savefig("pet_sweep.png", dpi=150, bbox_inches="tight")
print("saved pet_sweep.png")
