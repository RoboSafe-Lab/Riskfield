"""Distribution of the honest PET conflict surrogate (post-encroachment time, no
constant-velocity extrapolation) across InD/AD4CHE/rounD. Per-scene minimum over
neighbours (from conflict_labels_*_honest.npz, col 'pet'); conflict threshold
tau_PET=1.0s marked. Output: metric_dists.png"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FILES = [("InD", "conflict_labels_honest.npz", "#1f77b4"),
         ("AD4CHE", "conflict_labels_ad4che_honest.npz", "#d62728"),
         ("rounD", "conflict_labels_round_honest.npz", "#2ca02c")]
THR = 1.0       # tau_PET (s)
XMAX = 5.0      # plot range (finite values clipped)

plt.rcParams.update({"font.size": 12})
fig, ax = plt.subplots(figsize=(5.2, 3.6))
for name, f, color in FILES:
    try:
        z = np.load(f, allow_pickle=True)
    except FileNotFoundError:
        print(f"WARN missing {f}"); continue
    cols = list(z["cols"]); D = z["data"]
    v = D[:, cols.index("pet")].astype(float)
    finite = v[np.isfinite(v) & (v < 1e5)]
    frac_event = len(finite) / max(len(v), 1)
    ax.hist(np.clip(finite, 0, XMAX), bins=40, range=(0, XMAX), density=True,
            histtype="step", lw=2, color=color,
            label=f"{name} ({frac_event*100:.0f}% have PET)")
    print(f"RESULT {name} pet: n_event={len(finite)} median={np.median(finite):.2f}s "
          f"frac<{THR}s={(finite < THR).mean()*100:.1f}%")
ax.axvline(THR, ls="--", color="k", lw=1.5, alpha=0.8)
ax.text(THR, ax.get_ylim()[1] * 0.92, r"  $\tau_{\mathrm{PET}}{=}1.0$s", fontsize=10, va="top")
ax.set_title("post-encroachment time (PET)", fontsize=12)
ax.set_xlabel("PET [s]"); ax.set_ylabel("density")
ax.legend(fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig("metric_dists.png", dpi=150, bbox_inches="tight")
print("saved metric_dists.png")
