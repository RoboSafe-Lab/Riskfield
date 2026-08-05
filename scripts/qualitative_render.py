"""Render the 6 qualitative panels (critical + non-critical scene x {occupancy
density, optical-flow speed field, risk-field energy curve}) from qual_data.npz."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

SRC = sys.argv[1] if len(sys.argv) > 1 else "qual_data.npz"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT, exist_ok=True)
d = np.load(SRC, allow_pickle=True)
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "figure.dpi": 150})


EMAX = max(float(d["crit_energy_peak"].max()), float(d["ncrit_energy_peak"].max()))


def panels(tag):
    Ptot = d[f"{tag}_Ptot"]; speed = d[f"{tag}_speed"]
    ep = d[f"{tag}_energy_peak"]; ext = d[f"{tag}_extent"]
    k = int(d[f"{tag}_kstar"]); dt = float(d[f"{tag}_dt"])
    ego = d[f"{tag}_ego_xy"]; par = d[f"{tag}_par_xy"]; onset = int(d[f"{tag}_onset"])
    S, _, K = Ptot.shape
    occ = Ptot[:, :, k]
    occn = occ / max(occ.max(), 1e-9)
    alpha = np.clip(occn.T ** 0.5, 0, 1)        # density-weighted opacity for the speed field

    # 1) occupancy density
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    im = ax.imshow(occ.T, origin="lower", extent=ext, cmap="magma", aspect="auto",
                   norm=PowerNorm(gamma=0.5, vmin=0, vmax=occ.max()))
    ax.plot(ego[:, 0], ego[:, 1], "-", color="cyan", lw=1.3, alpha=0.9)
    ax.scatter(ego[k, 0], ego[k, 1], marker="*", s=130, c="cyan", edgecolors="k", zorder=5, label="ego")
    if par.shape[0] > k:
        ax.plot(par[:, 0], par[:, 1], "-", color="lime", lw=1.3, alpha=0.9)
        ax.scatter(par[k, 0], par[k, 1], marker="^", s=80, c="lime", edgecolors="k", zorder=5, label="partner")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="occupancy density")
    if par.shape[0] > k:
        ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    fig.tight_layout(); fig.savefig(f"{OUT}/qual_{tag}_pos.png", bbox_inches="tight"); plt.close(fig)

    # 2) optical-flow speed field (opacity = density, so motion shows where agents are)
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.set_facecolor("#111111")
    im = ax.imshow(speed[:, :, k].T, origin="lower", extent=ext, cmap="viridis",
                   aspect="auto", alpha=alpha, vmin=0)
    ax.scatter(ego[k, 0], ego[k, 1], marker="*", s=130, c="white", edgecolors="k", zorder=5)
    if par.shape[0] > k:
        ax.scatter(par[k, 0], par[k, 1], marker="^", s=80, c="white", edgecolors="k", zorder=5)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="speed [m/s]")
    fig.tight_layout(); fig.savefig(f"{OUT}/qual_{tag}_vel.png", bbox_inches="tight"); plt.close(fig)

    # 3) risk-field energy vs time
    t = np.arange(K) * dt
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.plot(t, ep, "-o", color="crimson", ms=3, lw=1.8)
    ax.fill_between(t, 0, ep, color="crimson", alpha=0.15)
    ax.set_ylim(0, EMAX * 1.08)
    if 0 <= onset < K:
        ax.axvline(onset * dt, ls="--", color="k", lw=1.2, alpha=0.7)
        ax.text(onset * dt, EMAX * 0.99, " conflict\n onset", fontsize=9, va="top")
    ax.text(0.96, 0.96, f"peak $={ep.max():.1f}$ J", transform=ax.transAxes,
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.85))
    ax.set_xlabel("time [s]"); ax.set_ylabel("peak risk energy [J]")
    ax.margins(x=0.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/qual_{tag}_energy.png", bbox_inches="tight"); plt.close(fig)
    print(f"{tag}: kstar={k} onset={onset} peakE={ep.max():.1f}J  -> 3 panels")


for tag in ("crit", "ncrit"):
    panels(tag)
print("done")
