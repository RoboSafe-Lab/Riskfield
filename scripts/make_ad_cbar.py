"""Standalone horizontal colorbar (inferno, 0..T_critical) for the AD4CHE risk grid,
so the four native panels can be placed sharp in LaTeX with one shared bar below."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm, matplotlib.colors

FIGDIR = sys.argv[1] if len(sys.argv) > 1 else "figures"
VMAX = 1238.0
fig, ax = plt.subplots(figsize=(6.0, 0.42))
smap = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, VMAX), cmap="inferno")
cb = fig.colorbar(smap, cax=ax, orientation="horizontal")
cb.set_label("expected collision energy [J]  (critical $=1238$ J)", fontsize=10)
cb.set_ticks([0, 0.5 * VMAX, VMAX]); cb.set_ticklabels(["0", "619", "1238"])
fig.savefig(f"{FIGDIR}/ad_cbar_horiz.png", dpi=220, bbox_inches="tight")
print("saved ad_cbar_horiz.png")
