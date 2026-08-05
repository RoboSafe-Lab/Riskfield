"""Compose the 4 colorbar-less AD4CHE risk panels into a 2x2 grid with ONE shared
colorbar (CRITICAL = T_critical). Output: figures/qual_ad4che_grid.png."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm, matplotlib.colors
from matplotlib.gridspec import GridSpec

FIGDIR = sys.argv[1] if len(sys.argv) > 1 else "figures"
VMAX = 1238.0
panels = [("qual_map_c530_risk.png", "(a) Conflict, site 17"),
          ("qual_map_c190_risk.png", "(b) Conflict, site 15"),
          ("qual_map_n850_risk.png", "(c) No conflict, site 17"),
          ("qual_map_n230_risk.png", "(d) No conflict, site 15")]
plt.rcParams.update({"font.size": 12})
imgs = [mpimg.imread(f"{FIGDIR}/{f}") for f, _ in panels]
hmin = min(im.shape[0] for im in imgs); wmin = min(im.shape[1] for im in imgs)


def center_crop(im):                                   # -> identical (hmin x wmin), no distortion
    h, w = im.shape[:2]; t, l = (h - hmin) // 2, (w - wmin) // 2
    return im[t:t + hmin, l:l + wmin]


imgs = [center_crop(im) for im in imgs]
fig = plt.figure(figsize=(9.2, 4.9))
gs = GridSpec(2, 3, width_ratios=[1, 1, 0.045], wspace=0.03, hspace=0.16, figure=fig)
cells = [gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1]]
for cell, im, (f, t) in zip(cells, imgs, panels):
    ax = fig.add_subplot(cell)
    ax.imshow(im)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, fontsize=11)
cax = fig.add_subplot(gs[:, 2])
smap = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, VMAX), cmap="inferno")
cb = fig.colorbar(smap, cax=cax)
cb.set_label("expected collision energy [J]", fontsize=10)
cb.set_ticks([0, 0.5 * VMAX, VMAX]); cb.set_ticklabels(["0", f"{0.5 * VMAX:.0f}", f"CRITICAL\n{VMAX:.0f} J"])
fig.savefig(f"{FIGDIR}/qual_ad4che_grid.png", dpi=200, bbox_inches="tight")
print("saved qual_ad4che_grid.png")
