"""Verify AD4CHE per-agent map-crop registration."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from model.MapEncoder import build_ad4che_rasters, crop_agent_maps
from datasets.AD4CHE import compute_scene_boundaries

ROOT = "data_ad4che"; R = 192; S = 64; CROP_M = 40.0; SC = 17
boxes = compute_scene_boundaries(ROOT)
rasters = build_ad4che_rasters(ROOT, [SC], boxes, R)
rast = torch.from_numpy(rasters[SC])[None]                      # (1,3,R,R)
(xlo, xhi), (ylo, yhi) = boxes[SC]; boxm = np.array([xhi - xlo, yhi - ylo])
probes_m = np.array([[-50, 0], [0, 5], [50, -5], [0, -10]], float)   # metric (x,y) on the road
pos_norm = (probes_m - np.array([xlo, ylo])) / boxm
pos = torch.from_numpy(pos_norm)[None].float()
hw = torch.from_numpy((CROP_M / 2.0) / boxm)[None, None].float().expand(1, pos.shape[1], 2)
crops = crop_agent_maps(rast.expand(1, 3, R, R), pos, hw, S)
P = pos.shape[1]
fig, axs = plt.subplots(1, P + 1, figsize=(3 * (P + 1), 3))
axs[0].imshow(np.transpose(rasters[SC], (2, 1, 0)), origin="lower"); axs[0].set_title(f"AD4CHE sc{SC} raster")
for j in range(P):
    axs[0].plot(pos_norm[j, 0] * R, pos_norm[j, 1] * R, "r+", ms=12)
    axs[j + 1].imshow(np.transpose(crops[0, j].numpy(), (2, 1, 0)), origin="lower")
    axs[j + 1].set_title(f"crop@({probes_m[j,0]:.0f},{probes_m[j,1]:.0f})")
fig.savefig("crop_test_ad.png", dpi=90, bbox_inches="tight")
print("RESULT ad raster", rasters[SC].shape, "nonzero", float((rasters[SC] != 0).mean()),
      "box", boxm.tolist(), "saved crop_test_ad.png")
