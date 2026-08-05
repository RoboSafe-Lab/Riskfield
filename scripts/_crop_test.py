"""Verify per-agent map-crop geometry on the real rounD raster."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from model.MapEncoder import build_location_maps, crop_agent_maps
from datasets.RounD import RounD, ROUND_BOUNDARIES

R = 192; S = 64; CROP_M = 40.0
rasters = build_location_maps("data_round", RounD.LOCATION_RECORDINGS, ROUND_BOUNDARIES, R)
rast = torch.from_numpy(rasters[0])[None]                       # (1,3,R,R)
(xlo, xhi), (ylo, yhi) = ROUND_BOUNDARIES[0]; boxm = np.array([xhi - xlo, yhi - ylo])
# probe positions: roundabout centre + 4 offsets, given as NORMALIZED [0,1]
probes_m = np.array([[78, -55], [40, -55], [115, -55], [78, -25], [78, -85]], float)  # metric (x,y)
pos_norm = (probes_m - np.array([xlo, ylo])) / boxm                                    # -> [0,1]
pos = torch.from_numpy(pos_norm)[None].float()                  # (1,P,2)
hw = torch.from_numpy((CROP_M / 2.0) / boxm)[None, None].float().expand(1, pos.shape[1], 2)
crops = crop_agent_maps(rast.expand(1, 3, R, R), pos, hw, S)    # (1,P,3,S,S)
P = pos.shape[1]
fig, axs = plt.subplots(1, P + 1, figsize=(3 * (P + 1), 3))
# full raster: array is [x,y] -> show transposed as image with y up
axs[0].imshow(np.transpose(rasters[0], (2, 1, 0)), origin="lower"); axs[0].set_title("raster (x->,y^)")
for j in range(P):
    axs[0].plot(pos_norm[j, 0] * R, pos_norm[j, 1] * R, "r+", ms=12)
    cimg = np.transpose(crops[0, j].numpy(), (2, 1, 0))         # (S,S,3) [x,y]->[y,x] for display
    axs[j + 1].imshow(cimg, origin="lower"); axs[j + 1].set_title(f"crop@({probes_m[j,0]:.0f},{probes_m[j,1]:.0f})m")
fig.savefig("crop_test.png", dpi=90, bbox_inches="tight")
print("RESULT raster", rasters[0].shape, "nonzero", float((rasters[0] != 0).mean()),
      "box", boxm.tolist(), "saved crop_test.png")
