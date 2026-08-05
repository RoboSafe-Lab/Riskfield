"""Map conditioning from InD drone orthophotos (no vector maps available).

We resample each location's background image into that location's normalized
coordinate box (the same frame agent positions live in), giving a per-location
BEV raster of the road layout. A small CNN encodes it into an embedding that is
added to the agent embeddings, so the flow's predictions are conditioned on the
road geometry (lanes/drivable area) -> sharper, road-following densities.

The map is static per location (the road doesn't move), so we precompute one
raster per InD location (1..4) and look it up by locationId at run time.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def crop_agent_maps(rasters, pos_xy, half_wh, out_s, heading=None):
    """Per-agent LOCAL map crops via differentiable grid_sample.

    rasters : (B, 3, R, R) per-sample location raster, indexed ``[., x, y]`` (x along
              dim2, y along dim3, both increasing in normalized [0,1]) -- the layout
              produced by ``build_location_maps`` (out[i,j] = world (x=u[i], y=u[j])).
    pos_xy  : (B, N, 2) agent centre in the location's normalized [0,1] frame.
    half_wh : (B, N, 2) normalized half-width of the crop per axis (metric-square crop
              -> (crop_m/2)/box_m per axis).
    out_s   : output crop resolution.
    heading : optional (B, N) heading [rad] to rotate the crop so the agent's heading
              points along +x of the crop (agent-centric). None -> axis-aligned.
    Returns (B, N, 3, out_s, out_s).
    """
    B, N = pos_xy.shape[0], pos_xy.shape[1]
    R = rasters.shape[-1]; dev = rasters.device
    u = torch.linspace(-1.0, 1.0, out_s, device=dev, dtype=rasters.dtype)
    gx, gy = torch.meshgrid(u, u, indexing="ij")          # local axes: gx//x, gy//y
    lx = gx.reshape(1, 1, out_s, out_s)                   # (1,1,S,S)
    ly = gy.reshape(1, 1, out_s, out_s)
    hx = half_wh[..., 0, None, None]; hy = half_wh[..., 1, None, None]   # (B,N,1,1)
    if heading is not None:                               # rotate local axes by heading
        c = torch.cos(heading)[..., None, None]; s = torch.sin(heading)[..., None, None]
        rx = c * lx - s * ly; ry = s * lx + c * ly
        offx, offy = hx * rx, hy * ry
    else:
        offx, offy = hx * lx, hy * ly
    cx = pos_xy[..., 0, None, None]; cy = pos_xy[..., 1, None, None]
    sx = (cx + offx).clamp(0.0, 1.0)                      # normalized sample coord along x
    sy = (cy + offy).clamp(0.0, 1.0)                      # along y
    # grid_sample input layout (C, dim2=x, dim3=y): grid[...,0]->dim3(y), grid[...,1]->dim2(x)
    grid = torch.stack([2.0 * sy - 1.0, 2.0 * sx - 1.0], dim=-1).reshape(B * N, out_s, out_s, 2)
    rin = rasters[:, None].expand(B, N, 3, R, R).reshape(B * N, 3, R, R)
    crops = F.grid_sample(rin, grid, mode="bilinear", align_corners=True, padding_mode="border")
    return crops.reshape(B, N, 3, out_s, out_s)

SCALE_DOWN = 12.0   # InD scale_down_factor (drone-dataset-tools convention)


def build_location_maps(data_dir, location_recordings, boundaries, smap=64, scale_down=None):
    """Return {loc: (3, smap, smap) float32 in [0,1]} BEV rasters of the
    orthophoto resampled into each location's normalized box. ``scale_down`` is
    the dataset's background-PNG downscale factor (drone-dataset-tools
    visualizer_params: InD 12, rounD 10); defaults to the InD value."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    from scipy.ndimage import map_coordinates

    sd = float(scale_down) if scale_down is not None else SCALE_DOWN
    maps = {}
    for loc, recs in location_recordings.items():
        rec = recs[0]
        try:
            o = float(pd.read_csv(os.path.join(data_dir, f"{rec}_recordingMeta.csv"))
                      .at[0, "orthoPxToMeter"]) * sd
            img = mpimg.imread(os.path.join(data_dir, f"{rec}_background.png"))
            img = np.asarray(img, dtype=np.float32)[..., :3]
            if img.max() > 1.5:
                img = img / 255.0
        except Exception as e:  # noqa: BLE001
            print(f"[MapEncoder] location {loc}: no background ({e}); zero map")
            maps[loc] = np.zeros((3, smap, smap), np.float32)
            continue
        (xlo, xhi), (ylo, yhi) = boundaries[loc]
        u = (np.arange(smap) + 0.5) / smap
        UU, VV = np.meshgrid(u, u, indexing="ij")        # axis0=x(u), axis1=y(v)
        X = xlo + UU * (xhi - xlo)
        Y = ylo + VV * (yhi - ylo)
        col = X / o                                      # image x-pixel
        row = -Y / o                                     # image y-pixel (y flipped)
        out = np.zeros((3, smap, smap), np.float32)
        for c in range(3):
            out[c] = map_coordinates(
                img[..., c], [row.ravel(), col.ravel()], order=1,
                mode="constant", cval=0.0).reshape(smap, smap)
        maps[loc] = out
    return maps


def build_ad4che_maps(data_dir, scenes, smap=64):
    """Return {scene: (3, smap, smap) float32 in [0,1]} per-scene road rasters by
    resampling ``maps/<scene>.jpg`` to smap x smap. The map conditions a per-LOCATION
    embedding (whole raster -> one vector), so a consistent per-scene image suffices;
    exact metric registration is not required (and AD4CHE provides no map scale)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    from scipy.ndimage import map_coordinates

    maps = {}
    for sc in scenes:
        p = os.path.join(data_dir, "maps", f"{int(sc)}.jpg")
        try:
            img = np.asarray(mpimg.imread(p), dtype=np.float32)[..., :3]
            if img.max() > 1.5:
                img = img / 255.0
            H, W = img.shape[:2]
            u = (np.arange(smap) + 0.5) / smap
            rr, cc = np.meshgrid(u * (H - 1), u * (W - 1), indexing="ij")
            out = np.stack([
                map_coordinates(img[..., c], [rr.ravel(), cc.ravel()], order=1).reshape(smap, smap)
                for c in range(3)])
            maps[int(sc)] = out.astype(np.float32)
        except Exception as e:  # noqa: BLE001
            print(f"[MapEncoder] AD4CHE scene {sc}: no map ({e}); zero map")
            maps[int(sc)] = np.zeros((3, smap, smap), np.float32)
    return maps


def build_ad4che_rasters(data_dir, scenes, boundaries, smap=192):
    """AD4CHE per-scene rasters resampled into each scene's NORMALIZATION box (so
    per-agent crops via ``crop_agent_maps`` align with agent positions). The jpg is
    centre-origin registered: world (x,y) -> col = x/sm + W/2, row = H/2 - y/sm.
    Returns {scene: (3, smap, smap)} indexed ``[., x, y]`` (matches build_location_maps)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    from scipy.ndimage import map_coordinates
    from datasets.AD4CHE import scene_scale

    maps = {}
    for sc in scenes:
        sc = int(sc)
        try:
            img = np.asarray(mpimg.imread(os.path.join(data_dir, "maps", f"{sc}.jpg")),
                             dtype=np.float32)[..., :3]
            if img.max() > 1.5:
                img = img / 255.0
            H, W = img.shape[:2]; sm = scene_scale(data_dir, sc)
            (xlo, xhi), (ylo, yhi) = boundaries[sc]
            u = (np.arange(smap) + 0.5) / smap
            UU, VV = np.meshgrid(u, u, indexing="ij")          # axis0=x, axis1=y
            X = xlo + UU * (xhi - xlo); Y = ylo + VV * (yhi - ylo)
            col = X / sm + W / 2.0                              # centre-origin jpg pixel
            row = H / 2.0 - Y / sm
            out = np.stack([
                map_coordinates(img[..., c], [row.ravel(), col.ravel()], order=1,
                                mode="constant", cval=0.0).reshape(smap, smap)
                for c in range(3)])
            maps[sc] = out.astype(np.float32)
        except Exception as e:  # noqa: BLE001
            print(f"[MapEncoder] AD4CHE scene {sc}: no raster ({e}); zero map")
            maps[sc] = np.zeros((3, smap, smap), np.float32)
    return maps


class MapEncoder(nn.Module):
    """Small CNN: per-location BEV raster (3, S, S) -> embedding (emb_dim)."""

    def __init__(self, in_ch=3, emb_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, 2, 1), nn.ReLU(inplace=True),   # S/2
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(inplace=True),      # S/4
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True),      # S/8
            nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(inplace=True),      # S/16
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),                  # (B,64)
        )
        self.fc = nn.Linear(64, emb_dim)

    def forward(self, m):                                # m: (B, 3, S, S)
        return self.fc(self.net(m))
