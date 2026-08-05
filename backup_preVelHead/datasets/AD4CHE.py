"""AD4CHE loader (congested-highway drone dataset), subclassing the InD loader.

AD4CHE recordings live under ``<root>/<scene 14..24>/<DJI_xxxx>/`` with
``NN_tracks_pixel.csv`` / ``NN_tracksMeta.csv`` / ``NN_recordingMeta.csv`` files
whose schema parallels InD but with different column names. Positions
(``x,y``) and velocities are already metric; ``orientation`` is the heading in
radians. ``recordingMeta.locationId`` is always -1, so the **scene directory
number (14..24) is the physical location**: recordings within a scene share one
registered coordinate frame and the per-scene map ``<root>/maps/<scene>.jpg``.
We therefore normalize **per scene** (14..24 = locations, like InD's 4
intersections).
"""

import os
import re
import glob
import numpy as np
import pandas as pd

AD4CHE_ROOT_DEFAULT = "datasets/AD4CHE"            # local; cluster: data_ad4che
SCENES = [str(s) for s in range(14, 25)]           # 14..24 = "locations"
# AD4CHE classes -> model class ids (reuse InD's 2-class FiLM head: car=0, big=1)
AD4CHE_CLASS_TO_ID = {"car": 0, "truck": 1, "bus": 1}
# heading feature is degrees in [0,360]; velocities/accels widened for highway.
AD4CHE_FEATURE_BOUNDARIES = np.array([[0, 360], [-40, 40], [-40, 40], [-6, 6], [-6, 6]])


def _recording_dirs(root, scene):
    return sorted(glob.glob(os.path.join(root, scene, "DJI_*")))


def scene_scale(root, scene):
    """Metres-per-pixel from a scene's recordingMeta 'scale' string
    (e.g. '1 pixel = 0.0375 m'). The map registration is centre-origin:
    x_pixel = x/scale + W/2,  y_pixel = -y/scale + H/2."""
    rd = _recording_dirs(root, str(int(scene)))[0]
    rm = glob.glob(os.path.join(rd, "*_recordingMeta.csv"))[0]
    s = str(pd.read_csv(rm).at[0, "scale"])
    return float(re.search(r"([0-9.]+)\s*m", s).group(1))


def compute_scene_boundaries(root=AD4CHE_ROOT_DEFAULT, pad=3.0):
    """Per-scene spatial box ``[[xlo,xhi],[ylo,yhi]]`` pooled over the scene's
    recordings (positions are already metric in tracks_pixel.csv)."""
    out = {}
    for sc in SCENES:
        xs_lo = xs_hi = ys_lo = ys_hi = None
        for rd in _recording_dirs(root, sc):
            f = glob.glob(os.path.join(rd, "*_tracks_pixel.csv"))
            if not f:
                continue
            df = pd.read_csv(f[0], usecols=["x", "y"])
            xlo, xhi = float(df["x"].min()), float(df["x"].max())
            ylo, yhi = float(df["y"].min()), float(df["y"].max())
            xs_lo = xlo if xs_lo is None else min(xs_lo, xlo)
            xs_hi = xhi if xs_hi is None else max(xs_hi, xhi)
            ys_lo = ylo if ys_lo is None else min(ys_lo, ylo)
            ys_hi = yhi if ys_hi is None else max(ys_hi, yhi)
        if xs_lo is None:
            continue
        out[int(sc)] = np.array([[xs_lo - pad, xs_hi + pad], [ys_lo - pad, ys_hi + pad]])
    return out


import hashlib
from datasets.InD import InD, normalize


class AD4CHE(InD):
    """AD4CHE loader: subclasses InD, adapts the tracks_pixel/tracksMeta schema to
    InD's, and normalizes per scene (14..24). AD4CHE vehicle classes are renamed to
    InD's vocabulary (truck/bus -> ``truck_bus``) so InD's 2-class head is reused."""

    def __init__(self, root="data_ad4che", **kw):
        super().__init__(root=root, **kw)
        self.target_classes = ["car", "truck_bus"]          # after class rename
        self.input_cols = ["xCenter", "yCenter"]            # after column rename
        self.feature_cols = ["heading", "xVelocity", "yVelocity",
                             "xAcceleration", "yAcceleration"]
        self._boxes = compute_scene_boundaries(root)
        # AD4CHE congested highway: hundreds of egos/recording -> cap + stride windows
        # so parsing is tractable (InD leaves these unset).
        self.max_egos_per_rec = int(os.environ.get("RF_AD4CHE_EGOS", "40"))
        self.window_stride = int(os.environ.get("RF_AD4CHE_WSTRIDE", "25"))
        # scene -> slash-free site ids "<scene>_<DJI_xxxx>" (slashes break cache paths)
        self.LOCATION_RECORDINGS = {
            int(sc): [f"{sc}_{os.path.basename(rd)}" for rd in _recording_dirs(root, sc)]
            for sc in SCENES if _recording_dirs(root, sc)
        }

    def boundaries_for_location(self, location_id):
        return self._boxes[int(location_id)]

    def _dim_columns(self):
        # AD4CHE tracksMeta: 'width' is the longitudinal length, 'height' the
        # lateral width (cars ~4.3x1.7, trucks ~7.7x2.3, buses ~10.7x2.4 m).
        return ("width", "height")

    def _cache_prefix(self, observation_sites):
        h = hashlib.md5("-".join(observation_sites).encode()).hexdigest()[:8]
        return (f"ad4che_{len(observation_sites)}sites_{h}"
                f"_maxcars{self.max_num_cars}_window{self.moving_window}"
                f"_ratio{self.train_ratio}_miss{self.missing_rate}"
                f"_step{self.sampling_step}_maxsamp{self.max_samples}"
                f"_fut{int(self.include_future)}")

    def _load_and_clean_data(self, site):
        """site = '<scene>_<DJI_xxxx>'. Returns (meta, tracks, tracks_meta) in InD schema."""
        scene, rec = site.split("_", 1)                     # '14', 'DJI_0009'
        rd = os.path.join(self.root, scene, rec)
        prefix = glob.glob(os.path.join(rd, "*_tracks_pixel.csv"))[0].rsplit(
            "_tracks_pixel.csv", 1)[0]
        meta = pd.read_csv(prefix + "_recordingMeta.csv")
        tracks = pd.read_csv(prefix + "_tracks_pixel.csv")
        tmeta = pd.read_csv(prefix + "_tracksMeta.csv")
        # --- rename columns to InD schema ---
        tracks = tracks.rename(columns={"id": "trackId", "x": "xCenter", "y": "yCenter"})
        tracks["heading"] = np.degrees(tracks["orientation"]) % 360.0
        tracks["trackLifetime"] = tracks.groupby("trackId").cumcount()
        tmeta = tmeta.rename(columns={"id": "trackId"})
        # --- rename classes to InD vocabulary (reuse InD's 2-class head) ---
        cmap = {"car": "car", "truck": "truck_bus", "bus": "truck_bus"}
        tmeta["class"] = tmeta["class"].map(lambda c: cmap.get(str(c).strip(), "other"))
        # InD `_parse` reads locationId from meta; inject the SCENE number.
        meta["locationId"] = int(scene)
        meta["orthoPxToMeter"] = 1.0          # positions already metric; kept for cache
        tracks = tracks.drop_duplicates(subset=["trackId", "frame"])
        return meta, tracks, tmeta

    def _collate_results(self, samples, meta, spatial_box):
        if not samples:
            return {}
        result = {}
        for k in samples[0].keys():
            arrs = [s[k] for s in samples]
            arr = np.stack(arrs) if isinstance(arrs[0], np.ndarray) else np.array(arrs)
            if k in self.spatial_keys:
                arr = normalize(arr, spatial_box)
            elif k in self.feature_keys:
                arr = normalize(arr, AD4CHE_FEATURE_BOUNDARIES)
            result[k] = arr
        result["orthoPxToMeter"] = 1.0
        return result


if __name__ == "__main__":   # boundaries smoke
    b = compute_scene_boundaries(os.environ.get("RF_AD4CHE_ROOT", AD4CHE_ROOT_DEFAULT))
    for k in sorted(b):
        (xlo, xhi), (ylo, yhi) = b[k]
        print(f"scene {k}: x[{xlo:.0f},{xhi:.0f}] ({xhi-xlo:.0f}m)  "
              f"y[{ylo:.0f},{yhi:.0f}] ({yhi-ylo:.0f}m)")
