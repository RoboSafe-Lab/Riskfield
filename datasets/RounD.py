"""rounD loader (roundabout drone dataset), subclassing the InD loader.

rounD uses the SAME drone-dataset-tools format as InD (``NN_tracks.csv`` /
``NN_tracksMeta.csv`` / ``NN_recordingMeta.csv`` / ``NN_background.png``), so it
reuses InD's pipeline almost unchanged. The provided subset is 22 recordings
(``02``..``23``), ALL at ``locationId 0`` (one roundabout, one coordinate frame,
25 fps) -> we treat it as a single location (like one InD intersection) with
per-location spatial normalization and map conditioning off the per-recording
background. rounD's richer class vocabulary is mapped onto InD's 2-class head
(car / truck_bus); VRUs/2-wheelers are dropped from the eligible ego/neighbour
set (target_classes), matching the vehicle-focused risk field.
"""

import os
import glob
import hashlib

import numpy as np
import pandas as pd

from datasets.InD import InD

ROUND_ROOT_DEFAULT = "data_round"            # local + cluster data dir
ROUND_RECORDINGS = [f"{i:02d}" for i in range(2, 24)]      # 02..23, all locationId 0
# Background-PNG downscale factor vs the ortho pixel scale. Official value from
# drone-dataset-tools data/visualizer_params/visualizer_params.json: rounD uses
# scale_down_factor=10 (InD uses 12). Also validated by projecting track points
# onto the background: at 10 the ring + every entry arm align; 12 is offset.
ROUND_BG_SCALE_DOWN = 10.0

# rounD class names -> InD's 2-class vocabulary (car=0, truck_bus=1; bicycle=2,
# other=3 are excluded from target_classes so they never act as ego/neighbour).
ROUND_CLASS_MAP = {
    "car": "car", "van": "car",
    "truck": "truck_bus", "bus": "truck_bus", "trailer": "truck_bus",
    "motorcycle": "other", "bicycle": "bicycle", "pedestrian": "other",
}

# Single-location spatial box (metres), pooled min/max of xCenter/yCenter over all
# 22 recordings + 5% pad (measured: x[14.2,143.5], y[-101.7,0.2]). y is negative-
# down as in InD. Used for per-location normalization AND the map crop.
ROUND_BOUNDARIES = {0: np.array([[7.7, 150.0], [-106.8, 5.3]])}

# heading/velocity/accel box (roundabout speeds are modest; widened vs InD's
# +/-10 so approach-road speeds are not pushed far outside [0,1]).
ROUND_FEATURE_BOUNDARIES = np.array([[0, 360], [-16, 16], [-16, 16], [-6, 6], [-6, 6]])


def compute_round_boundaries(root=ROUND_ROOT_DEFAULT, pad_frac=0.05):
    """Recompute the pooled spatial box from the data (fallback: ROUND_BOUNDARIES)."""
    xs_lo = xs_hi = ys_lo = ys_hi = None
    for rec in ROUND_RECORDINGS:
        f = os.path.join(root, f"{rec}_tracks.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f, usecols=["xCenter", "yCenter"])
        xlo, xhi = float(d["xCenter"].min()), float(d["xCenter"].max())
        ylo, yhi = float(d["yCenter"].min()), float(d["yCenter"].max())
        xs_lo = xlo if xs_lo is None else min(xs_lo, xlo)
        xs_hi = xhi if xs_hi is None else max(xs_hi, xhi)
        ys_lo = ylo if ys_lo is None else min(ys_lo, ylo)
        ys_hi = yhi if ys_hi is None else max(ys_hi, yhi)
    if xs_lo is None:
        return dict(ROUND_BOUNDARIES)
    px, py = (xs_hi - xs_lo) * pad_frac, (ys_hi - ys_lo) * pad_frac
    return {0: np.array([[xs_lo - px, xs_hi + px], [ys_lo - py, ys_hi + py]])}


class RounD(InD):
    """rounD loader: subclasses InD, remaps the class vocabulary, and registers all
    22 recordings under the single physical location 0 with per-location norm."""

    LOCATION_RECORDINGS = {0: list(ROUND_RECORDINGS)}

    def __init__(self, root="data_round", **kw):
        super().__init__(root=root, **kw)
        self.target_classes = ["car", "truck_bus"]      # after class rename
        self._boxes = ROUND_BOUNDARIES
        # Roundabout recordings are long+dense (hundreds of egos, thousands of
        # frames). InD's _parse enumerates ALL egos x ALL windows before
        # subsampling to max_samples, so without these caps the parse explodes
        # (left it 2h+ with no cache). Same throttle AD4CHE uses for congestion.
        self.max_egos_per_rec = int(os.environ.get("RF_ROUND_EGOS", "40"))
        self.window_stride = int(os.environ.get("RF_ROUND_WSTRIDE", "25"))

    def boundaries_for_location(self, location_id):
        return self._boxes[int(location_id)]

    def _load_and_clean_data(self, site):
        """InD-format IO, then map rounD classes onto InD's 2-class vocabulary."""
        meta, tracks, tracks_meta = super()._load_and_clean_data(site)
        tracks_meta = tracks_meta.copy()
        tracks_meta["class"] = tracks_meta["class"].map(
            lambda c: ROUND_CLASS_MAP.get(str(c).strip().lower(), "other"))
        return meta, tracks, tracks_meta

    def _cache_prefix(self, observation_sites):
        h = hashlib.md5("-".join(observation_sites).encode()).hexdigest()[:8]
        return (f"round_{len(observation_sites)}recs_{h}"
                f"_maxcars{self.max_num_cars}_window{self.moving_window}"
                f"_ratio{self.train_ratio}_miss{self.missing_rate}"
                f"_step{self.sampling_step}_maxsamp{self.max_samples}"
                f"_fut{int(self.include_future)}_normPerLoc")


if __name__ == "__main__":   # boundaries smoke
    b = compute_round_boundaries(os.environ.get("RF_ROUND_ROOT", ROUND_ROOT_DEFAULT))
    (xlo, xhi), (ylo, yhi) = b[0]
    print(f"rounD loc0: x[{xlo:.1f},{xhi:.1f}] ({xhi-xlo:.0f}m)  "
          f"y[{ylo:.1f},{yhi:.1f}] ({yhi-ylo:.0f}m)  recs={len(ROUND_RECORDINGS)}")
