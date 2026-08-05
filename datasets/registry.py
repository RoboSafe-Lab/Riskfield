"""Dataset registry: switch the pipeline between InD and AD4CHE via ``RF_DATASET``.

Returns a dict describing the dataset so ``main.py`` and the eval scripts stay
dataset-agnostic: the loader class + root, the per-location boundary fn, the
feature normalization box, the vehicle-dimension table, and the map source.
"""

import os


def get_dataset(name=None):
    name = (name or os.environ.get("RF_DATASET", "ind")).lower()
    if name == "ad4che":
        from datasets.AD4CHE import (AD4CHE, AD4CHE_FEATURE_BOUNDARIES,
                                     compute_scene_boundaries)
        root = os.environ.get("RF_AD4CHE_ROOT", "data_ad4che")
        boxes = compute_scene_boundaries(root)
        return dict(
            name="ad4che", LoaderClass=AD4CHE, root=root,
            feature_boundaries=AD4CHE_FEATURE_BOUNDARIES,
            veh_lw={0: (4.5, 1.9), 1: (12.0, 2.6)},          # car, truck/bus (highway)
            boundaries_for_location=lambda L, _b=boxes: _b[int(L)],
            map_dataset="ad4che", map_data_dir=root,
        )
    if name == "round":
        from datasets.RounD import (RounD, ROUND_BOUNDARIES, ROUND_BG_SCALE_DOWN)
        from datasets.InD import feature_boundaries as _ind_fb
        root = os.environ.get("RF_ROUND_ROOT", "data_round")
        return dict(
            name="round", LoaderClass=RounD, root=root,
            # NOTE: the rounD cache/models normalize features with InD's box
            # (RounD does not override _collate_results), so consumers must use
            # the SAME box; ROUND_FEATURE_BOUNDARIES is intentionally unused.
            feature_boundaries=_ind_fb,
            veh_lw={0: (4.5, 1.9), 1: (10.0, 2.6)},          # car, truck_bus
            boundaries_for_location=lambda L, _b=ROUND_BOUNDARIES: _b[int(L)],
            map_dataset="round", map_data_dir=root,
            bg_scale_down=ROUND_BG_SCALE_DOWN,               # rounD bg PNGs are /10, not /12
        )
    from datasets.InD import InD, feature_boundaries, boundaries_for_location
    root = os.environ.get("RF_DATA_ROOT", "data")
    return dict(
        name="ind", LoaderClass=InD, root=root,
        feature_boundaries=feature_boundaries,
        veh_lw={0: (4.5, 1.9), 1: (10.0, 2.6), 2: (1.8, 0.6), 3: (0.7, 0.7)},
        boundaries_for_location=boundaries_for_location,
        map_dataset="ind", map_data_dir=root,
        bg_scale_down=12.0,                                  # InD drone-dataset-tools factor
    )
