import math
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class_to_id = {"car": 0, "truck_bus": 1, "bicycle": 2, "other": 3}

# InD has 4 physically distinct intersections (locationId 1..4) with different
# coordinate frames. Spatial normalization must therefore be PER LOCATION,
# otherwise recordings from other locations fall outside [0, 1] and corrupt
# training. Boxes below are min/max of xCenter/yCenter over ALL recordings of
# each location, padded by 5% (measured from the dataset; see
# scripts/measure_location_bounds if you need to recompute).
LOCATION_SPATIAL_BOUNDARIES = {
    1: np.array([[17.21, 90.95], [-71.60, 8.84]]),
    2: np.array([[2.47, 103.20], [-59.36, 1.09]]),
    3: np.array([[1.15, 90.79], [-74.95, 5.33]]),
    4: np.array([[41.79, 191.73], [-120.55, 3.25]]),
}


def boundaries_for_location(location_id):
    """Spatial normalization box for an InD locationId (defaults to loc 1)."""
    return LOCATION_SPATIAL_BOUNDARIES.get(int(location_id),
                                           LOCATION_SPATIAL_BOUNDARIES[1])


# Backward-compatible default (location 1; site 08 lives here). Scripts that
# import `spatial_boundaries` directly and run on a single location-1 site keep
# working; multi-location code paths use per-sample location boundaries.
spatial_boundaries = LOCATION_SPATIAL_BOUNDARIES[1]
# Feature channels (heading, vx, vy, ax, ay) are physical and location
# independent, so a single global box is correct.
feature_boundaries = np.array([[0, 360], [-10, 10], [-10, 10], [-5, 5], [-5, 5]])

spatial_keys = ["input", "target"]
feature_keys = ["feature"]
other_keys = ["type", "carMask", "trackId", "startFrame", "locationId"]
float_keys = spatial_keys + feature_keys
all_keys = float_keys + other_keys


def normalize(data, boundaries):
    return (data - boundaries[:, 0]) / (boundaries[:, 1] - boundaries[:, 0])


def denormalize(data, boundaries):
    return (data * (boundaries[:, 1] - boundaries[:, 0])) + boundaries[:, 0]


class DictDataset(Dataset):
    def __init__(self, **data_dict):
        sizes = [v.shape[0] for v in data_dict.values()]
        assert len(set(sizes)) == 1, "should have the same data size"

        self.data = data_dict
        self.data_size = sizes[0]

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.data.items()}

    def __len__(self):
        return self.data_size


class InDObservationSite:
    def __init__(
        self,
        background,
        ortho_px_to_meter,
        boundaries,
        train_loader: DataLoader,
        test_loader: DataLoader,
        loc_boundaries=None,
    ):
        self.background = background
        self.ortho_px_to_meter = ortho_px_to_meter
        # Primary box (first site's location); valid for single-location
        # loaders. For mixed-location loaders use denormalize_loc with the
        # per-sample locationId instead.
        self.boundaries = boundaries
        self.loc_boundaries = loc_boundaries or LOCATION_SPATIAL_BOUNDARIES
        self.train_loader = train_loader
        self.test_loader = test_loader

    def normalize(self, data):
        return normalize(data, self.boundaries)

    def denormalize(self, data):
        return denormalize(data, self.boundaries)

    def denormalize_loc(self, data, location_id):
        """Denormalize using a specific InD locationId's box (correct for
        multi-location loaders where samples carry a `locationId`)."""
        return denormalize(data, boundaries_for_location(location_id))


# inputs: (num_samples, max_num_cars, moving_window, 2)
# features: (num_samples, max_num_cars, moving_window, feat + 1 after time append)
# types: (num_samples, max_num_cars)
# targets: (num_samples, pred_len, 2)
# track_ids: (num_samples)
# start_frames: (num_samples)


class InD:
    def __init__(
        self,
        root,
        max_samples,
        train_ratio,
        train_batch_size,
        test_batch_size,
        max_num_cars=8,
        max_empty_frames=0,
        moving_window=100,
        seq_len=50,
        missing_rate=0.0,
        sampling_step=1,
        should_shuffle=True,
        include_future=True,
    ):
        self.root = root
        self.max_samples = max_samples
        self.train_ratio = train_ratio
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.max_num_cars = max_num_cars
        self.max_empty_frames = max_empty_frames
        self.moving_window = moving_window
        self.history_len = seq_len
        self.missing_rate = missing_rate
        self.sampling_step = sampling_step
        self.should_shuffle = should_shuffle
        self.include_future = include_future

        self.pred_len = moving_window - self.history_len

        self.observation_sites = {}
        # TODO: Note here
        # self.target_classes = ["car", "truck_bus", "bicycle"]
        self.target_classes = ["car", "truck_bus"]
        self.feature_cols = [
            "heading",
            "xVelocity",
            "yVelocity",
            "xAcceleration",
            "yAcceleration",
        ]
        self.input_cols = ["xCenter", "yCenter"]

        # Key config (keep default behavior unless include_future=True)
        self.spatial_keys = list(spatial_keys) + (["future"] if include_future else [])
        self.feature_keys = list(feature_keys)
        self.other_keys = list(other_keys)
        # 'dims' (recorded length,width) is float but NOT spatially normalized.
        self.float_keys = self.spatial_keys + self.feature_keys + ["dims"]
        self.all_keys = self.float_keys + self.other_keys

    # InD recording -> location grouping (fixed for the public dataset).
    LOCATION_RECORDINGS = {
        1: [f"{i:02d}" for i in range(7, 18)],    # 07..17
        2: [f"{i:02d}" for i in range(18, 30)],   # 18..29
        3: [f"{i:02d}" for i in range(30, 33)],   # 30..32
        4: [f"{i:02d}" for i in range(0, 7)],     # 00..06
    }

    @property
    def observation_site_08(self) -> InDObservationSite:
        return self._get_observation_site(["08"])

    @property
    def observation_site_all(self) -> InDObservationSite:
        """All 33 InD recordings, each spatially normalized by its own
        location box (per-location normalization)."""
        sites = sorted(s for v in self.LOCATION_RECORDINGS.values() for s in v)
        return self._get_observation_site(sites)

    def observation_site_location(self, location_id: int) -> InDObservationSite:
        return self._get_observation_site(self.LOCATION_RECORDINGS[int(location_id)])

    def observation_site_by_scope(self, scope) -> InDObservationSite:
        """Resolve a config `site_scope` string to an observation site.

        "08" -> single site; "all" -> all 33; "locN" -> intersection N.
        """
        scope = str(scope).strip().lower()
        if scope in ("all", "*"):
            return self.observation_site_all
        if scope.startswith("loc"):
            return self.observation_site_location(int(scope[3:]))
        return self._get_observation_site([scope])

    def _get_observation_site(self, sites):
        key = "-".join(sites)
        if key not in self.observation_sites:
            self.observation_sites[key] = self._load_observation_site(sites)
        return self.observation_sites[key]

    def boundaries_for_location(self, location_id):
        """Per-location spatial box. Overridable by dataset subclasses."""
        return boundaries_for_location(location_id)

    def _dim_columns(self):
        """(length_col, width_col) in tracksMeta for the recorded vehicle box.
        InD stores 'length'/'width'; AD4CHE overrides (its 'width'/'height')."""
        return ("length", "width")

    def _build_dim_lookup(self, tracks_meta):
        """trackId -> (length, width) [m] from the recorded per-track dimensions,
        so the qualitative figure can draw each agent at its true GT footprint."""
        lcol, wcol = self._dim_columns()
        if lcol not in tracks_meta.columns or wcol not in tracks_meta.columns:
            self._dim_lookup = {}
            return
        self._dim_lookup = {
            tid: (float(l), float(w))
            for tid, l, w in zip(
                tracks_meta["trackId"], tracks_meta[lcol], tracks_meta[wcol]
            )
        }

    def _cache_prefix(self, observation_sites):
        """Cache filename stem. Overridable by subclasses (e.g. when the site
        list is long enough to exceed the filesystem name limit)."""
        return (
            f"car_ind_{'-'.join(observation_sites)}"
            f"_maxcars{self.max_num_cars}"
            f"_window{self.moving_window}"
            f"_ratio{self.train_ratio}"
            f"_miss{self.missing_rate}"
            f"_step{self.sampling_step}"
            f"_maxsamp{self.max_samples}"
            f"_fut{int(self.include_future)}"
            f"_normPerLoc"  # per-location spatial normalization scheme
        )

    ## Section: Refactor mega function _parse into smaller functions
    def _load_and_clean_data(self, site):
        """IO and data cleaning."""

        def load(suffix):
            path = os.path.join(self.root, f"{site}_{suffix}.csv")
            return pd.read_csv(path)

        meta = load("recordingMeta")
        tracks = load("tracks")
        tracks_meta = load("tracksMeta")

        # Remove duplicate entries to ensure .loc[frame] works correctly
        tracks = tracks.drop_duplicates(subset=["trackId", "frame"])
        return meta, tracks, tracks_meta

    def _get_filtered_ids(
        self, tracks_meta, target_classes, min_frames=None, max_frames=None
    ):
        """Filter track IDs based on class and frame count."""
        mask = tracks_meta["class"].isin(target_classes)
        if min_frames is not None:
            mask &= tracks_meta["numFrames"] >= min_frames
        if max_frames is not None:
            mask &= tracks_meta["numFrames"] <= max_frames
        return tracks_meta[mask]["trackId"].values

    def _get_nearby_neighbors(
        self, ego_id, last_frame, ego_pos, tracks_by_id, neighbor_ids
    ):
        """Get nearby neighbor IDs."""
        neighbors = []
        for cand_id in neighbor_ids:
            if cand_id == ego_id:
                continue

            grp = tracks_by_id.get(cand_id)
            if grp is None or last_frame not in grp.index:
                continue

            row = grp.loc[last_frame]
            dist = math.hypot(row["xCenter"] - ego_pos[0], row["yCenter"] - ego_pos[1])
            neighbors.append((dist, cand_id))

        neighbors.sort(key=lambda x: x[0])
        return [item[1] for item in neighbors[: self.max_num_cars - 1]]

    def _create_single_sample(
        self,
        ego_id,
        window_df,
        tracks_by_id,
        neighbor_ids,
        start_frame,
        tracks_meta_dict,
        dummy_target=False,
        location_id=1,
    ):
        """Create a single sample on one window."""
        history_frames = window_df["frame"].values[: self.history_len]
        last_h_frame = history_frames[-1]
        cars_type = np.zeros((self.max_num_cars,), dtype=int)
        cars_dims = np.zeros((self.max_num_cars, 2), dtype=float)  # (length, width) [m]
        dim_lookup = getattr(self, "_dim_lookup", {})

        # Ego data
        ego_pos_all = window_df[self.input_cols].to_numpy()
        ego_feat_all = window_df[self.feature_cols].to_numpy()
        ego_class = tracks_meta_dict.get(ego_id, "other")
        cars_type[0] = class_to_id.get(ego_class, 3)
        cars_dims[0] = dim_lookup.get(ego_id, (0.0, 0.0))
        if cars_type[0] == 3:
            print(f"Warning: unknown class for trackId {ego_id}, set to 'other'")

        # Select neighbors
        ego_last_pos = ego_pos_all[self.history_len - 1]
        selected_nids = self._get_nearby_neighbors(
            ego_id, last_h_frame, ego_last_pos, tracks_by_id, neighbor_ids
        )

        # Initialize multi-car arrays
        cars_pos = np.full(
            (self.max_num_cars, self.history_len, len(self.input_cols)),
            np.nan,
            dtype=float,
        )
        cars_feat = np.full(
            (self.max_num_cars, self.history_len, len(self.feature_cols)),
            np.nan,
            dtype=float,
        )

        # Fill ego data (index 0)
        cars_pos[0] = ego_pos_all[: self.history_len]
        cars_feat[0] = ego_feat_all[: self.history_len]

        # Fill neighbor data
        for i, nid in enumerate(selected_nids, start=1):
            neighbor_df = tracks_by_id.get(nid)
            if neighbor_df is None:
                continue

            hist_data = neighbor_df.reindex(history_frames)
            cars_pos[i] = hist_data[self.input_cols].values
            cars_feat[i] = hist_data[self.feature_cols].values

            n_class = tracks_meta_dict.get(nid, "other")
            cars_type[i] = class_to_id.get(n_class, 3)
            cars_dims[i] = dim_lookup.get(nid, (0.0, 0.0))
            if cars_type[i] == 3:
                print(f"Warning: unknown class for trackId {nid}, set to 'other'")

        if not dummy_target:
            # Ego target (future positions)
            target_future = ego_pos_all[
                self.history_len : self.history_len + self.pred_len
            ]
        else:
            # Create dummy target filled with NaNs
            target_future = np.full(
                (self.pred_len, len(self.input_cols)), np.nan, dtype=float
            )

        sample = {
            "input": cars_pos,
            "feature": cars_feat,
            "type": cars_type,
            "dims": cars_dims,
            "target": target_future,
            "trackId": ego_id,
            "startFrame": start_frame,
            "locationId": int(location_id),
        }

        if self.include_future:
            # Future ground-truth positions for ego + selected neighbors.
            # Shape: (max_num_cars, pred_len, 2)
            future_frames = window_df["frame"].values[
                self.history_len : self.history_len + self.pred_len
            ]
            cars_future = np.full(
                (self.max_num_cars, self.pred_len, len(self.input_cols)),
                np.nan,
                dtype=float,
            )
            cars_future[0] = target_future

            for i, nid in enumerate(selected_nids, start=1):
                neighbor_df = tracks_by_id.get(nid)
                if neighbor_df is None:
                    continue
                fut_data = neighbor_df.reindex(future_frames)
                cars_future[i] = fut_data[self.input_cols].values

            sample["future"] = cars_future

        return sample

    def _collate_results(self, samples, meta, spatial_box):
        """Handle empty samples and collate results.

        `spatial_box` is the per-location spatial normalization box for the
        recording these samples came from.
        """
        if not samples:
            return {}

        # Dynamically stack/collate all keys except orthoPxToMeter
        result = {}
        for k in samples[0].keys():
            arrs = [s[k] for s in samples]
            arr = np.stack(arrs) if isinstance(arrs[0], np.ndarray) else np.array(arrs)
            if k in self.spatial_keys:
                arr = normalize(arr, spatial_box)
            elif k in self.feature_keys:
                arr = normalize(arr, feature_boundaries)
            result[k] = arr
        result["orthoPxToMeter"] = meta.at[0, "orthoPxToMeter"]
        return result

    def _parse(self, observation_site, max_samples=None):
        # File loading and cleaning
        meta, tracks, tracks_meta = self._load_and_clean_data(observation_site)
        tracks_meta_dict = dict(zip(tracks_meta["trackId"], tracks_meta["class"]))
        self._build_dim_lookup(tracks_meta)

        # Per-location spatial normalization box for this recording.
        location_id = int(meta.at[0, "locationId"])
        spatial_box = self.boundaries_for_location(location_id)

        # Filter target track IDs
        target_track_ids = self._get_filtered_ids(
            tracks_meta,
            self.target_classes,
            max_frames=2000,
            # min_frames=(self.moving_window - self.max_empty_frames)
            # * self.sampling_step,
            min_frames=self.moving_window * self.sampling_step,
        )
        neighbor_ids = self._get_filtered_ids(
            tracks_meta,
            self.target_classes,
            max_frames=2000,
            # min_frames=(self.moving_window - self.max_empty_frames)
            # * self.sampling_step,
            min_frames=self.moving_window * self.sampling_step,
        )

        # Prepare tracks by ID for fast lookup
        tracks_by_id = {
            tid: grp.set_index("frame") for tid, grp in tracks.groupby("trackId")
        }

        # Dense datasets (e.g. AD4CHE congested highway) can have hundreds of
        # qualifying egos per recording; cap them (and stride windows below) so
        # parsing is tractable. Defaults preserve InD behavior.
        cap = getattr(self, "max_egos_per_rec", None)
        if cap is not None and len(target_track_ids) > cap:
            target_track_ids = np.random.choice(target_track_ids, cap, replace=False)

        raw_samples = []

        # Extract samples for each target track
        for ego_id in target_track_ids:
            ego_full_df = tracks[tracks["trackId"] == ego_id].reset_index(drop=True)

            if len(ego_full_df) == 0:
                continue  # target id has no track rows (AD4CHE meta lists extra ids)
            if len(ego_full_df) != max(ego_full_df["trackLifetime"]) + 1:
                continue  # skip inconsistent data

            ego_full_df = ego_full_df.iloc[:: self.sampling_step, :].reset_index(
                drop=True
            )

            # start = max(self.moving_window - len(ego_full_df) - 1, 0)
            # end = min(self.max_empty_frames, len(ego_full_df))
            # for i in range(start, end):
            #     # pad with NaN rows at the beginning
            #     nan_rows = pd.DataFrame(
            #         np.nan, columns=ego_full_df.columns, index=range(i + 1)
            #     )
            #     window_df = pd.concat(
            #         [nan_rows, ego_full_df.iloc[: self.moving_window - (i + 1)]],
            #         ignore_index=True,
            #     )
            #     start_frame = (
            #         int(ego_full_df.loc[0, "frame"]) - (i + 1)
            #         if not ego_full_df.empty
            #         else -1
            #     )
            #     sample = self._create_single_sample(
            #         ego_id,
            #         window_df,
            #         tracks_by_id,
            #         neighbor_ids,
            #         start_frame,
            #         tracks_meta_dict,
            #     )
            #     raw_samples.append(sample)

            end = len(ego_full_df) - self.moving_window + 1
            for i in range(0, end, getattr(self, "window_stride", 1)):
                window_df = ego_full_df[i : i + self.moving_window].reset_index(
                    drop=True
                )
                start_frame = int(window_df.loc[0, "frame"])
                sample = self._create_single_sample(
                    ego_id,
                    window_df,
                    tracks_by_id,
                    neighbor_ids,
                    start_frame,
                    tracks_meta_dict,
                    location_id=location_id,
                )
                raw_samples.append(sample)

        # Subsample if exceeding max_samples
        if max_samples is not None and len(raw_samples) > max_samples:
            idx = np.random.choice(len(raw_samples), max_samples, replace=False)
            raw_samples = [raw_samples[i] for i in idx]

        # Return collated results as dict
        return self._collate_results(raw_samples, meta, spatial_box)

    def _load_observation_site(self, observation_sites):
        background = os.path.join(
            f"{self.root}", f"{observation_sites[0]}_background.png"
        )

        cache_dir = os.path.join(self.root, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_prefix = self._cache_prefix(observation_sites)
        cache_path = os.path.join(cache_dir, f"{cache_prefix}.pt")
        if os.path.exists(cache_path):
            print(f"Loading cached dataset from {cache_path}")
            # PyTorch 2.6+ defaults `weights_only=True`, which breaks loading cached
            # dataset tensors that were saved via `torch.save(dict_of_tensors, ...)`.
            # This cache is created locally by this project, so it is a trusted source.
            data = torch.load(cache_path, weights_only=False)
            payloads = {
                mode: {
                    k.replace(f"{mode}_", ""): v
                    for k, v in data.items()
                    if k.startswith(f"{mode}_")
                }
                for mode in ["train", "test"]
            }
            ortho_px_to_meter = data["orthoPxToMeter"]
        else:
            ortho_px_to_meter = 0
            all_data = {"train": [], "test": []}

            for observation_site in observation_sites:
                parsed = self._parse(observation_site, max_samples=self.max_samples)
                if not parsed or "input" not in parsed or len(parsed["input"]) == 0:
                    continue  # recording yielded no usable samples
                ortho_px_to_meter = parsed["orthoPxToMeter"]

                # Get random train-test split indices
                n_data = len(parsed["input"])
                n_train = int(n_data * self.train_ratio)
                indices = np.random.permutation(n_data)
                split_idx = {"train": indices[:n_train], "test": indices[n_train:]}

                # Split into train and test sets
                for mode, idxs in split_idx.items():
                    tensors = {}
                    for key, value in parsed.items():
                        if key in self.float_keys:
                            tensors[key] = torch.FloatTensor(value[idxs])
                        elif key in self.other_keys:
                            tensors[key] = torch.LongTensor(value[idxs])

                    if mode == "train":
                        self._mask(tensors["input"], tensors["feature"])
                    tensors["carMask"] = (
                        ~torch.isnan(tensors["input"]).all(dim=(2, 3))
                        & (tensors["input"] != 0).any(dim=(2, 3))
                    ).long()
                    tensors["feature"] = self._append_time(tensors["feature"])

                    all_data[mode].append(tuple(tensors[k] for k in self.all_keys))

            final_data = {}
            payloads = {}
            for mode in ["train", "test"]:
                cols = list(zip(*all_data[mode]))
                keys = self.all_keys
                merged = {
                    k: (
                        torch.cat(c, dim=0)
                        if k not in ["trackId", "startFrame"]
                        else torch.cat(c)
                    )
                    for k, c in zip(keys, cols)
                }

                payloads[mode] = merged
                for k, v in merged.items():
                    final_data[f"{mode}_{k}"] = v

            final_data["orthoPxToMeter"] = ortho_px_to_meter
            torch.save(final_data, cache_path)
            print(f"Saved cached dataset to {cache_path}")

        loaders = {}
        for mode in ["train", "test"]:
            dataset = DictDataset(**payloads[mode])
            batch_size = (
                self.train_batch_size if mode == "train" else self.test_batch_size
            )
            loaders[mode] = DataLoader(
                dataset, batch_size=batch_size, shuffle=self.should_shuffle
            )

        # Primary box = location of the first requested recording (valid for
        # single-location loaders; mixed loaders should use per-sample
        # locationId via denormalize_loc).
        try:
            primary_loc = int(
                pd.read_csv(
                    os.path.join(self.root, f"{observation_sites[0]}_recordingMeta.csv")
                ).at[0, "locationId"]
            )
        except Exception:
            primary_loc = 1

        return InDObservationSite(
            background=background,
            ortho_px_to_meter=ortho_px_to_meter,
            boundaries=boundaries_for_location(primary_loc),
            train_loader=loaders["train"],
            test_loader=loaders["test"],
            loc_boundaries=LOCATION_SPATIAL_BOUNDARIES,
        )

    def _append_time(self, b: torch.Tensor):
        # b: torch tensor shape (batch, max_num_cars, seq_len, _)
        batch_size, num_cars, seq_len, _ = b.shape
        time = 0.04 * seq_len  # 25Hz sampling rate
        t = torch.linspace(0.0, time, seq_len)
        t = t.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1,1,seq_len,1)
        t = t.expand(batch_size, num_cars, seq_len, 1)
        return torch.cat([b, t], dim=-1)

    def _mask(self, input, feature):
        # input: (num_samples, max_num_cars, seq_len, 2)
        # feature: (num_samples, max_num_cars, seq_len, feat)
        if self.missing_rate <= 0:
            return

        num_samples, _, seq_len, _ = input.shape

        num_mask = int(num_samples * self.missing_rate)
        if num_mask == 0:
            return
        mask_idx = np.random.choice(num_samples, num_mask, replace=False)

        for idx in mask_idx:
            ratio = np.random.uniform(0.1, 0.7)
            n = int(seq_len * ratio)
            if n == 0:
                continue
            input[idx, 0, :n, :] = float("nan")
            feature[idx, 0, :n, :] = float("nan")

    def get_specific_sample(self, site, ego_id, start_frame):
        """
        获取指定车辆和起始帧的单条数据样本，格式与 DataLoader 返回的 Batch 一致。
        """
        # 1. 加载并清理数据
        meta, tracks, tracks_meta = self._load_and_clean_data(site)
        tracks_meta_dict = dict(zip(tracks_meta["trackId"], tracks_meta["class"]))
        self._build_dim_lookup(tracks_meta)
        location_id = int(meta.at[0, "locationId"])
        spatial_box = self.boundaries_for_location(location_id)
        tracks_by_id = {
            tid: grp.set_index("frame") for tid, grp in tracks.groupby("trackId")
        }

        # 2. 提取 Ego 车辆的 window 数据
        ego_full_df = (
            tracks[tracks["trackId"] == ego_id]
            .sort_values("frame")
            .reset_index(drop=True)
        )
        # 考虑到 sampling_step
        ego_full_df = ego_full_df.iloc[:: self.sampling_step, :].reset_index(drop=True)

        # 找到对应 start_frame 的索引
        start_idx_list = ego_full_df.index[ego_full_df["frame"] == start_frame].tolist()
        if not start_idx_list:
            raise ValueError(f"Frame {start_frame} not found for trackId {ego_id}")

        idx = start_idx_list[0]
        window_df = ego_full_df.iloc[idx : idx + self.moving_window].reset_index(
            drop=True
        )

        if len(window_df) < self.moving_window:
            # 如果长度不足，向后补 NaN (模拟 parse 中的逻辑，虽然 parse 平常是向前补)
            nan_rows = pd.DataFrame(
                np.nan,
                columns=window_df.columns,
                index=range(self.moving_window - len(window_df)),
            )
            window_df = pd.concat([window_df, nan_rows], ignore_index=True)

        # 3. 准备邻居 ID
        neighbor_ids = self._get_filtered_ids(tracks_meta, self.target_classes)

        # 4. 创建单条样本 (numpy format)
        raw_sample = self._create_single_sample(
            ego_id, window_df, tracks_by_id, neighbor_ids, start_frame,
            tracks_meta_dict, location_id=location_id,
        )

        # 5. 标准化与格式转换 (模拟 _collate_results)
        sample_dict = {}
        for k, v in raw_sample.items():
            arr = np.expand_dims(v, axis=0)  # 增加 Batch 维度 (1, ...)
            if k in self.spatial_keys:
                arr = normalize(arr, spatial_box)
            elif k in self.feature_keys:
                arr = normalize(arr, feature_boundaries)

            # 转换为 Tensor
            if k in self.float_keys:
                sample_dict[k] = torch.FloatTensor(arr)
            else:
                sample_dict[k] = torch.LongTensor(arr)

        # 6. 生成 carMask 和 append_time (模拟 _load_observation_site)
        sample_dict["carMask"] = (
            ~torch.isnan(sample_dict["input"]).all(dim=(2, 3))
            & (sample_dict["input"] != 0).any(dim=(2, 3))
        ).long()

        sample_dict["feature"] = self._append_time(sample_dict["feature"])

        return sample_dict
