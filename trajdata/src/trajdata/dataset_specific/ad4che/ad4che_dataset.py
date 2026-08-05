"""AD4CHE (Aerial Dataset for Chinese Highway and Expressway) dataset integration.

The AD4CHE dataset provides drone-recorded highway traffic data from Chinese
highways/expressways. It follows a similar format to the highD dataset with
tracks, tracksMeta, and recordingMeta CSV files per recording.

Data format details:
- Coordinates (x, y) are in meters in a local image-based coordinate system
- Velocity, acceleration are in m/s, m/s^2
- Frame rate is 30 Hz for all recordings
- Vehicle dimensions: 'width' is vehicle length, 'height' is vehicle width
  (bird's-eye view convention)
- 'orientation' is the heading angle in radians
- Recordings DJI_0001 to DJI_0068 (68 recordings across 11 locations)

Directory layout:
- Episodes are grouped by location: AD4CHE/<location_id>/DJI_XXXX/
  e.g., AD4CHE/13/DJI_0001/, AD4CHE/14/DJI_0009/, AD4CHE/17/DJI_0018/
- Tracks files are named XX_tracks_pixel.csv
- Maps (OpenDRIVE .xodr) are in AD4CHE/maps/ with 0-padded 3-digit names
  e.g., 013.xodr, 014.xodr, 017.xodr

Map support:
- OpenDRIVE (.xodr) maps cover all 68 recordings via 11 map files.
- RECORDING_TO_MAP defines which map file each recording uses.
"""

import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from tqdm import tqdm

from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures.agent import AgentMetadata, AgentType, FixedExtent
from trajdata.data_structures.environment import EnvMetadata
from trajdata.data_structures.scene_metadata import Scene, SceneMetadata
from trajdata.data_structures.scene_tag import SceneTag
from trajdata.dataset_specific.raw_dataset import RawDataset
from trajdata.dataset_specific.scene_records import Ad4cheRecord
from trajdata.maps import VectorMap
from trajdata.utils import arr_utils

# AD4CHE dataset constants
AD4CHE_DT: Final[float] = 1.0 / 30.0  # 30 fps

# Mapping from recording number → map file stem (without .xodr).
# Recordings sharing the same road segment point to the same map.
# The map stem is derived from the location directory name (zero-padded to 3 digits).
RECORDING_TO_MAP: Final[Dict[int, str]] = {
    # Map 024 — location dir "24"
    **{i: "024" for i in range(1, 9)},     # 1-8
    # Map 014 — location dir "14"
    9: "014", 10: "014", 14: "014",
    # Map 015 — location dir "15"
    11: "015", 12: "015", 13: "015", 15: "015", 16: "015",
    # Map 016 — location dir "16"
    17: "016",
    # Map 017 — location dir "17"
    **{i: "017" for i in range(18, 24)},   # 18-23
    **{i: "017" for i in range(26, 33)},   # 26-32
    34: "017", 35: "017",
    **{i: "017" for i in range(39, 45)},   # 39-44
    47: "017", 48: "017", 51: "017",
    **{i: "017" for i in range(65, 69)},   # 65-68
    # Map 018 — location dir "18"
    24: "018", 25: "018", 33: "018", 37: "018", 46: "018",
    # Map 019 — location dir "19"
    36: "019", 38: "019", 45: "019", 49: "019", 50: "019", 58: "019",
    # Map 020 — location dir "20"
    **{i: "020" for i in range(52, 58)},   # 52-57
    # Map 021 — location dir "21"
    59: "021", 61: "021", 63: "021",
    # Map 022 — location dir "22"
    62: "022", 64: "022",
    # Map 023 — location dir "23"
    60: "023",
}

# All recordings that have a known map
AD4CHE_RECORDINGS_WITH_MAP: Final[Tuple[int, ...]] = tuple(sorted(RECORDING_TO_MAP.keys()))

# All unique map locations
AD4CHE_MAP_LOCATIONS: Final[Tuple[str, ...]] = tuple(sorted(set(RECORDING_TO_MAP.values())))

# Default split definitions (can be overridden via config).
# 55 train, 13 val — every map location has at least 1 train recording.
# Maps with many recordings (024, 017, 020) get 2 val recordings for
# better coverage.  Single-recording maps (016, 023) go to train so the
# model sees every road geometry at least once.
AD4CHE_TRAIN_RECORDINGS: Final[Tuple[int, ...]] = (
    1, 2, 3, 4, 5, 6,                          # map 024 (6 train)
    9, 10,                                       # map 014 (2 train)
    11, 12, 13, 15,                              # map 015 (4 train)
    17,                                          # map 016 (1 train, sole rec)
    18, 19, 20, 21, 22, 23, 26, 27, 28, 29,     # map 017 (25 train)
    30, 31, 32, 34, 35, 39, 40, 41, 43, 44,
    47, 48, 51, 65, 66, 67,
    24, 25, 33, 37,                              # map 018 (4 train)
    36, 38, 45, 49, 50,                          # map 019 (5 train)
    52, 53, 54, 55,                              # map 020 (4 train)
    59, 61,                                      # map 021 (2 train)
    62,                                          # map 022 (1 train)
    60,                                          # map 023 (1 train, sole rec)
)
AD4CHE_VAL_RECORDINGS: Final[Tuple[int, ...]] = (
    7, 8,                                        # map 024
    14,                                          # map 014
    16,                                          # map 015
    42, 68,                                      # map 017
    46,                                          # map 018
    58,                                          # map 019
    56, 57,                                      # map 020
    63,                                          # map 021
    64,                                          # map 022
)


def ad4che_type_to_unified_type(label: str) -> AgentType:
    """Convert AD4CHE vehicle class to unified AgentType."""
    label = label.lower().strip()
    if label in ("car",):
        return AgentType.VEHICLE
    elif label in ("truck",):
        return AgentType.VEHICLE
    elif label in ("bus",):
        return AgentType.VEHICLE
    else:
        return AgentType.UNKNOWN


class Ad4cheDataset(RawDataset):
    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        # AD4CHE dataset tag parts: train and val splits
        dataset_parts: List[Tuple[str, ...]] = [
            ("train", "val"),
        ]

        return EnvMetadata(
            name=env_name,
            data_dir=data_dir,
            dt=AD4CHE_DT,
            parts=dataset_parts,
            scene_split_map=None,
            # Map locations for recordings with OpenDRIVE coverage
            map_locations=AD4CHE_MAP_LOCATIONS,
        )

    def load_dataset_obj(self, verbose: bool = False) -> None:
        if verbose:
            print(f"Loading {self.name} dataset...", flush=True)

        data_dir_path = Path(self.metadata.data_dir)

        # Store filepath, scene length, and recording meta per recording
        self.dataset_obj: Dict[str, Tuple[Path, Path, Path, int, int]] = dict()

        # Collect DJI_XXXX directories from location subdirs (14/, 15/, ...)
        # and also from the top level for backward compatibility.
        rec_dirs: List[Path] = []
        for loc_dir in sorted(data_dir_path.iterdir()):
            if loc_dir.is_dir() and loc_dir.name.isdigit():
                for rec_dir in sorted(loc_dir.glob("DJI_*")):
                    if rec_dir.is_dir():
                        rec_dirs.append(rec_dir)
        # Fallback: also check top-level DJI_* dirs (old flat layout)
        for rec_dir in sorted(data_dir_path.glob("DJI_*")):
            if rec_dir.is_dir() and rec_dir not in rec_dirs:
                rec_dirs.append(rec_dir)

        for rec_dir in sorted(rec_dirs, key=lambda p: p.name):
            rec_num_str = rec_dir.name.split("_")[1]  # e.g., "0009"
            rec_num = int(rec_num_str)
            prefix = str(rec_num).zfill(2)  # e.g., "09"

            tracks_path = rec_dir / f"{prefix}_tracks_pixel.csv"
            tracks_meta_path = rec_dir / f"{prefix}_tracksMeta.csv"
            recording_meta_path = rec_dir / f"{prefix}_recordingMeta.csv"

            if not tracks_path.exists():
                continue

            # Read recording meta to get frame rate and compute scene length
            rec_meta = pd.read_csv(recording_meta_path)
            frame_rate = int(rec_meta["frameRate"].values[0])

            # Read tracks meta to get max frame
            tracks_meta = pd.read_csv(tracks_meta_path)
            max_frame = int(tracks_meta["finalFrame"].max())
            scene_length = max_frame + 1  # frames are 0-indexed

            scene_name = f"DJI_{rec_num_str}"
            self.dataset_obj[scene_name] = (
                tracks_path,
                tracks_meta_path,
                recording_meta_path,
                scene_length,
                rec_num,
            )

        if verbose:
            print(
                f"Loaded {len(self.dataset_obj)} AD4CHE recordings.",
                flush=True,
            )

    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        all_scenes_list: List[Ad4cheRecord] = list()
        scenes_list: List[SceneMetadata] = list()

        for idx, (scene_name, (_, _, _, scene_length, rec_num)) in enumerate(
            sorted(self.dataset_obj.items())
        ):
            # Determine location based on recording number
            location = RECORDING_TO_MAP.get(rec_num, "unknown")

            # Determine split based on recording number
            if rec_num in AD4CHE_TRAIN_RECORDINGS:
                rec_split = "train"
            elif rec_num in AD4CHE_VAL_RECORDINGS:
                rec_split = "val"
            else:
                rec_split = "train"  # default: non-map recordings go to train

            all_scenes_list.append(
                Ad4cheRecord(scene_name, scene_length, location, idx)
            )

            if (
                rec_split in scene_tag
                and (
                    scene_desc_contains is None
                    or any(s in scene_name for s in scene_desc_contains)
                )
            ):
                scene_metadata = SceneMetadata(
                    env_name=self.metadata.name,
                    name=scene_name,
                    dt=self.metadata.dt,
                    raw_data_idx=idx,
                )
                scenes_list.append(scene_metadata)

        self.cache_all_scenes_list(env_cache, all_scenes_list)
        return scenes_list

    def _get_matching_scenes_from_cache(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[Scene]:
        all_scenes_list: List[Ad4cheRecord] = env_cache.load_env_scenes_list(
            self.name
        )

        scenes_list: List[Scene] = list()
        for scene_record in all_scenes_list:
            scene_name, scene_length, location, data_idx = scene_record

            # Determine split from recording number
            rec_num = int(scene_name.split("_")[1])
            if rec_num in AD4CHE_TRAIN_RECORDINGS:
                rec_split = "train"
            elif rec_num in AD4CHE_VAL_RECORDINGS:
                rec_split = "val"
            else:
                rec_split = "train"

            if (
                rec_split in scene_tag
                and (
                    scene_desc_contains is None
                    or any(s in scene_name for s in scene_desc_contains)
                )
            ):
                scene = Scene(
                    self.metadata,
                    scene_name,
                    location,
                    rec_split,
                    scene_length,
                    data_idx,
                    None,
                )
                scenes_list.append(scene)

        return scenes_list

    def get_scene(self, scene_info: SceneMetadata) -> Scene:
        _, scene_name, _, data_idx = scene_info

        if scene_name in self.dataset_obj:
            _, _, _, scene_length, rec_num = self.dataset_obj[scene_name]
        else:
            # Fallback
            scene_length = 0
            rec_num = int(scene_name.split("_")[1])

        location = RECORDING_TO_MAP.get(rec_num, "unknown")

        if rec_num in AD4CHE_TRAIN_RECORDINGS:
            rec_split = "train"
        elif rec_num in AD4CHE_VAL_RECORDINGS:
            rec_split = "val"
        else:
            rec_split = "train"

        return Scene(
            self.metadata,
            scene_name,
            location,
            rec_split,
            scene_length,
            data_idx,
            None,
        )

    def get_agent_info(
        self, scene: Scene, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        scene_name = scene.name

        # Check if already cached
        scene_metadata_path = EnvCache.scene_metadata_path(
            cache_path, scene.env_name, scene.name, scene.dt
        )
        if scene_metadata_path.exists():
            import time

            while True:
                try:
                    already_done_scene = EnvCache.load(scene_metadata_path)
                    break
                except Exception:
                    time.sleep(1)

            return (
                already_done_scene.agents,
                already_done_scene.agent_presence,
            )

        # Load data from source
        tracks_path, tracks_meta_path, _, scene_length, rec_num = self.dataset_obj[
            scene_name
        ]

        # Read tracks meta for vehicle dimensions and class
        tracks_meta_df = pd.read_csv(tracks_meta_path)

        # Read tracks data
        data_df = pd.read_csv(tracks_path)
        data_df.columns = data_df.columns.str.strip()

        # Rename columns to match trajdata conventions
        data_df.rename(
            columns={
                "frame": "scene_ts",
                "id": "agent_id",
                "x": "x",
                "y": "y",
                "xVelocity": "vx",
                "yVelocity": "vy",
                "xAcceleration": "ax",
                "yAcceleration": "ay",
                "orientation": "heading",
            },
            inplace=True,
        )

        # Add z coordinate (0 for drone-recorded highway data)
        data_df["z"] = 0.0

        # Convert agent_id to string for consistency
        data_df["agent_id"] = data_df["agent_id"].astype(str)

        # Build agent class and extent dictionaries from tracksMeta
        agent_class: Dict[str, str] = dict()
        agent_length: Dict[str, float] = dict()
        agent_width: Dict[str, float] = dict()

        for _, row in tracks_meta_df.iterrows():
            aid = str(int(row["id"]))
            agent_class[aid] = row["class"]
            # In AD4CHE: 'width' is vehicle length, 'height' is vehicle width
            agent_length[aid] = float(row["width"])
            agent_width[aid] = float(row["height"])

        # Select and sort the relevant columns
        data_df.set_index(["agent_id", "scene_ts"], inplace=True)
        data_df.sort_index(inplace=True)

        data_df.reset_index(level=1, inplace=True)

        # Build agent metadata and presence lists
        agent_list: List[AgentMetadata] = []
        agent_presence: List[List[AgentMetadata]] = [
            [] for _ in range(scene.length_timesteps)
        ]

        for agent_id, frames in data_df.groupby("agent_id")["scene_ts"]:
            start_frame = int(frames.iat[0])
            last_frame = int(frames.iat[-1])

            agent_type = ad4che_type_to_unified_type(
                agent_class.get(agent_id, "car")
            )

            # Vehicle height is approximately 1.5m for cars, 3.5m for trucks
            veh_length = agent_length.get(agent_id, 4.0)
            veh_width = agent_width.get(agent_id, 1.8)
            veh_height = 1.5 if veh_length < 6.0 else 3.5

            agent_metadata = AgentMetadata(
                name=str(agent_id),
                agent_type=agent_type,
                first_timestep=start_frame,
                last_timestep=last_frame,
                extent=FixedExtent(veh_length, veh_width, veh_height),
            )

            agent_list.append(agent_metadata)
            for frame in frames:
                frame_int = int(frame)
                if frame_int < scene.length_timesteps:
                    agent_presence[frame_int].append(agent_metadata)

        # Save agent data to cache
        scene_with_agents = Scene(
            env_metadata=scene.env_metadata,
            name=scene.name,
            location=scene.location,
            data_split=scene.data_split,
            length_timesteps=scene.length_timesteps,
            raw_data_idx=scene.raw_data_idx,
            data_access_info=scene.data_access_info,
            description=scene.description,
            agents=agent_list,
            agent_presence=agent_presence,
        )

        # Reset index for saving
        data_df.reset_index(inplace=True)
        data_df["agent_id"] = data_df["agent_id"].astype(str)
        data_df.set_index(["agent_id", "scene_ts"], inplace=True)

        cache_class.save_agent_data(
            data_df.loc[
                :,
                ["x", "y", "z", "vx", "vy", "ax", "ay", "heading"],
            ],
            cache_path,
            scene_with_agents,
        )
        EnvCache.save_scene_with_path(cache_path, scene_with_agents)

        return agent_list, agent_presence

    def _find_map_file(self, map_stem: str) -> Optional[Path]:
        """Search for a .xodr map file in candidate locations."""
        data_dir_path = Path(self.metadata.data_dir)
        map_file = f"{map_stem}.xodr"
        for candidate in [
            data_dir_path / map_file,
            data_dir_path / "maps" / map_file,
            data_dir_path / "maps" / "opendrive014-024" / map_file,
            data_dir_path.parent / "maps" / map_file,
            data_dir_path.parent / "maps" / "opendrive014-024" / map_file,
            data_dir_path.parent.parent / "maps" / map_file,
        ]:
            if candidate.exists():
                return candidate
        return None

    def cache_maps(
        self,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ) -> None:
        """Cache all OpenDRIVE maps referenced by RECORDING_TO_MAP.

        Uses the xodr parser to extract lane geometries and converts
        them to trajdata VectorMap RoadLane elements.
        """
        from trajdata.dataset_specific.xodr.parser import parse_xodr
        from trajdata.maps.vec_map_elements import Polyline, RoadLane

        for map_stem in AD4CHE_MAP_LOCATIONS:
            map_path = self._find_map_file(map_stem)

            if map_path is None:
                print(
                    f"Warning: AD4CHE map file '{map_stem}.xodr' not found. "
                    f"Skipping."
                )
                continue

            print(
                f"Caching AD4CHE map '{map_stem}' from {map_path} at "
                f"{map_params['px_per_m']:.2f} px/m"
            )

            # Read and parse the xodr file
            with open(map_path, "r", encoding="utf-8") as f:
                xodr_str = f.read()

            parsed = parse_xodr(xodr_str, resolution=0.5)

            # Create a VectorMap and populate with parsed lane data
            vector_map = VectorMap(map_id=f"{self.name}:{map_stem}")

            # Build stable numeric lane ID mapping
            lane_id_map = {
                old_id: str(i) for i, old_id in enumerate(parsed.lanes.keys())
            }

            max_point_dist = 2.0  # meters between interpolated polyline points

            for lg in parsed.lanes.values():
                center_pl = Polyline(lg.center).interpolate(max_dist=max_point_dist)
                left_pl = (
                    Polyline(lg.left_edge).interpolate(max_dist=max_point_dist)
                    if lg.left_edge is not None
                    else None
                )
                right_pl = (
                    Polyline(lg.right_edge).interpolate(max_dist=max_point_dist)
                    if lg.right_edge is not None
                    else None
                )

                # Remap connectivity to numeric IDs
                adj_left = {
                    lane_id_map[x]
                    for x in lg.can_change_left
                    if x in lane_id_map
                }
                adj_right = {
                    lane_id_map[x]
                    for x in lg.can_change_right
                    if x in lane_id_map
                }
                next_lanes = {
                    lane_id_map[x] for x in lg.next_lanes if x in lane_id_map
                }
                prev_lanes = {
                    lane_id_map[x] for x in lg.prev_lanes if x in lane_id_map
                }

                lane_elem = RoadLane(
                    id=lane_id_map[lg.unique_id],
                    center=center_pl,
                    left_edge=left_pl,
                    right_edge=right_pl,
                    adj_lanes_left=adj_left,
                    adj_lanes_right=adj_right,
                    next_lanes=next_lanes,
                    prev_lanes=prev_lanes,
                )
                vector_map.add_map_element(lane_elem)

            # Set the map extent
            vector_map.extent = parsed.extent

            map_cache_class.finalize_and_cache_map(cache_path, vector_map, map_params)
