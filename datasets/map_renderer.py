"""Matplotlib-based OpenDRIVE map renderer for trajectory visualization.

Renders road surfaces, lane markings, and lane-type coloring as a background
layer for trajectory plots. Works in agent-centric coordinates by transforming
world-frame XODR geometry using the agents_from_world_tf matrix.

XODR parsing logic adapted from dfm/scripts/map_viewer.py to avoid pandas dep.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

# Recording → map stem mapping (same as map_viewer.py)
RECORDING_TO_MAP: Dict[int, str] = {
    **{i: "024" for i in range(1, 9)},
    9: "014", 10: "014", 14: "014",
    11: "015", 12: "015", 13: "015", 15: "015", 16: "015",
    17: "016",
    **{i: "017" for i in range(18, 24)},
    **{i: "017" for i in range(26, 33)},
    34: "017", 35: "017",
    **{i: "017" for i in range(39, 45)},
    47: "017", 48: "017", 51: "017",
    **{i: "017" for i in range(65, 69)},
    24: "018", 25: "018", 33: "018", 37: "018", 46: "018",
    36: "019", 38: "019", 45: "019", 49: "019", 50: "019", 58: "019",
    **{i: "020" for i in range(52, 58)},
    59: "021", 61: "021", 63: "021",
    62: "022", 64: "022",
    60: "023",
}

# Lane-type fill colors (RGBA, semi-transparent)
LANE_FILL_COLORS: Dict[str, str] = {
    "driving": "#e8e8e8",
    "exit": "#fff3d0",
    "entry": "#fff3d0",
    "onRamp": "#d4edda",
    "offRamp": "#d4edda",
    "shoulder": "#f0f0f0",
    "stop": "#f8d7da",
}
DEFAULT_FILL = "#eeeeee"

# Lane marking styles for matplotlib
MARKING_STYLES: Dict[str, dict] = {
    "solid": dict(color="#666666", linewidth=1.5, linestyle="-"),
    "broken": dict(color="#888888", linewidth=1.0, linestyle=(0, (5, 5))),
    "curb": dict(color="#8B4513", linewidth=2.0, linestyle="-"),
}
DEFAULT_MARKING = dict(color="#999999", linewidth=1.0, linestyle="-")

# Cache parsed maps to avoid re-parsing for each sample
_map_cache: Dict[str, Tuple[dict, dict]] = {}


def _world_to_local(
    world_pts: np.ndarray,
    agents_from_world_tf: np.ndarray,
) -> np.ndarray:
    """Transform world coordinates to agent-centric frame.

    Args:
        world_pts: [N, 2] or [N, 3] points in world (XODR) coordinates.
        agents_from_world_tf: [3, 3] or [4, 4] homogeneous transform.

    Returns:
        [N, 2] points in agent-local coordinates.
    """
    N = world_pts.shape[0]
    xy = world_pts[:, :2]

    if agents_from_world_tf.shape == (3, 3):
        # 2D affine: [3, 3] @ [x, y, 1]^T
        ones = np.ones((N, 1))
        homo = np.hstack([xy, ones])  # [N, 3]
        local = (agents_from_world_tf @ homo.T).T  # [N, 3]
        return local[:, :2]
    else:
        # 3D homogeneous: [4, 4] @ [x, y, z, 1]^T
        if world_pts.shape[1] >= 3:
            pts_3d = world_pts[:, :3]
        else:
            pts_3d = np.column_stack([xy, np.zeros(N)])
        ones = np.ones((N, 1))
        homo = np.hstack([pts_3d, ones])  # [N, 4]
        local = (agents_from_world_tf @ homo.T).T  # [N, 4]
        return local[:, :2]


def _eval_width_at(
    s_local: float,
    w_polys: List[Tuple[float, float, float, float, float]],
) -> float:
    """Evaluate lane width at a local s position using width polynomial records."""
    chosen = w_polys[0]
    for wp in w_polys:
        if wp[0] <= s_local + 1e-9:
            chosen = wp
        else:
            break
    s_off, a, b, c, d = chosen
    ds = s_local - s_off
    return a + b * ds + c * ds**2 + d * ds**3


def _parse_sections(
    xodr_str: str, resolution: float = 1.0,
) -> Tuple[Dict[str, SimpleNamespace], Dict[str, str]]:
    """Parse XODR into per-section lane geometries with successor links.

    Adapted from dfm/scripts/map_viewer.py to avoid pandas dependency.
    """
    try:                                   # vendored, self-contained (numpy+scipy only) -- preferred
        from datasets.xodr_min.geometry import (
            sample_centerline, recompute_headings,
        )
        from datasets.xodr_min.lane_processing import (
            _apply_lane_offset, DRIVEABLE_LANE_TYPES,
        )
    except Exception:                      # fall back to a full trajdata install if present
        from trajdata.dataset_specific.xodr.geometry import (
            sample_centerline, recompute_headings,
        )
        from trajdata.dataset_specific.xodr.lane_processing import (
            _apply_lane_offset, DRIVEABLE_LANE_TYPES,
        )

    root = ET.fromstring(xodr_str)
    section_lanes: Dict[str, SimpleNamespace] = {}
    successor_map: Dict[str, str] = {}

    road_connections: Dict[str, dict] = {}
    num_sections_per_road: Dict[str, int] = {}
    for road in root.findall("road"):
        rid = road.attrib["id"]
        lanes_elem = road.find("lanes")
        if lanes_elem is not None:
            num_sections_per_road[rid] = len(lanes_elem.findall("laneSection"))
        link = road.find("link")
        if link is None:
            continue
        info: dict = {}
        for tag in ("successor", "predecessor"):
            el = link.find(tag)
            if el is not None and el.attrib.get("elementType") == "road":
                info[tag] = {
                    "road_id": el.attrib["elementId"],
                    "contactPoint": el.attrib.get("contactPoint", "start"),
                }
        if info:
            road_connections[rid] = info

    for road in root.findall("road"):
        road_id = road.attrib["id"]
        cx, cy, cz, hdg = sample_centerline(road, resolution)
        if cx.size == 0:
            continue
        lanes_elem = road.find("lanes")
        if lanes_elem is None:
            continue

        s_grid = np.arange(len(cx)) * resolution
        cx, cy = _apply_lane_offset(lanes_elem, s_grid, cx, cy, hdg)
        sections = lanes_elem.findall("laneSection")
        sec_s_starts = [float(s.attrib.get("s", "0")) for s in sections]
        road_len = s_grid[-1] + resolution

        for si, section in enumerate(sections):
            s0 = sec_s_starts[si]
            s1 = sec_s_starts[si + 1] if si + 1 < len(sections) else road_len
            sec_len = s1 - s0
            i0 = max(0, int(np.searchsorted(s_grid, s0)))
            i1 = min(len(s_grid), int(np.searchsorted(s_grid, s1, side="right")))
            if i1 - i0 < 2:
                i1 = min(len(s_grid), i0 + 2)
                if i1 - i0 < 2:
                    continue

            scx, scy, scz, shdg = cx[i0:i1], cy[i0:i1], cz[i0:i1], hdg[i0:i1]
            n_pts = len(scx)
            sample_ds = np.linspace(0, sec_len, n_pts)

            for side_name in ("left", "right"):
                side_el = section.find(side_name)
                if side_el is None:
                    continue
                lane_els = sorted(
                    side_el.findall("lane"),
                    key=lambda l: abs(int(l.attrib["id"])),
                )
                direction = 1 if side_name == "left" else -1
                ref_widths = np.zeros(n_pts)

                for lane_el in lane_els:
                    lid = int(lane_el.attrib["id"])
                    ltype = lane_el.attrib.get("type", "none").lower()
                    is_drv = ltype in DRIVEABLE_LANE_TYPES

                    w_polys: List[Tuple[float, float, float, float, float]] = []
                    for w in lane_el.findall("width"):
                        w_polys.append((
                            float(w.attrib["sOffset"]),
                            float(w.attrib["a"]),
                            float(w.attrib["b"]),
                            float(w.attrib["c"]),
                            float(w.attrib["d"]),
                        ))
                    w_polys.sort(key=lambda x: x[0])
                    if not w_polys:
                        w_polys = [(0.0, 0.0, 0.0, 0.0, 0.0)]

                    lane_widths = np.array([
                        _eval_width_at(ds, w_polys) for ds in sample_ds
                    ])
                    lane_widths = np.maximum(lane_widths, 0.0)

                    theta = shdg + direction * np.pi / 2
                    nx = np.cos(theta)
                    ny = np.sin(theta)
                    inner_x = scx + ref_widths * nx
                    inner_y = scy + ref_widths * ny
                    outer_x = scx + (ref_widths + lane_widths) * nx
                    outer_y = scy + (ref_widths + lane_widths) * ny
                    mid_x = scx + (ref_widths + lane_widths / 2) * nx
                    mid_y = scy + (ref_widths + lane_widths / 2) * ny

                    road_marks: List[Tuple[float, str, str]] = []
                    for rm in lane_el.findall("roadMark"):
                        road_marks.append((
                            float(rm.attrib.get("sOffset", "0")),
                            rm.attrib.get("type", "none").lower(),
                            rm.attrib.get("color", "standard").lower(),
                        ))
                    road_marks.sort(key=lambda x: x[0])
                    if not road_marks:
                        road_marks = [(0.0, "none", "standard")]

                    uid = f"{road_id}_s{si}_{lid}"
                    section_lanes[uid] = SimpleNamespace(
                        center=np.stack([mid_x, mid_y, scz], axis=1),
                        left_edge=np.stack([inner_x, inner_y, scz], axis=1),
                        right_edge=np.stack([outer_x, outer_y, scz], axis=1),
                        headings=recompute_headings(mid_x, mid_y),
                        lane_type=ltype,
                        is_driving=is_drv,
                        unique_id=uid,
                        road_marks=road_marks,
                        sec_len=sec_len,
                    )
                    ref_widths = ref_widths + lane_widths

            center_el = section.find("center")
            if center_el is not None:
                for lane_el in center_el.findall("lane"):
                    if lane_el.attrib.get("id") != "0":
                        continue
                    road_marks = []
                    for rm in lane_el.findall("roadMark"):
                        road_marks.append((
                            float(rm.attrib.get("sOffset", "0")),
                            rm.attrib.get("type", "none").lower(),
                            rm.attrib.get("color", "standard").lower(),
                        ))
                    road_marks.sort(key=lambda x: x[0])
                    if not road_marks:
                        road_marks = [(0.0, "none", "standard")]
                    uid = f"{road_id}_s{si}_0"
                    section_lanes[uid] = SimpleNamespace(
                        center=np.stack([scx, scy, scz], axis=1),
                        left_edge=np.stack([scx, scy, scz], axis=1),
                        right_edge=np.stack([scx, scy, scz], axis=1),
                        headings=recompute_headings(scx, scy),
                        lane_type="none",
                        is_driving=False,
                        unique_id=uid,
                        road_marks=road_marks,
                        sec_len=sec_len,
                    )

    # Build successor map
    for road in root.findall("road"):
        road_id = road.attrib["id"]
        lanes_elem = road.find("lanes")
        if lanes_elem is None:
            continue
        sections = lanes_elem.findall("laneSection")
        n_sec = len(sections)
        for si, section in enumerate(sections):
            for side_name in ("left", "right"):
                side_el = section.find(side_name)
                if side_el is None:
                    continue
                for lane_el in side_el.findall("lane"):
                    lid = int(lane_el.attrib["id"])
                    uid = f"{road_id}_s{si}_{lid}"
                    if uid not in section_lanes:
                        continue
                    link_el = lane_el.find("link")
                    if link_el is None:
                        continue
                    succ = link_el.find("successor")
                    if succ is not None:
                        succ_lid = succ.attrib.get("id")
                        if succ_lid:
                            if si + 1 < n_sec:
                                target = f"{road_id}_s{si+1}_{succ_lid}"
                            else:
                                rc = road_connections.get(road_id, {})
                                sr = rc.get("successor")
                                if sr:
                                    t_road = sr["road_id"]
                                    cp = sr["contactPoint"]
                                    t_si = 0 if cp == "start" else (
                                        num_sections_per_road.get(t_road, 1) - 1
                                    )
                                    target = f"{t_road}_s{t_si}_{succ_lid}"
                                else:
                                    target = None
                            if target and target in section_lanes:
                                successor_map[uid] = target

    return section_lanes, successor_map


def _get_parsed_map(
    map_stem: str,
    map_dir: Path,
) -> Tuple[dict, dict]:
    """Parse XODR and cache result. Returns (section_lanes, successor_map)."""
    key = str(map_dir / f"{map_stem}.xodr")
    if key in _map_cache:
        return _map_cache[key]

    xodr_path = map_dir / f"{map_stem}.xodr"
    if not xodr_path.exists():
        raise FileNotFoundError(f"Map file not found: {xodr_path}")
    xodr_str = xodr_path.read_text(encoding="utf-8")
    section_lanes, successor_map = _parse_sections(xodr_str, resolution=1.0)

    _map_cache[key] = (section_lanes, successor_map)
    return section_lanes, successor_map


def render_road_background(
    ax: plt.Axes,
    recording_id: Optional[int],
    agents_from_world_tf: np.ndarray,
    map_dir: str | Path = "AD4CHE/maps/opendrive014-024",
    viewport: Optional[Tuple[float, float, float, float]] = None,
    margin: float = 20.0,
    map_stem: Optional[str] = None,
) -> bool:
    """Render OpenDRIVE road as matplotlib background on the given axes.

    Args:
        ax: Matplotlib axes to render on.
        recording_id: AD4CHE recording number (1-68). Ignored if map_stem is given.
        agents_from_world_tf: [4, 4] world-to-agent transform matrix.
        map_dir: Path to directory containing .xodr files.
        viewport: Optional (xmin, xmax, ymin, ymax) to clip rendering.
            If None, renders all lanes (slower for large maps).
        margin: Extra margin around viewport for clipping (meters).
        map_stem: Explicit .xodr file stem (e.g. "017"); when provided it takes
            precedence over the recording_id -> RECORDING_TO_MAP lookup. The
            AD4CHE .xodr stems are named by scene-directory number (014..024).

    Returns:
        True if map was rendered, False if fallback needed.
    """
    map_dir = Path(map_dir)
    if map_stem is None:
        map_stem = RECORDING_TO_MAP.get(recording_id)
    if map_stem is None:
        return False

    try:
        section_lanes, successor_map = _get_parsed_map(map_stem, map_dir)
    except (FileNotFoundError, ImportError):
        return False

    tf = np.array(agents_from_world_tf, dtype=np.float64)

    # Determine viewport for clipping
    if viewport is not None:
        xmin, xmax, ymin, ymax = viewport
    else:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
    clip_xmin = xmin - margin
    clip_xmax = xmax + margin
    clip_ymin = ymin - margin
    clip_ymax = ymax + margin

    # Collect patches and lines for batch rendering
    road_patches = []
    marking_segments = []
    marking_styles = []
    centerline_segments = []

    for uid, lane in section_lanes.items():
        if not lane.is_driving and lane.lane_type not in (
            "exit", "entry", "onRamp", "offRamp", "shoulder"
        ):
            continue

        # Transform edges to local coordinates
        left_local = _world_to_local(lane.left_edge[:, :2], tf)
        right_local = _world_to_local(lane.right_edge[:, :2], tf)

        # Quick viewport check — skip if entire lane is outside
        all_x = np.concatenate([left_local[:, 0], right_local[:, 0]])
        all_y = np.concatenate([left_local[:, 1], right_local[:, 1]])
        if (all_x.max() < clip_xmin or all_x.min() > clip_xmax or
                all_y.max() < clip_ymin or all_y.min() > clip_ymax):
            continue

        # Road surface polygon (left edge forward, right edge backward)
        poly_x = np.concatenate([left_local[:, 0], right_local[::-1, 0]])
        poly_y = np.concatenate([left_local[:, 1], right_local[::-1, 1]])
        poly_verts = np.column_stack([poly_x, poly_y])

        fill_color = LANE_FILL_COLORS.get(lane.lane_type, DEFAULT_FILL)
        patch = plt.Polygon(poly_verts, closed=True,
                            facecolor=fill_color, edgecolor="none",
                            alpha=0.6, zorder=0)
        road_patches.append(patch)

        # Lane center line (subtle)
        center_local = _world_to_local(lane.center[:, :2], tf)
        centerline_segments.append(center_local)

        # Road markings on outer edge
        road_marks = getattr(lane, "road_marks", [])
        n_pts = len(right_local)
        sec_len = getattr(lane, "sec_len", 1.0)
        sample_ds = np.linspace(0, sec_len, n_pts)

        for i, (s_off, mtype, mcolor) in enumerate(road_marks):
            if mtype == "none":
                continue
            next_s = road_marks[i + 1][0] if i + 1 < len(road_marks) else sec_len
            i_start = int(np.searchsorted(sample_ds, s_off))
            i_end = int(np.searchsorted(sample_ds, next_s, side="right"))
            if i_end <= i_start:
                i_end = min(n_pts, i_start + 2)
            seg = right_local[i_start:i_end]
            if len(seg) >= 2:
                marking_segments.append(seg)
                marking_styles.append(
                    MARKING_STYLES.get(mtype, DEFAULT_MARKING)
                )

    # Draw road surface patches
    for patch in road_patches:
        ax.add_patch(patch)

    # Draw lane center lines (very subtle)
    if centerline_segments:
        center_lc = LineCollection(
            centerline_segments,
            colors="#d0d0d0", linewidths=0.3, alpha=0.5, zorder=0,
        )
        ax.add_collection(center_lc)

    # Draw lane markings
    for seg, style in zip(marking_segments, marking_styles):
        ax.plot(seg[:, 0], seg[:, 1],
                color=style["color"],
                linewidth=style["linewidth"],
                linestyle=style.get("linestyle", "-"),
                alpha=0.7, zorder=0.5)

    return True
