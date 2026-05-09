from __future__ import annotations

from typing import Optional

import torch


def denormalize_xy(xy_norm: torch.Tensor) -> torch.Tensor:
    """Denormalize normalized (x,y) using datasets.InD.spatial_boundaries."""
    from datasets.InD import spatial_boundaries

    bounds = torch.as_tensor(
        spatial_boundaries, dtype=xy_norm.dtype, device=xy_norm.device
    )  # (2, 2)
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    return xy_norm * (hi - lo) + lo


def _apply_car_mask(
    future_xy: torch.Tensor,
    car_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if car_mask is None:
        return future_xy
    if car_mask.dim() != 2 or car_mask.shape[:2] != future_xy.shape[:2]:
        raise ValueError("car_mask must have shape (B, N)")
    mask = car_mask.to(dtype=torch.bool, device=future_xy.device)
    return future_xy.masked_fill(~mask[:, :, None, None], float("nan"))


def compute_min_distance_ego_neighbors_from_future(
    future_xy_norm: torch.Tensor,
    *,
    car_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Minimum ego-to-neighbor distance over the future horizon (ground-truth).

    Only checks ego (index 0) vs other vehicles (1..N-1). Does NOT check neighbor-neighbor.

    Args:
        future_xy_norm: (B, N, T, 2) normalized positions.
        car_mask: (B, N) optional mask (1/0) indicating which vehicles are present.

    Returns:
        min_dist_m: (B,) minimum center-to-center distance in meters.
            If there are no neighbors or no valid positions, returns +inf.
    """
    if future_xy_norm.dim() != 4 or future_xy_norm.size(-1) != 2:
        raise ValueError("future_xy_norm must have shape (B, N, T, 2)")

    B, N, T, _ = future_xy_norm.shape
    if N < 2 or T < 1:
        return torch.full((B,), float("inf"), device=future_xy_norm.device)

    future_xy = denormalize_xy(future_xy_norm)
    future_xy = _apply_car_mask(future_xy, car_mask)

    ego = future_xy[:, 0:1, :, :]  # (B,1,T,2)
    nbr = future_xy[:, 1:, :, :]  # (B,M,T,2)
    dist = torch.linalg.norm(nbr - ego, dim=-1)  # (B,M,T)
    dist = torch.where(torch.isnan(dist), torch.full_like(dist, float("inf")), dist)
    return dist.amin(dim=(1, 2))


def compute_min_distance_ego_neighbors_from_batch(batch: dict) -> torch.Tensor:
    if "future" not in batch:
        raise KeyError("Batch must contain 'future'. Create InD(..., include_future=True).")
    return compute_min_distance_ego_neighbors_from_future(
        batch["future"],
        car_mask=batch.get("carMask"),
    )


def _cross2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def compute_has_intersection_ego_neighbors_from_future(
    future_xy_norm: torch.Tensor,
    *,
    car_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Whether ego future polyline intersects any neighbor future polyline.

    Only checks ego (index 0) vs other vehicles (1..N-1). Does NOT check neighbor-neighbor.

    Returns:
        has_intersection: (B,) bool tensor.
    """
    if future_xy_norm.dim() != 4 or future_xy_norm.size(-1) != 2:
        raise ValueError("future_xy_norm must have shape (B, N, T, 2)")

    B, N, T, _ = future_xy_norm.shape
    if N < 2 or T < 2:
        return torch.zeros((B,), dtype=torch.bool, device=future_xy_norm.device)

    future_xy = denormalize_xy(future_xy_norm)
    future_xy = _apply_car_mask(future_xy, car_mask)

    ego = future_xy[:, 0, :, :]  # (B,T,2)
    nbr = future_xy[:, 1:, :, :]  # (B,M,T,2)
    S = T - 1

    ego0 = ego[:, :-1, :]  # (B,S,2)
    ego1 = ego[:, 1:, :]
    nbr0 = nbr[:, :, :-1, :]  # (B,M,S,2)
    nbr1 = nbr[:, :, 1:, :]

    valid_ego = (~torch.isnan(ego0).any(dim=-1)) & (~torch.isnan(ego1).any(dim=-1))  # (B,S)
    valid_nbr = (~torch.isnan(nbr0).any(dim=-1)) & (~torch.isnan(nbr1).any(dim=-1))  # (B,M,S)

    r = ego1 - ego0  # (B,S,2)
    s = nbr1 - nbr0  # (B,M,S,2)

    P0 = ego0[:, None, :, None, :]  # (B,1,S,1,2)
    Q0 = nbr0[:, :, None, :, :]  # (B,M,1,S,2)
    r_b = r[:, None, :, None, :]  # (B,1,S,1,2)
    s_b = s[:, :, None, :, :]  # (B,M,1,S,2)

    denom = _cross2(r_b, s_b)  # (B,M,S,S)
    qmp = Q0 - P0  # (B,M,S,S,2)

    t = _cross2(qmp, s_b) / (denom + eps)  # (B,M,S,S)
    u = _cross2(qmp, r_b) / (denom + eps)

    valid_pair = valid_ego[:, None, :, None] & valid_nbr[:, :, None, :]
    intersects = (
        (denom.abs() > eps)
        & (t >= 0.0)
        & (t <= 1.0)
        & (u >= 0.0)
        & (u <= 1.0)
        & valid_pair
    )

    return intersects.any(dim=(1, 2, 3))


def compute_has_intersection_ego_neighbors_from_batch(batch: dict) -> torch.Tensor:
    if "future" not in batch:
        raise KeyError("Batch must contain 'future'. Create InD(..., include_future=True).")
    return compute_has_intersection_ego_neighbors_from_future(
        batch["future"],
        car_mask=batch.get("carMask"),
    )
