"""Baseline risk detectors for the conflict-detection comparison.

All return a scalar risk score per scene (higher = riskier), so each can be used
as a binary detector of the surrogate conflict labels (scripts/conflict_labels.py).
All baselines see only PRESENT state + history (no future leakage); the learned
methods (Ours / PORA-style) use the model's predicted future densities.

- ttc_score      : kinematic min time-to-collision from the present state (CV).
- dsf_score      : handcrafted Driving Safety Field potential at the ego (present).
- pora_style_score: PORA-style spatio-temporal occupancy OVERLAP on the SAME
                    predicted densities (cell co-occupancy product; no exact
                    normalization, no oriented box, no physical severity).
- ours_peak      : peak of our two-model expected-energy field (J).
"""

import numpy as np

# NOTE: no DT here on purpose. Both baselines act on present-state velocities in
# m/s that the CALLER computes and passes in, so the frame interval belongs to the
# caller (see eval_conflict.py's RF_DT). A module-level DT here was dead and only
# invited someone to divide by it twice.
EPS = 1e-6


def ttc_score(p0, v0, P_others, V_others, tau_cap=10.0):
    """Present-state closing TTC. p0,v0: (2,) m, m/s; P_others,V_others: (M,2).
    Returns (score, min_ttc). score = 1/min_ttc (0 if never closing)."""
    min_ttc = np.inf
    for pj, vj in zip(P_others, V_others):
        d = pj - p0; v = vj - v0
        dv = float(d @ v)
        if dv < -EPS:                       # closing
            ttc = -float(d @ d) / dv
            if 0 < ttc < min_ttc:
                min_ttc = ttc
    min_ttc = min(min_ttc, tau_cap)
    return (1.0 / (min_ttc + 0.1) if np.isfinite(min_ttc) else 0.0), min_ttc


def dsf_score(p0, v0, P_others, V_others, beta=0.3):
    """Handcrafted Driving Safety Field potential at the ego from each neighbour:
    R_j = 1/(dist^2+eps) * exp(beta * closing_speed). Sum over neighbours.
    Captures the distance + relative-velocity field used by DSF methods."""
    R = 0.0
    for pj, vj in zip(P_others, V_others):
        d = pj - p0; dist = float(np.linalg.norm(d)) + EPS
        v = vj - v0
        closing = -float(d @ v) / dist      # >0 if approaching
        R += (1.0 / (dist * dist)) * np.exp(beta * max(closing, 0.0))
    return float(R)


def pora_style_score(P_ego_2d, dens, K):
    """PORA-style occupancy-overlap collision probability on the SAME predicted
    densities. risk_k = sum_j sum_g P_ego(g,k) * P_j(g,k); score = peak over k.
    Returns (peak, per_k profile (K,))."""
    per_k = np.zeros(K, np.float32)
    for j, Pj in dens.items():
        if j == 0:
            continue
        per_k += (P_ego_2d * Pj).reshape(-1, K).sum(0)   # cell co-occupancy
    return float(per_k.max()), per_k


def ours_peak(rf):
    """Our two-model expected-energy field rf (S,S,K) -> (peak J, per_k (K,))."""
    per_k = rf.reshape(-1, rf.shape[-1]).max(0)
    return float(per_k.max()), per_k
