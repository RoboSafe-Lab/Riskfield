"""Shared two-model risk-field engine (single source of truth for the field math).

Both the renderer (animate_joint.py) and the calibrator (calibrate_joint.py)
import this so T_critical is computed against the EXACT field they display.

Field:  R_k(g) = P_ego(g,k) · Σ_j (P_j^joint(·,k|a_ego) ∗ 1_box)(g) · C_j(g,k)
  - P_ego        : ind_8 single-target marginal + map (the ego's own motion);
  - P_j^joint    : ind_7 scene-level AR joint, ego-conditioned, evaluated EXACTLY
                   on the grid via an AR-MAP rollout (ego MAP first, then each
                   nearest-first neighbour's per-frame MAP fills Y_<i);
  - meeting prob : per-cell oriented-box SAT overlap (headings from optical flow);
  - C_j          : 1/2 μ ||v_ego(g,k) − v_j(g,k)||² with per-cell OF velocities.
The map is load-bearing for BOTH models, so location_id is always passed.
"""

import os
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, convolve as ndi_convolve, shift as ndi_shift
from scipy.signal import savgol_filter

DT = float(os.environ.get("RF_DT", "0.08"))   # InD 25fps*step2=0.08; AD4CHE 30fps*step2=0.0667
M_R = 1500.0 / 2.0                       # reduced mass of two 1500 kg vehicles (kg)
A_SCALE = 100.0                          # ego action proxy scale -- MUST match training
DILATE_SIGMA = 2.0
MC_SIGMA = float(os.environ.get("RF_MC_SIGMA", "1.0"))   # KDE bandwidth (cells) for Monte-Carlo occupancy
SG_WIN, SG_POLY = 11, 2                  # Savitzky-Golay window/order for centroid vel
# Vehicle (length,width) by class id -- dataset-aware (AD4CHE: highway truck/bus).
if os.environ.get("RF_DATASET", "ind").lower() == "ad4che":
    VEH_LW = {0: (4.5, 1.9), 1: (12.0, 2.6)}
else:
    VEH_LW = {0: (4.5, 1.9), 1: (10.0, 2.6), 2: (1.8, 0.6), 3: (0.7, 0.7)}  # L,W (m)
VCAP = 30.0                              # physical per-cell speed cap (m/s)
# Agents whose OBSERVED speed is below this (m/s) are treated as static: their
# heading is ill-defined (no motion to derive it from) and the predicted-density
# optical flow / centroid yields a spurious velocity+heading, which mis-orients a
# long parked-vehicle box across the ego's lane -> false collision. For such
# agents we skip velocity/heading estimation and use a zero velocity + the stable
# OBSERVED yaw, so a side-pass of a parked vehicle is correctly non-colliding.
V_STATIC = float(os.environ.get("RF_VSTATIC", "0.5"))

_HS_AVG = np.array([[1, 2, 1], [2, 0, 2], [1, 2, 1]], np.float32) / 12.0


def base_logpx(z, det):
    d = z.shape[-1]
    return -0.5 * (z.pow(2).sum(-1) + d * np.log(2 * np.pi)) - det


def agent_class(vt, a):
    return int(vt[0, a].reshape(-1)[0])


def horn_schunck(I1, I2, alpha=0.5, n_iter=40):
    s = max(float(I1.max()), float(I2.max()), 1e-12)
    I1 = (I1 / s).astype(np.float32); I2 = (I2 / s).astype(np.float32)
    Ix = 0.5 * (np.gradient(I1, axis=0) + np.gradient(I2, axis=0))
    Iy = 0.5 * (np.gradient(I1, axis=1) + np.gradient(I2, axis=1))
    It = I2 - I1
    denom = alpha ** 2 + Ix ** 2 + Iy ** 2
    u = np.zeros_like(I1); w = np.zeros_like(I1)
    for _ in range(n_iter):
        ubar = ndi_convolve(u, _HS_AVG, mode="nearest")
        wbar = ndi_convolve(w, _HS_AVG, mode="nearest")
        d = (Ix * ubar + Iy * wbar + It) / denom
        u = ubar - Ix * d
        w = wbar - Iy * d
    return u, w


def _sat_mask(dxm, dym, the, hle, hwe, thj, hlj, hwj):
    ue = (np.cos(the), np.sin(the)); ve = (-np.sin(the), np.cos(the))
    uj = (np.cos(thj), np.sin(thj)); vj = (-np.sin(thj), np.cos(thj))
    sep = np.zeros(dxm.shape, dtype=bool)
    for n in (ue, ve, uj, vj):
        re = hle * abs(ue[0] * n[0] + ue[1] * n[1]) + hwe * abs(ve[0] * n[0] + ve[1] * n[1])
        rj = hlj * abs(uj[0] * n[0] + uj[1] * n[1]) + hwj * abs(vj[0] * n[0] + vj[1] * n[1])
        sep |= np.abs(dxm * n[0] + dym * n[1]) > (re + rj)
    return ~sep


def of_heading_field(vfield, obs_yaw):
    sp = np.hypot(vfield[..., 0], vfield[..., 1])
    th = np.arctan2(vfield[..., 1], vfield[..., 0])
    return np.where(sp < 0.5, obs_yaw, th).astype(np.float32)


class JointRiskField:
    """Two-model engine. Build once with the loaded models + grid; call .field()
    per scene window."""

    def __init__(self, m_ego, m_joint, grid, S, K, dev, min_hist=30):
        self.m_ego, self.m_joint = m_ego, m_joint
        self.grid, self.S, self.K, self.dev = grid, S, K, dev
        self.G = grid.shape[0]
        self.min_hist = min_hist

    # ---- ego marginal (ind_8 + map) ------------------------------------
    def ego_density(self, x, feat, vt, a, loc_t):
        m = self.m_ego; K = self.K; G = self.G
        xr = torch.roll(x, -a, 1); fr = torch.roll(feat, -a, 1); vr = torch.roll(vt, -a, 1)
        emb, _ = m.encoder(None, torch.cat([xr, fr], -1), vr, per_agent=False)
        memb = m._map_emb(loc_t, self.dev)
        if memb is not None:
            emb = emb + memb
        cond = m._flow_condition(emb, K, None)
        if cond.dim() == 2:
            cond = cond.unsqueeze(1).expand(-1, K, -1)
        condG = cond.expand(G, K, cond.shape[-1]).contiguous()
        y = self.grid.view(G, 1, 2).expand(G, K, 2).contiguous()
        z, det = m.flow(y, condG, sampling_frequency=1)
        P = base_logpx(z, det).exp()
        return P / P.sum(0, keepdim=True).clamp(min=1e-9)

    # ---- learned per-cell velocity field (S,S,K,2) [m/s] ----------------
    def _obs_velocity(self, xw, a, scale):
        """Observed velocity vector (vx,vy) [m/s] of agent a from its last (up to
        10) valid history frames -- the physical anchor for the learned residual."""
        p = xw[0, a]; valid = ~torch.isnan(p[:, 0]); p = p[valid]
        if p.shape[0] < 2:
            return np.zeros(2, np.float32)
        n = min(10, p.shape[0] - 1)
        d = (p[-1] - p[-1 - n]) * scale / (n * DT)              # (2,) m/s
        return d.detach().cpu().numpy().astype(np.float32)

    def _vel_delta_from_cond(self, m, condG):
        """Learned m/s residual at every grid cell -> (S,S,K,2)."""
        yq = self.grid.view(self.G, 1, 2).expand(self.G, self.K, 2).contiguous()
        delta = m.velocity_at(yq, condG)                         # (G,K,2) m/s residual
        return delta.reshape(self.S, self.S, self.K, 2).detach().cpu().numpy()

    def ego_velocity_field(self, x, feat, vt, a, loc_t, scale):
        m = self.m_ego; K = self.K; G = self.G
        xr = torch.roll(x, -a, 1); fr = torch.roll(feat, -a, 1); vr = torch.roll(vt, -a, 1)
        emb, _ = m.encoder(None, torch.cat([xr, fr], -1), vr, per_agent=False)
        memb = m._map_emb(loc_t, self.dev)
        if memb is not None:
            emb = emb + memb
        cond = m._flow_condition(emb, K, None)
        if cond.dim() == 2:
            cond = cond.unsqueeze(1).expand(-1, K, -1)
        condG = cond.expand(G, K, cond.shape[-1]).contiguous()
        vo = self._obs_velocity(x, a, scale)                    # (2,) m/s anchor
        return vo[None, None, None, :] + self._vel_delta_from_cond(m, condG)

    # ---- ind_7 joint, ego-conditioned others (AR-MAP rollout) ----------
    def _joint_cond_density(self, agent_emb, s_seq, y_fill, order, i, scale=None):
        m = self.m_joint; K = self.K; G = self.G
        cond = m.ar_decoder(agent_emb, s_seq, y_fill, order)     # (1,N,K,E)
        ci = cond[:, i]
        condG = ci.expand(G, K, ci.shape[-1]).contiguous()
        yq = self.grid.view(G, 1, 2).expand(G, K, 2).contiguous()
        z, det = m.flow(yq, condG, sampling_frequency=1)
        P = base_logpx(z, det).exp()
        P = P / P.sum(0, keepdim=True).clamp(min=1e-9)
        if scale is not None:
            return P, self._vel_delta_from_cond(m, condG)       # m/s residual (anchor added by caller)
        return P

    def joint_densities(self, xw, fw, vt, P_ego, contributors, loc_t, scale=None):
        m = self.m_joint; K = self.K; grid = self.grid
        agent_emb, car_valid = m.encoder(
            None, torch.cat([xw, fw], -1), vt, per_agent=True)
        memb = m._map_emb(loc_t, self.dev)
        if memb is not None:
            agent_emb = agent_emb + memb.unsqueeze(1)
        ego_pos = grid[P_ego.argmax(0)]                          # (K,2) normalized
        a = ego_pos[2:] - 2 * ego_pos[1:-1] + ego_pos[:-2]
        a_ego = torch.cat([a[:1], a, a[-1:]], 0)[None] * A_SCALE  # (1,K,2)
        s_seq = m.world_model.forward_scene(agent_emb, car_valid, K, a_ego,
                                            return_dyn=False)
        order = m._compute_ordering(xw, car_valid)
        y_fill = torch.zeros(1, xw.shape[1], K, 2, device=self.dev)
        y_fill[:, 0] = ego_pos[None]
        out = {}; vel = {}
        for i in order[0].tolist():
            if i == 0 or not bool(car_valid[0, i]):
                continue
            res = self._joint_cond_density(agent_emb, s_seq, y_fill, order, i, scale=scale)
            P, vfield = res if scale is not None else (res, None)
            y_fill[0, i] = grid[P.argmax(0)]
            if i in contributors:
                out[i] = P
                if vfield is not None:                           # anchor + learned residual (m/s)
                    vel[i] = self._obs_velocity(xw, i, scale)[None, None, None, :] + vfield
        return (out, vel) if scale is not None else out

    # ---- per-cell optical-flow velocity field --------------------------
    def optical_flow_vel(self, P, scale_t):
        S, K = self.S, self.K
        arr = P.reshape(S, S, K).detach().cpu().numpy()
        cx, cy = float(scale_t[0]) / S, float(scale_t[1]) / S
        v = np.zeros((S, S, K, 2), np.float32)
        for k in range(K - 1):
            u, w = horn_schunck(arr[:, :, k], arr[:, :, k + 1])
            v[:, :, k, 0] = u * cx / DT
            v[:, :, k, 1] = w * cy / DT
        v[:, :, K - 1] = v[:, :, K - 2]
        np.clip(v, -VCAP, VCAP, out=v)
        return v

    def centroid_heading(self, P, scale, obs_yaw):
        """Per-step box-orientation heading from the SMOOTH centroid path, broadcast
        to (S,S,K). Heading = direction of the Savitzky-Golay-smoothed centroid
        velocity E[pos|k]=sum_g g P(g,k); falls back to the last observed yaw where
        the centroid is near-stationary (<0.5 m/s). This is ~4 deg accurate vs GT
        and temporally consistent, unlike the per-cell optical-flow heading (~28
        deg, frame-to-frame jitter); the predicted densities are unimodal on InD
        so the centroid (mean) is a faithful representative."""
        S, K = self.S, self.K
        gm = self.grid * scale                                 # (G,2) metres (offset-free)
        cen = torch.einsum("gd,gk->kd", gm, P).cpu().numpy()   # (K,2)
        w = min(SG_WIN, K if K % 2 else K - 1)
        if w >= SG_POLY + 2:
            cv = savgol_filter(cen, window_length=w, polyorder=SG_POLY, deriv=1,
                               delta=DT, axis=0)
        else:
            cv = np.gradient(cen, DT, axis=0)
        sp = np.hypot(cv[:, 0], cv[:, 1])
        th = np.arctan2(cv[:, 1], cv[:, 0])
        th = np.where(sp < 0.5, obs_yaw, th).astype(np.float32)  # (K,)
        return np.broadcast_to(th[None, None, :], (S, S, K)).copy()

    def centroid_velocity(self, P, scale):
        """Bulk centroid velocity per frame (K,2) [m/s]: Savitzky-Golay derivative of
        the density mean E[pos|k]. Physical relative-velocity source for the severity
        term -- robust to the per-cell optical-flow front outliers (a <0.1% tail of
        cells at the density's spreading edge that the energy argmax would otherwise
        select, inflating |dv| ~3-7x above the bulk relative speed)."""
        K = self.K
        gm = self.grid * scale
        cen = torch.einsum("gd,gk->kd", gm, P).cpu().numpy()       # (K,2) metres
        method = os.environ.get("RF_VMETHOD", "poly")   # default: global cubic fit of the centroid path
        w = min(int(os.environ.get("RF_SGWIN", SG_WIN)), K if K % 2 else K - 1)
        if method == "poly":          # global low-order polynomial fit -> analytic derivative
            deg = int(os.environ.get("RF_PDEG", "3")); t = np.arange(K) * DT
            cv = np.stack([np.polyval(np.polyder(np.polyfit(t, cen[:, d], deg)), t)
                           for d in range(2)], axis=1)
        elif method == "cdiff":       # savgol-smoothed path + wide central difference
            cs = savgol_filter(cen, window_length=w, polyorder=SG_POLY, axis=0) if w >= SG_POLY + 2 else cen
            m = int(os.environ.get("RF_CSTEN", "5")); cv = np.zeros((K, 2), np.float32)
            for k in range(K):
                lo, hi = max(0, k - m), min(K - 1, k + m)
                cv[k] = (cs[hi] - cs[lo]) / ((hi - lo) * DT + 1e-9)
        else:                          # savgol least-squares derivative (default)
            cv = (savgol_filter(cen, window_length=w, polyorder=SG_POLY, deriv=1, delta=DT, axis=0)
                  if w >= SG_POLY + 2 else np.gradient(cen, DT, axis=0))
        return cv.astype(np.float32)                               # (K,2)

    def box_overlap_oriented(self, P, P_ego_2d, te_field, tj_field,
                             dims_e, dims_j, scale_t, nbin=8):
        S, K = self.S, self.K
        dx, dy = float(scale_t[0]) / S, float(scale_t[1]) / S
        reach = dims_e[0] + dims_j[0]
        kx, ky = max(int(np.ceil(reach / dx)), 1), max(int(np.ceil(reach / dy)), 1)
        ii, jj = np.mgrid[-kx:kx + 1, -ky:ky + 1]
        dxm, dym = ii * dx, jj * dy
        Pj = P.reshape(S, S, K).detach().cpu().numpy()
        edges = np.linspace(-np.pi, np.pi, nbin + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        out = np.zeros((S, S, K), np.float32)
        kern = {}
        for k in range(K):
            pj = Pj[:, :, k]; pe = P_ego_2d[:, :, k]
            if pj.max() < 1e-12 or pe.max() < 1e-12:
                continue
            be_idx = np.clip(np.digitize(te_field[:, :, k], edges) - 1, 0, nbin - 1)
            bj_idx = np.clip(np.digitize(tj_field[:, :, k], edges) - 1, 0, nbin - 1)
            for be in np.unique(be_idx[pe > pe.max() * 1e-3]):
                emask = (be_idx == be)
                for bj in np.unique(bj_idx[pj > pj.max() * 1e-3]):
                    pj_b = pj * (bj_idx == bj)
                    if pj_b.sum() < 1e-12:
                        continue
                    key = (int(be), int(bj))
                    msk = kern.get(key)
                    if msk is None:
                        msk = _sat_mask(dxm, dym, centers[be], dims_e[0], dims_e[1],
                                        centers[bj], dims_j[0], dims_j[1]).astype(np.float32)
                        kern[key] = msk
                    out[:, :, k] += emask * ndi_convolve(pj_b, msk, mode="constant")
        return out * P_ego_2d

    def _obs_speed(self, xw, a, scale):
        """Observed speed (m/s) of agent a over its last (up to 10) valid history
        frames -- used to flag static agents. scale maps normalized->metres; the
        position offset cancels in the difference."""
        p = xw[0, a]
        valid = ~torch.isnan(p[:, 0])
        p = p[valid]
        if p.shape[0] < 2:
            return 0.0
        n = min(10, p.shape[0] - 1)
        d = (p[-1] - p[-1 - n]) * scale
        return float(torch.linalg.norm(d)) / (n * DT)

    @staticmethod
    def load_scene_dims(path):
        """Sidecar of recorded per-agent dims keyed by test-scene index:
        {scene: (N,2) [L,W] m, NaN = unmatched}. Built by scene_dims_sidecar.py.
        NOTE: bind the arrays ONCE -- each NpzFile[...] access decompresses a
        fresh full copy, and storing views of per-access copies pins one full
        array per scene (~630 GB for InD's 71k scenes -> OOM)."""
        z = np.load(path, allow_pickle=True)
        sidx = z["sidx"]; dims = z["dims"]
        return {int(s): dims[i] for i, s in enumerate(sidx)}

    def _static_kinematics(self, S, K, obs_yaw):
        """Zero velocity field + constant observed-yaw heading for a static agent."""
        return (np.zeros((S, S, K, 2), np.float32),
                np.broadcast_to(np.float32(obs_yaw), (S, S, K)).copy())

    def centroid_path(self, P, scale):
        """Predicted-position path (K,2) metres = density centroid per step."""
        gm = self.grid * scale
        return torch.einsum("gd,gk->kd", gm, P).cpu().numpy()

    def maneuver_shift(self, P, scale, s):
        """Recompute the EGO's own predicted occupancy under a speed-scaled maneuver.
        Driving the SAME predicted route at s x the baseline speed puts the ego at
        mu_0 + s*(mu_k - mu_0) at frame k, so we translate each frame's density by
        (s-1)*(mu_k - mu_0). Brake (s<1) holds it back; accelerate (s>1) carries it
        forward, shifting the meeting geometry. Position scales by s and (its time
        derivative) speed scales by s -- consistent with the v_ego_scale severity
        term, no double counting (shift = geometry channel, scale = severity channel).
        Pure kinematics (distance = speed x time); the joint stays conditioned on the
        factual ego, so surrounding agents do NOT react (cf. the negative result)."""
        S, K, G = self.S, self.K, self.G
        cen = self.centroid_path(P, scale)                       # (K,2) metres
        disp = (float(s) - 1.0) * (cen - cen[0:1])               # (K,2) metres per frame
        dg = 0.9 / (S - 1)                                       # grid spacing (linspace 0.05..0.95)
        sca = scale.detach().cpu().numpy()                       # (2,) metres span
        P2d = P.reshape(S, S, K).detach().cpu().numpy()
        out = np.empty_like(P2d)
        for k in range(K):
            di = disp[k, 0] / (sca[0] * dg)                      # cells: axis-0 <-> x (grid dim 0)
            dj = disp[k, 1] / (sca[1] * dg)                      # cells: axis-1 <-> y (grid dim 1)
            out[:, :, k] = ndi_shift(P2d[:, :, k], (di, dj), order=1, mode="constant", cval=0.0)
        outf = out.reshape(G, K)
        outf = outf / np.clip(outf.sum(0, keepdims=True), 1e-9, None)   # renormalize edge-clipped mass
        return torch.tensor(outf, device=P.device, dtype=P.dtype)

    # ---- full field for one scene window -------------------------------
    def _mc_resample(self, P, n):
        """Monte-Carlo occupancy (trajectory-prediction-style counterpart of the exact
        density): draw n samples/frame from the predicted density P (G,K) and KDE-rebin
        -> empirical occupancy (G,K). Captures finite-sample noise + discretization the
        exact flow density avoids. Equivalent to per-frame marginals of n sampled tracks."""
        S, K, G = self.S, self.K, self.G
        Pn = P.detach().cpu().numpy().astype(np.float64)        # (G,K)
        rng = np.random.default_rng(0)
        out = np.zeros((S, S, K), np.float32)
        for k in range(K):
            pk = Pn[:, k]; s = pk.sum()
            if s <= 0:
                continue
            idx = rng.choice(G, size=n, p=pk / s)
            gi, gj = np.unravel_index(idx, (S, S))
            cnt = np.zeros((S, S), np.float32)
            np.add.at(cnt, (gi, gj), 1.0)
            out[:, :, k] = gaussian_filter(cnt, MC_SIGMA)
        outf = out.reshape(G, K)
        outf = outf / np.clip(outf.sum(0, keepdims=True), 1e-9, None)
        return torch.tensor(outf, device=self.dev, dtype=P.dtype)

    def field(self, xw, fw, vt, contributors, loc_t, scale, return_dens=False,
              return_mp=False, return_diag=False, v_ego_scale=1.0, dims_m=None, mc_n=0,
              return_raw_rf=False):
        """xw,fw: (1,N,Th,*) window; returns rf (S,S,K) [, dens dict] [, mp].
        rf = expected collision energy (J); mp = meeting-probability field
        (sum_j P_ego*(P_j*1_box), no severity) -- the detection-appropriate score.
        v_ego_scale scales the ego speed for the severity term, modelling a candidate
        ego maneuver (brake<1, accelerate>1) for the counterfactual objective J(A).
        dims_m: optional per-agent RECORDED footprint, {a: (L,W) m} or (N,2) array
        (NaN = unknown) -- replaces the class-default VEH_LW boxes in the meeting-
        probability convolution (defaults oversize e.g. AD4CHE trucks 12 m vs the
        recorded ~7.7 m, inflating overlap for queued/passing traffic).
        Severity (RF_SEV): "closing" (default) uses the CLOSING component of the
        relative velocity along the ego->agent line-of-sight, so a parallel
        side-pass carries ~zero severity; "mag" restores the legacy |dv|."""
        S, K = self.S, self.K
        learned = os.environ.get("RF_VMETHOD") == "learned"   # severity v from the learned head
        sev_mode = os.environ.get("RF_SEV", "closing")

        def _hd(a):
            if dims_m is not None:
                d = dims_m.get(a) if hasattr(dims_m, "get") else (
                    dims_m[a] if a < len(dims_m) else None)
                if d is not None and np.isfinite(float(d[0])) and float(d[0]) > 0.5:
                    return (float(d[0]) / 2.0, float(d[1]) / 2.0)
            return tuple(x / 2 for x in VEH_LW.get(agent_class(vt, a), (4.5, 1.9)))

        P_ego = self.ego_density(xw, fw, vt, 0, loc_t)
        if mc_n > 0:
            P_ego = self._mc_resample(P_ego, mc_n)           # trajectory-sample occupancy
        # Candidate ego maneuver: recompute the ego's OWN occupancy under the speed
        # scale (geometry channel), so braking/accelerating shifts WHERE the ego is
        # and hence the meeting geometry. P_ego (factual) still conditions the joint
        # and feeds the severity velocity (scaled by v_ego_scale) -- no double count.
        P_ego_man = (self.maneuver_shift(P_ego, scale, v_ego_scale)
                     if v_ego_scale != 1.0 else P_ego)
        P_ego_2d = P_ego_man.reshape(S, S, K).cpu().numpy()
        dims_e = _hd(0)
        mu_e = self.centroid_path(P_ego_man, scale)          # (K,2) ego predicted mean (maneuvered)
        def _vfull_centroid(P):                              # (K,2) bulk -> (S,S,K,2)
            vc = self.centroid_velocity(P, scale)
            return np.broadcast_to(vc[None, None], (S, S, K, 2)).astype(np.float32)
        obs_yaw_e = float(fw[0, 0, -1, 0]) * 2 * np.pi
        if self._obs_speed(xw, 0, scale) < V_STATIC:
            te_field = np.broadcast_to(np.float32(obs_yaw_e), (S, S, K)).copy()
            v_ego_full = np.zeros((S, S, K, 2), np.float32)      # static -> no motion
        else:
            te_field = self.centroid_heading(P_ego_man, scale, obs_yaw_e)
            v_ego_full = (self.ego_velocity_field(xw, fw, vt, 0, loc_t, scale)
                          if learned else _vfull_centroid(P_ego))   # baseline speed; x v_ego_scale below
        if learned:
            Pj, Vj = self.joint_densities(xw, fw, vt, P_ego, contributors, loc_t, scale=scale)
        else:
            Pj = self.joint_densities(xw, fw, vt, P_ego, contributors, loc_t); Vj = {}
        rf = np.zeros((S, S, K), dtype=np.float32)
        mp = np.zeros((S, S, K), dtype=np.float32)
        dens = {0: P_ego_2d}
        diag = []
        for j in contributors:
            if torch.isnan(xw[0, j]).any() or j not in Pj:
                continue
            P = Pj[j]
            if mc_n > 0:
                P = self._mc_resample(P, mc_n)               # trajectory-sample occupancy
            if return_dens:
                dens[j] = P.reshape(S, S, K).cpu().numpy()
            dims_j = _hd(j)
            obs_yaw_j = float(fw[0, j, -1, 0]) * 2 * np.pi
            if self._obs_speed(xw, j, scale) < V_STATIC:         # static -> no spurious heading
                tj_field = np.broadcast_to(np.float32(obs_yaw_j), (S, S, K)).copy()
                v_j_full = np.zeros((S, S, K, 2), np.float32)
            else:
                tj_field = self.centroid_heading(P, scale, obs_yaw_j)
                v_j_full = Vj[j] if learned else _vfull_centroid(P)
            pcoll = self.box_overlap_oriented(P, P_ego_2d, te_field, tj_field,
                                              dims_e, dims_j, scale)
            mp += pcoll
            dv = v_ego_full * v_ego_scale - v_j_full         # (S,S,K,2) ego-minus-j rel. velocity
            if sev_mode == "closing":
                # closing speed along the ego->agent line of sight: parallel
                # side-passes carry ~zero severity; rear-end/head-on keep full.
                mu_j = self.centroid_path(P, scale)          # (K,2)
                u = mu_j - mu_e
                n = np.linalg.norm(u, axis=-1, keepdims=True)
                u = (u / np.clip(n, 1e-6, None)).astype(np.float32)
                closing = np.clip((dv * u[None, None]).sum(-1), 0.0, None)   # (S,S,K)
                Cfield = (0.5 * M_R * closing ** 2).astype(np.float32)
            else:
                Cfield = (0.5 * M_R * (dv ** 2).sum(-1)).astype(np.float32)
            contrib_rf = pcoll * Cfield
            rf += contrib_rf
            if return_diag and float(contrib_rf.max()) > 0:
                gi, gj, gk = np.unravel_index(int(np.argmax(contrib_rf)), contrib_rf.shape)
                pe = float(P_ego_2d[gi, gj, gk]); pc = float(pcoll[gi, gj, gk])
                dv_diag = (float(closing[gi, gj, gk]) if sev_mode == "closing"
                           else float(np.sqrt(float((dv[gi, gj, gk] ** 2).sum()))))
                diag.append(dict(
                    j=int(j), cls=int(agent_class(vt, j)),
                    v_obs=float(self._obs_speed(xw, j, scale)),
                    cell=(int(gi), int(gj), int(gk)),
                    P_ego=pe, overlap=(pc / pe if pe > 1e-12 else 0.0), M=pc,
                    dv=dv_diag,
                    vego=(float(v_ego_full[gi, gj, gk, 0]), float(v_ego_full[gi, gj, gk, 1])),
                    vj=(float(v_j_full[gi, gj, gk, 0]), float(v_j_full[gi, gj, gk, 1])),
                    C=float(Cfield[gi, gj, gk]), energy=float(contrib_rf[gi, gj, gk])))
            if os.environ.get("RF_DEBUG"):
                vobs = self._obs_speed(xw, j, scale)
                print(f"DBG j={j} cls={agent_class(vt, j)} vobs={vobs:.2f} "
                      f"static={vobs < V_STATIC} pcoll_max={float(pcoll.max()):.3g} "
                      f"E_max={float(contrib_rf.max()):.1f}J yaw={np.degrees(obs_yaw_j) % 360:.0f}deg "
                      f"dims={dims_j}")
        rf_raw = rf.copy() if return_raw_rf else None          # un-dilated rf, for the dilation-sigma sweep
        for k in range(K):
            rf[:, :, k] = gaussian_filter(rf[:, :, k], DILATE_SIGMA)
        out = (rf,)
        if return_dens: out = out + (dens,)
        if return_mp:   out = out + (mp,)
        if return_diag: out = out + (diag,)
        if return_raw_rf: out = out + (rf_raw,)
        return out[0] if len(out) == 1 else out
