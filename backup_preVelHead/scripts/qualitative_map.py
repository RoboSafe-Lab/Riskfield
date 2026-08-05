"""Render the qualitative panels ON the real aerial map (like animate_joint.py):
per scene -> predicted occupancy clouds, optical-flow velocity quiver, and the
risk-field energy curve, all at the highest-risk frame. Scenes via RF_SCENE_IDS."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datasets.registry import get_dataset
from model.RiskFlow import RiskFlow
from riskflow_config import default_dict
from scripts.joint_field import JointRiskField, V_STATIC

reg = get_dataset(); c = default_dict(); dev = "cuda" if torch.cuda.is_available() else "cpu"
S = int(os.environ.get("RF_GRID", "64")); K = c["seq_len"]; DT = float(os.environ.get("RF_DT", "0.08"))
SCALE_DOWN = 12.0; MIN_HIST = 30; bf = reg["boundaries_for_location"]; VEH_LW = reg["veh_lw"]
OUT = os.environ.get("RF_OUTDIR", "qual_map"); os.makedirs(OUT, exist_ok=True)
IDS = [int(s) for s in os.environ.get("RF_SCENE_IDS", "1500,70560").split(",")]
TAGS = os.environ.get("RF_TAGS", "crit,ncrit").split(","); targets = dict(zip(IDS, TAGS))
AG_PAL = [(0.0, 0.85, 1.0), (1.0, 0.55, 0.0), (0.25, 0.95, 0.35), (0.95, 0.3, 0.9),
          (1.0, 0.9, 0.2), (0.6, 0.6, 1.0), (1.0, 0.45, 0.45), (0.4, 1.0, 0.85)]
plt.rcParams.update({"font.size": 12})
ind = reg["LoaderClass"](root=reg["root"], max_samples=c["maximum_samples"], train_ratio=c["train_ratio"],
    train_batch_size=c["train_batch_size"], test_batch_size=1, missing_rate=c["masked_data_ratio"],
    max_num_cars=c["max_num_cars"], max_empty_frames=c["max_empty_frames"], seq_len=c["seq_len"],
    moving_window=c["seq_len"] * 2, sampling_step=c["sampling_step"], should_shuffle=False, include_future=c["include_future"])
site = ind.observation_site_by_scope("all")


def build(sl):
    return RiskFlow(seq_len=c["seq_len"], input_dim=c["input_dim"], feature_dim=c["feature_dim"],
        embedding_dim=c["embedding_dim"], hidden_dim=c["hidden_dim"], max_num_cars=c["max_num_cars"],
        num_classes=c["num_classes"], gru_layers=c["gru_layers"], num_heads=c["num_heads"], dropout=c["dropout"],
        norm_rotation=c["norm_rotate"], flow_layers=c["flow_layers"], flow_hidden_dim=c["flow_hidden_dim"],
        coupling_layers=c["coupling_layers"], use_cnf=c["use_cnf"], use_cgmm=c["use_cgmm"], gmm_modes=c["gmm_modes"],
        use_world_model=True, wm_state_dim=c["wm_state_dim"], action_dim=c["action_dim"], scene_level=sl,
        use_map=True, map_size=c["map_size"], map_data_dir=reg["map_data_dir"], map_dataset=reg["map_dataset"]).to(dev).eval()


ckpt_ego = os.environ.get("RF_CKPT_EGO", "serialized/riskflow_ind_8.pt")
ckpt_joint = os.environ.get("RF_CKPT_JOINT", "serialized/riskflow_ind_7.pt")
me = build(False); me.load_state_dict(torch.load(ckpt_ego, map_location=dev), strict=False)
mj = build(True);  mj.load_state_dict(torch.load(ckpt_joint, map_location=dev), strict=False)
g1 = torch.linspace(0.05, 0.95, S); GX, GY = torch.meshgrid(g1, g1, indexing="ij")
grid = torch.stack([GX.reshape(-1), GY.reshape(-1)], -1).to(dev)
eng = JointRiskField(me, mj, grid, S, K, dev, min_hist=MIN_HIST)
g1n = g1.numpy()
L = np.load("conflict_labels.npz", allow_pickle=True)["data"]; rows = {int(r[0]): r for r in L}


def box_xy(cx, cy, th, Ln, Wn):
    hl, hw = Ln / 2, Wn / 2; cs, sn = np.cos(th), np.sin(th)
    cor = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    rot = np.stack([cor[:, 0] * cs - cor[:, 1] * sn, cor[:, 0] * sn + cor[:, 1] * cs], 1)
    return rot + np.array([cx, cy])


def dens_rgba(P2d, rgb, maxa=0.6):
    m = float(P2d.max()); d = (P2d / m) ** 0.55 if m > 0 else P2d
    img = np.zeros((*P2d.shape, 4), np.float32); img[..., 0], img[..., 1], img[..., 2] = rgb
    img[..., 3] = np.clip(d, 0, 1) * maxa; return img


def load_bg(loc):
    if reg["name"] == "ad4che":               # per-scene map, centre-origin registration
        from datasets.AD4CHE import scene_scale
        bg = mpimg.imread(os.path.join(reg["root"], "maps", f"{loc}.jpg"))
        Hp, Wp = bg.shape[0], bg.shape[1]; sm = scene_scale(reg["root"], loc)
        return bg, [-Wp / 2 * sm, Wp / 2 * sm, -Hp / 2 * sm, Hp / 2 * sm]
    rec = reg["LoaderClass"].LOCATION_RECORDINGS[loc][0]
    o = float(pd.read_csv(f"data/{rec}_recordingMeta.csv").at[0, "orthoPxToMeter"]) * SCALE_DOWN
    bg = mpimg.imread(f"data/{rec}_background.png")
    return bg, [0, bg.shape[1] * o, -bg.shape[0] * o, 0]


scenes = {}
for i, b in enumerate(site.test_loader):
    if i not in targets:
        continue
    x = b["input"].to(dev); feat = b["feature"].to(dev); vt = b["type"].to(dev); fut = b["future"].to(dev)
    if torch.isnan(x[:, 0, -2:, :]).any():
        continue
    loc = int(b["locationId"].view(-1)[0])
    # Recover each agent's recorded GT footprint (length,width) WITHOUT touching the
    # cached split: re-derive the same deterministic window via get_specific_sample
    # (neighbour order is distance-sorted, so it matches the cached batch), picking
    # the recording in this location whose ego history matches this sample.
    dimsb = None
    ego_id = int(np.asarray(b["trackId"]).reshape(-1)[0])
    start_frame = int(np.asarray(b["startFrame"]).reshape(-1)[0])
    ego_ref = np.nan_to_num(x[0, 0].cpu().numpy())
    for st in getattr(ind, "LOCATION_RECORDINGS", {}).get(loc, []):
        try:
            ss = ind.get_specific_sample(st, ego_id, start_frame)
        except Exception:
            continue
        if np.allclose(np.nan_to_num(ss["input"][0, 0].cpu().numpy()), ego_ref, atol=1e-3):
            dimsb = np.asarray(ss["dims"][0]); break    # (max_cars,2)=(length,width)
    # field contributors: observed-present + enough history (NO future requirement,
    # so a conflict partner with partial GT future still contributes its risk).
    contrib = [a for a in range(1, x.shape[1]) if not torch.isnan(x[0, a, -1]).any()
               and int((~torch.isnan(x[0, a, :, 0])).sum()) >= MIN_HIST]
    # future-valid agents -> drawable GT boxes/trajectories (ego included).
    valid = [a for a in [0] + contrib if not torch.isnan(fut[0, a]).any()]
    bx = bf(loc); xlo, xhi = float(bx[0, 0]), float(bx[0, 1]); ylo, yhi = float(bx[1, 0]), float(bx[1, 1])
    scale = torch.tensor([xhi - xlo, yhi - ylo], device=dev); LOC_T = torch.tensor([loc], device=dev)
    with torch.no_grad():
        rf, dens, diag = eng.field(x, feat, vt, contrib, LOC_T, scale, return_dens=True, return_diag=True)
    # velocity field: mirror the field engine -- observed-static agents use ZERO
    # velocity (the optical flow of a parked car's broad, spreading density is spurious).
    velf = {}
    for a in dens:
        if float(eng._obs_speed(x, a, scale)) < V_STATIC:
            velf[a] = np.zeros((S, S, K, 2), np.float32)
        else:
            velf[a] = eng.optical_flow_vel(torch.tensor(dens[a].reshape(-1, K), device=dev), scale)
    if os.environ.get("RF_VDIAG"):       # optical-flow vs centroid bulk speed (severity sanity)
        cxm, cym = float(scale[0]) / S, float(scale[1]) / S
        gm = eng.grid * scale
        Pe = torch.tensor(dens[0].reshape(-1, K), device=dev)
        cen = torch.einsum("gd,gk->kd", gm, Pe).cpu().numpy()
        cv = np.gradient(cen, DT, axis=0); csp = np.hypot(cv[:, 0], cv[:, 1])
        ofs = np.sqrt((velf[0] ** 2).sum(-1)); Pe2 = dens[0]
        wsp = (ofs * Pe2).reshape(-1, K).sum(0) / np.clip(Pe2.reshape(-1, K).sum(0), 1e-9, None)
        print(f"VDIAG scene={i} ego_v_obs={float(eng._obs_speed(x, 0, scale)):.2f} "
              f"cell=({cxm:.2f},{cym:.2f})m | centroid_bulk med={np.median(csp):.2f} max={csp.max():.2f} "
              f"| optflow_Pweighted med={np.median(wsp):.2f} max={wsp.max():.2f} "
              f"| frac_cells>=29m/s={float((ofs >= 29).mean()):.3f}", flush=True)
    if os.environ.get("RF_DEBUG"):
        for a in [0] + contrib:
            nfr = int((~torch.isnan(x[0, a, :, 0])).sum()); hn = bool(torch.isnan(x[0, a]).any())
            print(f"AGENT i={i} a={a} histframes={nfr} nan_in_window={hn} "
                  f"in_dens={a in dens} fut_valid={not bool(torch.isnan(fut[0, a]).any())}", flush=True)
    sca = np.array([xhi - xlo, yhi - ylo]); lon = np.array([xlo, ylo])
    spd = [float(eng._obs_speed(x, a, scale)) for a in range(x.shape[1])]        # observed speed (m/s)
    scenes[i] = dict(tag=targets[i], rf=rf, dens=dens, velf=velf, loc=loc, xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi,
        valid=valid, contrib=contrib, fut=fut[0].cpu().numpy(), sca=sca, lon=lon, diag=diag, spd=spd,
        vt=[int(vt[0, a].reshape(-1)[0]) for a in range(x.shape[1])],
        yaw=[float(feat[0, a, -1, 0]) * 2 * np.pi for a in range(x.shape[1])],   # recorded heading
        dim=[(float(dimsb[a, 0]), float(dimsb[a, 1]))                            # recorded GT (length,width)
             if (dimsb is not None and float(dimsb[a, 0]) > 0.5) else None
             for a in range(x.shape[1])])
    # risk decomposition (Risk = P_ego . M . C) for the analysis table.
    print(f"ANALYSIS scene={i} tag={targets[i]} loc={loc} peakE={float(rf.max()):.0f}J "
          f"ego_v={spd[0]:.2f} ego_cls={scenes[i]['vt'][0]}", flush=True)
    for d in sorted(diag, key=lambda z: z["energy"], reverse=True):
        ve, vj = d.get("vego", (0, 0)), d.get("vj", (0, 0))
        me_, mj = float(np.hypot(*ve)), float(np.hypot(*vj))
        ang = float(np.degrees(np.arccos(np.clip(
            (ve[0] * vj[0] + ve[1] * vj[1]) / (me_ * mj + 1e-9), -1, 1))))
        print(f"  DECOMP j={d['j']} cls={d['cls']} v_obs={d['v_obs']:.2f} "
              f"P_ego={d['P_ego']:.3g} overlap={d['overlap']:.3g} M={d['M']:.3g} "
              f"dv={d['dv']:.2f} C={d['C']:.0f}J energy={d['energy']:.0f}J | "
              f"vego=({ve[0]:.2f},{ve[1]:.2f})|{me_:.2f} vj=({vj[0]:.2f},{vj[1]:.2f})|{mj:.2f} angle={ang:.0f}deg", flush=True)
    if os.environ.get("RF_VPROF") and diag:    # predicted centroid-speed profile vs GT (smoothness/reliability)
        dj = sorted(diag, key=lambda z: z["energy"], reverse=True)[0]["j"]
        sve = np.hypot(*eng.centroid_velocity(torch.tensor(dens[0].reshape(-1, K), device=dev), scale).T)
        print(f"  VPROF ego pred-speed/k: {np.round(sve, 1).tolist()}", flush=True)
        if dj in dens:
            svj = np.hypot(*eng.centroid_velocity(torch.tensor(dens[dj].reshape(-1, K), device=dev), scale).T)
            gtj = fut[0, dj].cpu().numpy() * sca + lon
            sgj = np.hypot(*np.gradient(gtj, DT, axis=0).T)
            print(f"  VPROF j{dj} pred-speed/k: {np.round(svj, 1).tolist()}", flush=True)
            print(f"  VPROF j{dj} GT-speed/k:   {np.round(sgj, 1).tolist()}", flush=True)
    if len(scenes) == len(targets):
        break

EMAX = max(float(s["rf"].reshape(-1, K).max(0).max()) for s in scenes.values())
for i, sc in scenes.items():
    tag = sc["tag"]; rf = sc["rf"]; dens = sc["dens"]; velf = sc["velf"]
    xlo, xhi, ylo, yhi = sc["xlo"], sc["xhi"], sc["ylo"], sc["yhi"]; sca, lon = sc["sca"], sc["lon"]
    valid, contrib = sc["valid"], sc["contrib"]; fut = sc["fut"]
    EGO_C, OTHER_C = AG_PAL[0], AG_PAL[1]                     # ego one colour, all others share one
    AG = {a: (EGO_C if a == 0 else OTHER_C) for a in [0] + contrib}
    cloud = [a for a in [0] + contrib if a in dens]          # agents with a predicted density
    drawn = [a for a in valid if a in dens]                  # only draw modelled agents (skip partially-observed)
    # genuine conflict -> peak-energy frame; zero-risk scene -> short horizon (the
    # argmax of ~0 energy lands at a far frame where densities are diffuse AND the
    # optical-flow velocity is unreliable).
    ks = (int(rf.reshape(-1, K).max(0).argmax()) if float(rf.reshape(-1, K).max()) > 5.0
          else min(K // 6, K - 1))
    if os.environ.get("RF_DEBUG"):
        cx_, cy_ = sca[0] / S, sca[1] / S
        for a in drawn:
            tr_ = fut[a] * sca + lon; v0 = float(np.linalg.norm(np.gradient(tr_, DT, axis=0)[0]))
            for kk in (8, 25, 47):
                P2 = dens[a][:, :, kk]; nc = int((P2 > 0.1 * P2.max()).sum())
                print(f"SPREAD i={i} a={a} v0={v0:.2f} k={kk} ncells>10%={nc} "
                      f"radius={(nc * cx_ * cy_ / 3.14159) ** 0.5:.1f}m", flush=True)
    bg, bge = load_bg(sc["loc"])
    pts = (fut[drawn].reshape(-1, 2)) * sca + lon; pts = pts[~np.isnan(pts[:, 0])]
    AR = 1.35
    if reg["name"] == "ad4che":     # fixed window -> every panel identical: full map
        half_h = (bge[3] - bge[2]) / 2.0        # height (AD4CHE maps are all 144x81 m) x
        half_w = half_h * AR                    # AR-width, x-centred on the agents so the
        cym = (bge[2] + bge[3]) / 2.0           # bar spans the full (shared) panel height.
        ccx = float(np.clip(np.nanmean(pts[:, 0]), bge[0] + half_w, bge[1] - half_w))
        cx0, cx1 = ccx - half_w, ccx + half_w; cy0, cy1 = cym - half_h, cym + half_h
    else:                           # InD: adaptive crop tight around the agents
        pad = 12.0
        cx0, cx1 = pts[:, 0].min() - pad, pts[:, 0].max() + pad
        cy0, cy1 = pts[:, 1].min() - pad, pts[:, 1].max() + pad
        cx0, cx1 = max(cx0, bge[0]), min(cx1, bge[1]); cy0, cy1 = max(cy0, bge[2]), min(cy1, bge[3])
        w, h = cx1 - cx0, cy1 - cy0
        if w / h < AR:
            e = (AR * h - w) / 2; cx0, cx1 = cx0 - e, cx1 + e
        else:
            e = (w / AR - h) / 2; cy0, cy1 = cy0 - e, cy1 + e
        cx0, cx1 = max(cx0, bge[0]), min(cx1, bge[1]); cy0, cy1 = max(cy0, bge[2]), min(cy1, bge[3])
    Xg = g1n * sca[0] + xlo; Yg = g1n * sca[1] + ylo; MX, MY = np.meshgrid(Xg, Yg, indexing="ij")

    def boxes(ax):
        for a in drawn:
            tr = fut[a] * sca + lon
            vel = np.gradient(tr, DT, axis=0)                    # m/s
            kf = 0                                               # draw the vehicle at the present
            # (forecast origin); the predicted density/risk are shown ahead at k*.
            # Always use the GT recorded orientation -- velocity-derived heading is
            # noisy/wrong for slow vehicles and mis-orients long (truck) boxes.
            th = sc["yaw"][a]
            # true per-vehicle GT footprint when available; else the class default.
            dims = sc["dim"][a] or VEH_LW.get(sc["vt"][a], (4.5, 1.9)); p = tr[kf]
            ax.plot(tr[:, 0], tr[:, 1], "-", color=AG[a], lw=1.0, alpha=0.65, zorder=3)
            ax.add_patch(Polygon(box_xy(p[0], p[1], th, *dims), closed=True, fill=False,
                edgecolor=AG[a], lw=1.8, zorder=6))
            ax.plot([p[0]], [p[1]], ("o" if a == 0 else "s"), color=AG[a], ms=(8 if a == 0 else 5),
                mec="white", zorder=7)
            if os.environ.get("RF_LABELS_ID"):       # mark agent id + observed speed
                lbl = f"{'E' if a == 0 else a}: {sc['spd'][a]:.0f}"
                ax.annotate(lbl, (p[0], p[1]), textcoords="offset points", xytext=(6, 5),
                    fontsize=7.5, color="white", weight="bold", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.12", fc=AG[a], ec="none", alpha=0.85))

    # position density panel
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(bg, extent=bge, origin="upper", zorder=0)
    for a in cloud:
        ax.imshow(dens_rgba(dens[a][:, :, ks].T, AG[a]), extent=[xlo, xhi, ylo, yhi], origin="lower", zorder=2)
    boxes(ax)
    ax.set_xlim(cx0, cx1); ax.set_ylim(cy0, cy1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(f"{OUT}/qual_map_{tag}_pos.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # velocity panel: one per-agent centroid (bulk) velocity arrow -- the relative
    # velocity that feeds the severity (Delta v = v_ego - v_j); static agents -> none.
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(bg, extent=bge, origin="upper", zorder=0)
    qx, qy, qu, qv, qs = [], [], [], [], []
    for a in cloud:
        if float(sc["spd"][a]) < V_STATIC:
            vvec = np.zeros(2, np.float32)
        else:
            cvk = eng.centroid_velocity(torch.tensor(dens[a].reshape(-1, K), device=dev), scale)
            vvec = np.asarray(cvk[ks], np.float32)
        p = fut[a][0] * sca + lon
        qx.append(float(p[0])); qy.append(float(p[1])); qu.append(float(vvec[0])); qv.append(float(vvec[1]))
        qs.append(float(np.hypot(vvec[0], vvec[1])))
    qs = np.array(qs)
    Q = ax.quiver(qx, qy, qu, qv, qs, cmap="viridis", zorder=5, angles="xy",
        scale_units="xy", scale=0.7, width=0.013, clim=(0, max(qs.max() if len(qs) else 1, 1)))
    boxes(ax)
    ax.set_xlim(cx0, cx1); ax.set_ylim(cy0, cy1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.08)   # colorbar matches image height
    cb = fig.colorbar(Q, cax=cax); cb.set_label("bulk speed [m/s]", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/qual_map_{tag}_vel.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # energy curve
    ep = rf.reshape(-1, K).max(0); t = np.arange(K) * DT; r = rows.get(i); onset = int(r[3]) if r is not None else -1
    emax_s = max(float(ep.max()), 1e-6)          # per-scene axis (each conflict on its own scale)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(t, ep, "-o", color="crimson", ms=3, lw=1.8); ax.fill_between(t, 0, ep, color="crimson", alpha=0.15)
    ax.set_ylim(0, emax_s * 1.08)
    if 0 <= onset < K:
        ax.axvline(onset * DT, ls="--", color="k", lw=1.2, alpha=0.7)
        ax.text(onset * DT, emax_s * 0.99, " conflict\n onset", fontsize=10, va="top")
    ax.text(0.96, 0.96, f"peak $={ep.max():.1f}$ J", transform=ax.transAxes, ha="right", va="top", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.85))
    ax.set_xlabel("time [s]"); ax.set_ylabel("peak risk energy [J]"); ax.margins(x=0.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/qual_map_{tag}_energy.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # risk-field heat overlay (inferno) on the map -> shows WHERE the criticality is
    cmapI = matplotlib.colormaps["inferno"]
    fixed_vmax = float(os.environ["RF_VMAX"]) if os.environ.get("RF_VMAX") else None
    vmax = fixed_vmax if fixed_vmax else max(float(ep.max()), 1e-6)

    def rf_rgba(fr):
        dd = np.clip(fr / vmax, 0, 1) ** 0.55; img = cmapI(dd)
        img[..., 3] = np.clip(dd * 1.3, 0, 0.72); return img
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(bg, extent=bge, origin="upper", zorder=0)
    ax.imshow(rf_rgba(rf[:, :, ks].T), extent=[xlo, xhi, ylo, yhi], origin="lower", zorder=3)
    boxes(ax)
    ax.set_xlim(cx0, cx1); ax.set_ylim(cy0, cy1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if not os.environ.get("RF_NOCBAR"):       # RF_NOCBAR=1 -> omit per-panel bar (for a shared one)
        smap = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, vmax), cmap=cmapI)
        rcax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.08)   # colorbar matches image height
        cb = fig.colorbar(smap, cax=rcax); cb.set_label("expected collision energy [J]", fontsize=10)
        if fixed_vmax:    # T_critical colorbar (top = critical), matching the reference style
            cb.set_ticks([0, 0.5 * vmax, vmax]); cb.set_ticklabels(["0", f"{0.5 * vmax:.0f}", f"CRITICAL\n{vmax:.0f} J"])
    fig.tight_layout(); fig.savefig(f"{OUT}/qual_map_{tag}_risk.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"RESULT {tag} scene={i} loc={sc['loc']} kstar={ks} peakE={ep.max():.1f}J nagents={len(valid)}")
print("done")
