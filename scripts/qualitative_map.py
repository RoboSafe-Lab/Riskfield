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
SDIMS = (JointRiskField.load_scene_dims(os.environ["RF_DIMS"])
         if os.environ.get("RF_DIMS") else None)   # recorded per-agent dims sidecar

reg = get_dataset(); c = default_dict(); dev = "cuda" if torch.cuda.is_available() else "cpu"
S = int(os.environ.get("RF_GRID", "64")); K = c["seq_len"]; DT = float(os.environ.get("RF_DT", "0.08"))
SCALE_DOWN = float(reg.get("bg_scale_down", 12.0)); MIN_HIST = 30; bf = reg["boundaries_for_location"]; VEH_LW = reg["veh_lw"]
OUT = os.environ.get("RF_OUTDIR", "qual_map"); os.makedirs(OUT, exist_ok=True)
# AD4CHE: render the clean OpenDRIVE lane map as the panel background (the aerial
# photo is too cluttered to read the close-pass conflicts). RF_MAPBG=0 forces the
# photo. The .xodr lives in the same metric, centre-origin frame as the tracks, so
# the world->panel transform is identity (validated against recorded track points).
USE_MAP = reg["name"] == "ad4che" and os.environ.get("RF_MAPBG", "1") != "0"
XODR_DIR = os.environ.get("RF_XODR_DIR", os.path.join(reg["root"], "maps", "opendrive014-024"))
if USE_MAP:
    from datasets.map_renderer import render_road_background
IDS = [int(s) for s in os.environ.get("RF_SCENE_IDS", "1500,70560").split(",")]
TAGS = os.environ.get("RF_TAGS", "crit,ncrit").split(","); targets = dict(zip(IDS, TAGS))
AG_PAL = [(0.0, 0.85, 1.0), (1.0, 0.55, 0.0), (0.25, 0.95, 0.35), (0.95, 0.3, 0.9),
          (1.0, 0.9, 0.2), (0.6, 0.6, 1.0), (1.0, 0.45, 0.45), (0.4, 1.0, 0.85)]
# Panels are shrunk ~4x (height=2.9cm) in the paper, so render with large fonts
# to keep axis labels/ticks legible (esp. the energy-curve panels d/h).
plt.rcParams.update({"font.size": 20, "axes.labelsize": 20, "axes.titlesize": 18,
                     "xtick.labelsize": 17, "ytick.labelsize": 17, "legend.fontsize": 17})
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
        use_map=True, map_size=c["map_size"], map_data_dir=reg["map_data_dir"], map_dataset=reg["map_dataset"],
        map_local=os.environ.get("RF_MAP_LOCAL", "0").lower() in ("1", "true"),
        map_crop_m=c.get("map_crop_m", 40.0), map_raster_res=c.get("map_raster_res", 192)).to(dev).eval()


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


def keep_cbar_labels_inside(fig, cb):
    # the bottom/top colorbar ticks sit at the very edges of the bar (which matches
    # the image height); their labels are va='center', so half the end labels hang
    # outside the image. Anchor the lowest label by its bottom and the highest by its
    # top so every number stays completely within the image height.
    fig.canvas.draw()
    lbls = cb.ax.get_yticklabels()
    if lbls:
        lbls[0].set_verticalalignment("bottom")
        lbls[-1].set_verticalalignment("top")


def load_bg(loc):
    if reg["name"] == "ad4che":               # per-scene map, centre-origin registration
        from datasets.AD4CHE import scene_scale
        bg = mpimg.imread(os.path.join(reg["root"], "maps", f"{loc}.jpg"))
        Hp, Wp = bg.shape[0], bg.shape[1]; sm = scene_scale(reg["root"], loc)
        return bg, [-Wp / 2 * sm, Wp / 2 * sm, -Hp / 2 * sm, Hp / 2 * sm]
    rec = reg["LoaderClass"].LOCATION_RECORDINGS[loc][0]   # InD/rounD: per-recording ortho background
    o = float(pd.read_csv(os.path.join(reg["root"], f"{rec}_recordingMeta.csv"))
              .at[0, "orthoPxToMeter"]) * SCALE_DOWN
    bg = mpimg.imread(os.path.join(reg["root"], f"{rec}_background.png"))
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
        rf, dens, diag = eng.field(x, feat, vt, contrib, LOC_T, scale, return_dens=True, return_diag=True,
                                   dims_m=(SDIMS.get(i) if SDIMS else (dimsb if dimsb is not None else None)), scene_idx=i)
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
    if os.environ.get("RF_SPREAD"):      # occupancy anisotropy + curved-path orientation error
        gm = eng.grid.cpu().numpy() * sca + lon                                   # (G,2) metric grid
        # diagnose the ego (a=0) and the dominant conflict partner (curved neighbour)
        dj0 = (sorted(diag, key=lambda z: z["energy"], reverse=True)[0]["j"]
               if diag else None)
        for a in [0] + ([dj0] if dj0 is not None and dj0 in dens else []):
            Pa = dens[a].reshape(-1, K)
            gta = fut[0, a].cpu().numpy() * sca + lon if not bool(torch.isnan(fut[0, a]).any()) else None
            gva = np.gradient(gta, axis=0) if gta is not None else None           # GT vel/frame (metric)
            for kk in sorted(set([0, K // 2, K - 1])):
                w = Pa[:, kk]; s = float(w.sum())
                if s <= 0:
                    continue
                cx, cy = (w * gm[:, 0]).sum() / s, (w * gm[:, 1]).sum() / s
                dxr, dyr = gm[:, 0] - cx, gm[:, 1] - cy
                sxx = (w * dxr * dxr).sum() / s; syy = (w * dyr * dyr).sum() / s; sxy = (w * dxr * dyr).sum() / s
                Sig = np.array([[sxx, sxy], [sxy, syy]])
                # GT heading at kk -> longitudinal/lateral spread + principal-axis orientation error
                hv = gva[kk] if gva is not None else np.array([1.0, 0.0]); hn = float(np.hypot(*hv))
                h = hv / hn if hn > 1e-6 else np.array([1.0, 0.0]); p = np.array([-h[1], h[0]])
                sL = np.sqrt(max(h @ Sig @ h, 0)); sT = np.sqrt(max(p @ Sig @ p, 0))
                ev, evec = np.linalg.eigh(Sig); pax = evec[:, int(np.argmax(ev))]
                oerr = float(np.degrees(np.arccos(np.clip(abs(pax @ h), 0, 1))))   # 0=aligned, 90=cross
                print(f"SPREAD i={i} a={a} k={kk} sigLong={sL:.2f} sigLat={sT:.2f} "
                      f"sigLat/Long={sT/max(sL,1e-6):.2f} orient_err={oerr:.0f}deg gtspeed={hn/DT:.2f}", flush=True)
        gt = fut[0, 0].cpu().numpy() * sca + lon
        print(f"SPREAD i={i} ego GT net_dlon={gt[-1,0]-gt[0,0]:.2f} net_dlat={gt[-1,1]-gt[0,1]:.2f} "
              f"path_curv_range_lon={gt[:,0].max()-gt[:,0].min():.1f} lat={gt[:,1].max()-gt[:,1].min():.1f}", flush=True)
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
    if USE_MAP:                     # OpenDRIVE lanes drawn live -> clip to the scene box
        bg, bge = None, [xlo, xhi, ylo, yhi]
    else:
        bg, bge = load_bg(sc["loc"])
    pts = (fut[drawn].reshape(-1, 2)) * sca + lon; pts = pts[~np.isnan(pts[:, 0])]
    AR = 1.35
    if reg["name"] == "ad4che":     # AD4CHE (OpenDRIVE map OR aerial photo): zoom to ego + the top conflict partner
        djs = [d["j"] for d in sorted(sc["diag"], key=lambda z: z["energy"], reverse=True)]
        focus = [a for a in ([0] + djs[:1]) if a in drawn] or drawn
        cpts = (fut[focus].reshape(-1, 2)) * sca + lon; cpts = cpts[~np.isnan(cpts[:, 0])]
        pad = float(os.environ.get("RF_ZOOMPAD", "8"))
    else:                           # InD/rounD: adaptive crop around all drawn agents
        cpts = pts; pad = 12.0
    cx0, cx1 = cpts[:, 0].min() - pad, cpts[:, 0].max() + pad
    cy0, cy1 = cpts[:, 1].min() - pad, cpts[:, 1].max() + pad
    cx0, cx1 = max(cx0, bge[0]), min(cx1, bge[1]); cy0, cy1 = max(cy0, bge[2]), min(cy1, bge[3])
    w, h = cx1 - cx0, cy1 - cy0
    if w / h < AR:
        e = (AR * h - w) / 2; cx0, cx1 = cx0 - e, cx1 + e
    else:
        e = (w / AR - h) / 2; cy0, cy1 = cy0 - e, cy1 + e
    cx0, cx1 = max(cx0, bge[0]), min(cx1, bge[1]); cy0, cy1 = max(cy0, bge[2]), min(cy1, bge[3])
    if os.environ.get("RF_SQUARE", "") != "":              # square crop so all panels share one height
        _sq = min(cx1 - cx0, cy1 - cy0)
        _mx = 0.5 * (cx0 + cx1); _my = 0.5 * (cy0 + cy1)
        cx0, cx1 = _mx - _sq / 2.0, _mx + _sq / 2.0
        cy0, cy1 = _my - _sq / 2.0, _my + _sq / 2.0
    Xg = g1n * sca[0] + xlo; Yg = g1n * sca[1] + ylo; MX, MY = np.meshgrid(Xg, Yg, indexing="ij")
    # imshow extent for the density/risk arrays: the grid samples live at
    # normalized 0.05..0.95 (NOT the full [0,1] box). Stretching them over
    # [xlo,xhi] displaces rendered mass outward by up to ~4% of the box
    # (~6 m at the edges) -- the "cloud behind the ego" artifact. Use the
    # sample-centred extent instead.
    _h = (g1n[1] - g1n[0]) / 2.0
    GEXT = [xlo + (g1n[0] - _h) * sca[0], xlo + (g1n[-1] + _h) * sca[0],
            ylo + (g1n[0] - _h) * sca[1], ylo + (g1n[-1] + _h) * sca[1]]

    def draw_bg(ax):                # panel background: live OpenDRIVE lanes or aerial photo
        if USE_MAP:
            ax.set_facecolor("white")
            render_road_background(ax, None, np.eye(4), map_dir=XODR_DIR, margin=10.0,
                viewport=(cx0, cx1, cy0, cy1), map_stem=f"{int(sc['loc']):03d}")
        else:
            ax.imshow(bg, extent=bge, origin="upper", zorder=0)

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
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    draw_bg(ax)
    for a in cloud:
        ax.imshow(dens_rgba(dens[a][:, :, ks].T, AG[a]), extent=GEXT, origin="lower", zorder=2)
    boxes(ax)
    ax.set_xlim(cx0, cx1); ax.set_ylim(cy0, cy1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(pad=0.3); fig.savefig(f"{OUT}/qual_map_{tag}_pos.png", dpi=150, bbox_inches="tight", pad_inches=0.02); plt.close(fig)

    # velocity panel: one per-agent centroid (bulk) velocity arrow -- the relative
    # velocity that feeds the severity (Delta v = v_ego - v_j); static agents -> none.
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    draw_bg(ax)
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
    cb = fig.colorbar(Q, cax=cax); cb.set_label("speed [m/s]", fontsize=16); cb.ax.tick_params(labelsize=13)
    keep_cbar_labels_inside(fig, cb)
    fig.tight_layout(pad=0.3); fig.savefig(f"{OUT}/qual_map_{tag}_vel.png", dpi=150, bbox_inches="tight", pad_inches=0.02); plt.close(fig)

    # energy curve
    ep = rf.reshape(-1, K).max(0); t = np.arange(K) * DT
    emax_s = max(float(ep.max()), 1e-6)          # per-scene axis (each conflict on its own scale)
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(t, ep, "-o", color="crimson", ms=3, lw=1.8); ax.fill_between(t, 0, ep, color="crimson", alpha=0.15)
    ax.set_ylim(0, emax_s * 1.08)
    ax.text(0.96, 0.96, f"peak $={ep.max():.1f}$ J", transform=ax.transAxes, ha="right", va="top", fontsize=16,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.85))
    ax.set_xlabel("time [s]", fontsize=14); ax.set_ylabel("peak risk energy [J]", fontsize=14)
    ax.tick_params(labelsize=13); ax.margins(x=0.02)
    ax.set_box_aspect((cy1 - cy0) / (cx1 - cx0))           # match the map panels' aspect so the row aligns in height
    fig.tight_layout(pad=0.3); fig.savefig(f"{OUT}/qual_map_{tag}_energy.png", dpi=150, bbox_inches="tight", pad_inches=0.02); plt.close(fig)

    # risk-field heat overlay (inferno) on the map. Scale: RF_LOG="vmin,vmax"
    # activates the UNIFIED LOG scale (same energy = same colour across all
    # datasets/figures; spans the ~3 decades between InD and AD4CHE energies);
    # otherwise linear with RF_VMAX or this scene's peak.
    cmapI = matplotlib.colormaps["inferno"]
    fixed_vmax = float(os.environ["RF_VMAX"]) if os.environ.get("RF_VMAX") else None
    vmax = fixed_vmax if fixed_vmax else max(float(ep.max()), 1e-6)
    LOGRNG = ([float(t) for t in os.environ["RF_LOG"].split(",")]
              if os.environ.get("RF_LOG") else None)

    def rf_rgba(fr):
        fr = np.nan_to_num(np.asarray(fr, dtype=np.float32), nan=0.0)  # NaN cell -> transparent
        if LOGRNG:
            lo, hi = LOGRNG
            dd = np.clip((np.log10(np.clip(fr, lo, hi)) - np.log10(lo))
                         / (np.log10(hi) - np.log10(lo)), 0, 1)
            dd[fr < lo] = 0.0
        else:
            dd = np.clip(fr / vmax, 0, 1) ** 0.55
        img = cmapI(dd)
        img[..., 3] = np.clip(dd * 1.3, 0, 0.72); return img

    def _cb_norm():
        return (matplotlib.colors.LogNorm(LOGRNG[0], LOGRNG[1]) if LOGRNG
                else matplotlib.colors.Normalize(0, vmax))
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    draw_bg(ax)
    ax.imshow(rf_rgba(rf[:, :, ks].T), extent=GEXT, origin="lower", zorder=3)
    boxes(ax)
    ax.set_xlim(cx0, cx1); ax.set_ylim(cy0, cy1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if not os.environ.get("RF_NOCBAR"):       # RF_NOCBAR=1 -> omit per-panel bar (for a shared one)
        smap = matplotlib.cm.ScalarMappable(norm=_cb_norm(), cmap=cmapI)
        rcax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.08)   # colorbar matches image height
        cb = fig.colorbar(smap, cax=rcax); cb.set_label("energy [J]", fontsize=16); cb.ax.tick_params(labelsize=13)
        if fixed_vmax and not LOGRNG:
            cb.set_ticks([0, 0.5 * vmax, vmax])   # plain values in J, no criticality marker
        keep_cbar_labels_inside(fig, cb)
    fig.tight_layout(pad=0.3); fig.savefig(f"{OUT}/qual_map_{tag}_risk.png", dpi=150, bbox_inches="tight", pad_inches=0.02); plt.close(fig)

    # risk-evolution strip: the SAME field at three key frames (current -> mid-horizon
    # -> peak), shared T_critical colorbar. The ego is shown as its full predicted
    # occupancy distribution (filled cyan cloud) plus an oriented box at the predicted
    # mean; each SURROUNDING vehicle is an oriented vehicle box (recorded footprint +
    # heading, white halo for contrast over the heat) at ITS predicted-mean position,
    # so vehicles advance across the frames but read as vehicles, not contour blobs.
    # RF_ESTRIP=1 to emit.
    if os.environ.get("RF_ESTRIP"):
        _onset_env = os.environ.get("RF_ESTRIP_ONSET", "")
        if _onset_env != "":                              # early-warning strip
            mark_onset = min(int(_onset_env), K - 1)
            if os.environ.get("RF_ESTRIP_PAIR", "") != "":  # recognition frame | recorded PET | energy curve
                _gf = rf.reshape(-1, K); _gs = int(_gf.max(1).argmax())   # conflict cell = max-risk cell
                _cx, _cy = np.unravel_index(_gs, (rf.shape[0], rf.shape[1]))
                ek_cell = rf[_cx, _cy, :]; _em = float(ek_cell.max())
                krec = int(np.argmax(ek_cell > 0.4 * _em)) if _em > 0 else ks  # field first reaches 40% of its peak
                fr3 = sorted(set([krec, mark_onset]))
                if len(fr3) < 2:
                    fr3 = [krec, min(krec + 2, K - 1)]
            else:                                          # 3-panel: current -> field peak -> PET conflict
                fr3 = sorted(set([0, ks, mark_onset]))
                for extra in (K // 2, ks // 2, K - 1):     # pad to 3 distinct if they collide
                    if len(fr3) >= 3:
                        break
                    fr3 = sorted(set(fr3 + [extra]))
                fr3 = fr3[:3]
        else:
            mark_onset = None
            fr3 = sorted(set([0, K // 2, ks]))            # current, mid-horizon, peak
            for extra in (K - 1, K // 4, K // 3, ks // 2):
                if len(fr3) >= 3:
                    break
                fr3 = sorted(set(fr3 + [extra]))
            fr3 = fr3[:3]

        def pred_mean(P2):                                 # E[pos | k] on the metric grid
            s = float(P2.sum())
            if s <= 0:
                return None
            return float((MX * P2).sum() / s), float((MY * P2).sum() / s)

        PAIR = os.environ.get("RF_ESTRIP_PAIR", "") != "" and mark_onset is not None
        ncol = 3 if PAIR else len(fr3)
        figs, axs = plt.subplots(1, ncol, figsize=(4.0 * ncol + 0.6, 3.5))
        axs = np.atleast_1d(axs)
        if PAIR:
            print(f"CONFCELL scene={i} krec={fr3[0]} ({fr3[0]*DT:.2f}s) kpeak={ks} ({ks*DT:.2f}s) "
                  f"onset={mark_onset} ({mark_onset*DT:.2f}s) lead={(mark_onset-fr3[0])*DT:.2f}s", flush=True)
            _s = min(cx1 - cx0, cy1 - cy0)                  # square the crop so all 3 panels share height
            _mx = 0.5 * (cx0 + cx1); _my = 0.5 * (cy0 + cy1)
            cx0, cx1 = _mx - _s / 2.0, _mx + _s / 2.0
            cy0, cy1 = _my - _s / 2.0, _my + _s / 2.0
        for col, kk in enumerate(fr3):
            axc = axs[col]; draw_bg(axc)
            data_panel = PAIR and (mark_onset is not None) and (kk == mark_onset)
            if data_panel:                                 # panel (b): ORIGINAL recorded scene at the PET frame
                for a in drawn:
                    Pm = fut[a] * sca + lon                 # recorded GT trajectory (metric)
                    axc.plot(Pm[:, 0], Pm[:, 1], "-", color=AG[a], lw=1.0, alpha=0.5, zorder=3)
                    k2 = min(kk + 1, len(Pm) - 1); k1 = max(k2 - 1, 0)
                    dv = Pm[k2] - Pm[k1]
                    th = float(np.arctan2(dv[1], dv[0])) if np.hypot(dv[0], dv[1]) > 0.25 else sc["yaw"][a]
                    dims_a = sc["dim"][a] or VEH_LW.get(sc["vt"][a], (4.5, 1.9))
                    bxy = box_xy(Pm[kk][0], Pm[kk][1], th, *dims_a)
                    axc.add_patch(Polygon(bxy, closed=True, fill=True, facecolor=AG[a],
                        edgecolor="white", lw=1.6, alpha=0.9, zorder=5))
            else:
                axc.imshow(rf_rgba(rf[:, :, kk].T), extent=GEXT, origin="lower", zorder=3)
                for a in cloud:                            # predicted distribution of each vehicle at k
                    P = dens[a][:, :, kk]
                    if float(P.max()) <= 0:
                        continue
                    # FUTURE risk -> position DISTRIBUTIONS only (ego cyan, others
                    # orange); vehicle boxes are factual and appear only at t=0.
                    axc.imshow(dens_rgba(P.T, AG[a], maxa=0.55 if a == 0 else 0.45),
                        extent=GEXT, origin="lower", zorder=4)
                if kk == 0:                                # current frame: recorded vehicle boxes
                    for a in drawn:
                        p0 = fut[a][0] * sca + lon
                        dims_a = sc["dim"][a] or VEH_LW.get(sc["vt"][a], (4.5, 1.9))
                        bxy = box_xy(p0[0], p0[1], sc["yaw"][a], *dims_a)
                        axc.add_patch(Polygon(bxy, closed=True, fill=False,
                            edgecolor="white", lw=2.0, zorder=5))
                        axc.add_patch(Polygon(bxy, closed=True, fill=False,
                            edgecolor=AG[a], lw=1.2, zorder=5.1))
            axc.set_xlim(cx0, cx1); axc.set_ylim(cy0, cy1); axc.set_aspect("equal")
            axc.set_xticks([]); axc.set_yticks([])
            ttl = f"$t = {kk * DT:.2f}$ s"
            if data_panel:
                ttl += "  (recorded PET conflict)"
            elif PAIR:
                ttl += "  (field recognizes)"
            elif mark_onset is not None and kk == mark_onset:
                ttl += "  (PET conflict)"
            elif kk == ks:
                ttl += "  (field peak)"
            axc.set_title(ttl, fontsize=(12 if PAIR else 15))
            if not data_panel:
                axc.text(0.97, 0.05, f"{float(rf[:, :, kk].max()):.0f} J", transform=axc.transAxes,
                    ha="right", va="bottom", fontsize=(11 if PAIR else 14), color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.5))
        if PAIR:                                            # panel (c): risk energy at the conflict cell vs time
            axe = axs[2]; tt = np.arange(K) * DT
            axe.plot(tt, ek_cell, "-", color="#d62728", lw=2.4, zorder=3)
            axe.axvline(fr3[0] * DT, ls="--", color="#0072B2", lw=1.6)     # recognition frame
            axe.axvline(mark_onset * DT, ls="--", color="k", lw=1.6)       # PET conflict
            axe.axvspan(fr3[0] * DT, mark_onset * DT, color="#0072B2", alpha=0.10)
            _ym = float(ek_cell.max()) or 1.0
            axe.text(fr3[0] * DT, _ym * 1.02, "recognizes", color="#0072B2", ha="center", va="bottom", fontsize=8.5)
            axe.text(mark_onset * DT, _ym * 1.02, "PET", color="k", ha="center", va="bottom", fontsize=8.5)
            axe.set_xlim(0, min(K - 1, mark_onset + 5) * DT); axe.set_ylim(0, _ym * 1.18)
            axe.set_xlabel("time [s]", fontsize=11); axe.set_ylabel("risk at conflict cell [J]", fontsize=10)
            axe.set_title("energy at the conflict cell", fontsize=12); axe.grid(alpha=0.25)
            axe.tick_params(labelsize=9)
            axe.set_box_aspect(1.0)                         # square box -> same height as the (square) map panels
        smap = matplotlib.cm.ScalarMappable(norm=_cb_norm(), cmap=cmapI)
        # colorbar height locked to the (aspect-equal) panel image, not the full
        # subplot row -- otherwise the bar is taller than the images.
        _cbax_host = axs[0] if PAIR else axs[-1]
        ccax = make_axes_locatable(_cbax_host).append_axes("right", size="4%", pad=0.08)
        cb = figs.colorbar(smap, cax=ccax)
        cb.set_label("energy [J]", fontsize=(12 if PAIR else 16))
        cb.ax.tick_params(labelsize=(8 if PAIR else 10))
        if fixed_vmax and not LOGRNG:
            cb.set_ticks([0, 0.5 * vmax, vmax])   # plain values in J, no criticality marker
        figs.savefig(f"{OUT}/qual_map_{tag}_estrip.png", dpi=150, bbox_inches="tight"); plt.close(figs)
        print(f"ESTRIP {tag} frames={fr3} times={[round(k*DT,2) for k in fr3]} "
              f"peakcellE={[round(float(rf[:,:,k].max()),1) for k in fr3]}", flush=True)
    print(f"RESULT {tag} scene={i} loc={sc['loc']} kstar={ks} peakE={ep.max():.1f}J nagents={len(valid)}")
print("done")
