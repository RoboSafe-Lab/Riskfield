"""Conflict-detection efficacy comparison on the all-33 InD test split.

Scores each method per scene ONCE (scores are label-independent), saves them, then
evaluates as a binary detector against MULTIPLE conflict-label sets (label-robustness
sweep): L1 (motion, primary), L2 (strict), L3 (TTC-independent min-distance). This
tests whether the kinematic TTC baseline's edge is a label artifact or robust.

Methods (higher=riskier):
  ours      = peak of our two-model expected-energy field (prob x severity).
  ours_prob = peak meeting-probability field (detection-appropriate; no severity).
  pora      = PORA-style cell occupancy-overlap on the same densities.
  ttc, dsf  = kinematic baselines from the present state.

Metrics: AUROC, AP (numpy), early-warning lead time. Output: conflict_eval.json,
scores cached in conflict_scores.npz.

Env: RF_LABEL_FILES (comma-sep npz), RF_GRID (48), RF_STRIDE (20), RF_MAX_SCENES.
"""

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from datasets.registry import get_dataset
_reg = get_dataset()                          # RF_DATASET env (ind|ad4che)
boundaries_for_location = _reg["boundaries_for_location"]
from model.RiskFlow import RiskFlow
from riskflow_config import default_dict
from scripts.joint_field import JointRiskField
SDIMS = (JointRiskField.load_scene_dims(os.environ["RF_DIMS"])
         if os.environ.get("RF_DIMS") else None)   # recorded per-agent dims sidecar
from scripts.baselines import ttc_score, dsf_score, pora_style_score, ours_peak

DT = 0.08
S = int(os.environ.get("RF_GRID", "48"))
MAX_SCENES = int(os.environ.get("RF_MAX_SCENES", "0"))
STRIDE = int(os.environ.get("RF_STRIDE", "20"))
LABEL_FILES = os.environ.get("RF_LABEL_FILES",
    "conflict_labels.npz,conflict_labels_L2.npz,conflict_labels_L3.npz").split(",")
from scipy.ndimage import gaussian_filter
MC_N = int(os.environ.get("RF_MC", "0"))       # >0: add a Monte-Carlo trajectory-sample occupancy baseline
MP_SMOOTH = os.environ.get("RF_MP_SMOOTH", "") != ""   # add deterministic Gaussian-smoothed meeting-prob reads
SMSIG = [1.0, 2.0, 3.0]
SM_METHODS = ["prob_s%d" % int(s) for s in SMSIG] if MP_SMOOTH else []
ESWEEP = os.environ.get("RF_ENERGY_SWEEP", "") != ""   # add energy reads at rf-dilation sigma = 1/2/3
ESIG = [1.0, 2.0, 3.0]
ES_METHODS = ["es%d" % int(s) for s in ESIG] if ESWEEP else []
METHODS = ["ours", "ours_prob", "pora", "ttc", "dsf"] + (["mc"] if MC_N else []) + SM_METHODS + ES_METHODS
PROFILE = ["ours", "ours_prob", "pora"] + (["mc"] if MC_N else []) + SM_METHODS + ES_METHODS

def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0: return float("nan")
    u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); avg = (csum - cnt + csum + 1) / 2.0
    R = avg[inv][y == 1].sum()
    return float((R - npos*(npos+1)/2) / (npos*nneg))

def ap(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int)
    if y.sum() == 0: return float("nan")
    o = np.argsort(-s, kind="mergesort"); y = y[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / y.sum()
    return float(((rec - np.concatenate([[0], rec[:-1]])) * prec).sum())

def build(sl):
    c = default_dict()
    return RiskFlow(seq_len=c["seq_len"], input_dim=c["input_dim"], feature_dim=c["feature_dim"],
        embedding_dim=c["embedding_dim"], hidden_dim=c["hidden_dim"], max_num_cars=c["max_num_cars"],
        num_classes=c["num_classes"], gru_layers=c["gru_layers"], num_heads=c["num_heads"], dropout=c["dropout"],
        norm_rotation=c["norm_rotate"], flow_layers=c["flow_layers"], flow_hidden_dim=c["flow_hidden_dim"],
        coupling_layers=c["coupling_layers"], use_cnf=c["use_cnf"], use_cgmm=c["use_cgmm"], gmm_modes=c["gmm_modes"],
        use_world_model=True, wm_state_dim=c["wm_state_dim"], action_dim=c["action_dim"], scene_level=sl,
        use_map=True, map_size=c["map_size"], map_data_dir=_reg["map_data_dir"],
        map_dataset=_reg["map_dataset"],
        map_local=os.environ.get("RF_MAP_LOCAL", "0").lower() in ("1", "true"),
        map_crop_m=c.get("map_crop_m", 40.0), map_raster_res=c.get("map_raster_res", 192))

def main():
    c = default_dict(); dev = "cuda" if torch.cuda.is_available() else "cpu"; K = c["seq_len"]
    ind = _reg["LoaderClass"](root=_reg["root"], max_samples=c["maximum_samples"], train_ratio=c["train_ratio"],
              train_batch_size=c["train_batch_size"], test_batch_size=1, missing_rate=c["masked_data_ratio"],
              max_num_cars=c["max_num_cars"], max_empty_frames=c["max_empty_frames"], seq_len=c["seq_len"],
              moving_window=c["seq_len"]*2, sampling_step=c["sampling_step"], should_shuffle=False,
              include_future=c["include_future"])
    site = ind.observation_site_by_scope("all")
    ckpt_ego = os.environ.get("RF_CKPT_EGO", "serialized/riskflow_ind_8.pt")
    ckpt_joint = os.environ.get("RF_CKPT_JOINT", "serialized/riskflow_ind_7.pt")
    me = build(False).to(dev).eval(); me.load_state_dict(torch.load(ckpt_ego, map_location=dev), strict=False)
    mj = build(True).to(dev).eval();  mj.load_state_dict(torch.load(ckpt_joint, map_location=dev), strict=False)
    g1 = torch.linspace(0.05, 0.95, S); GX, GY = torch.meshgrid(g1, g1, indexing="ij")
    grid = torch.stack([GX.reshape(-1), GY.reshape(-1)], -1).to(dev)
    eng = JointRiskField(me, mj, grid, S, K, dev, min_hist=30)

    sidx = []; sc = {m: [] for m in METHODS}; prof = {m: [] for m in PROFILE}
    n = 0
    CACHE = f"conflict_scores_{os.environ.get('RF_DATASET','ind')}_s{STRIDE}_g{S}{('_mc%d' % MC_N) if MC_N else ''}{'_mps' if MP_SMOOTH else ''}{'_es' if ESWEEP else ''}.npz"
    if os.environ.get("RF_USE_SCORE_CACHE", "1") != "0" and os.path.exists(CACHE):
        z = np.load(CACHE)
        sidx = z["sidx"]; sc = {m: list(z[m]) for m in METHODS}
        prof = {m: list(z["prof_" + m]) for m in PROFILE}; n = len(sidx)
        print(f"RESULT loaded cached scores n={n} from {CACHE} (label-independent)", flush=True)
    else:
      with torch.no_grad():
        for i, b in enumerate(site.test_loader):
            if STRIDE > 1 and (i % STRIDE) != 0: continue
            x = b["input"].to(dev); f = b["feature"].to(dev); vt = b["type"].to(dev); fut = b["future"].to(dev)
            if torch.isnan(x[:,0,-2:,:]).any() or torch.isnan(fut[0,0]).any(): continue
            loc = int(b["locationId"].view(-1)[0]); loc_t = torch.tensor([loc], device=dev)
            bx = boundaries_for_location(loc)
            scale = torch.tensor([float(bx[0,1]-bx[0,0]), float(bx[1,1]-bx[1,0])], device=dev)
            sca = scale.cpu().numpy(); lo = np.array([float(bx[0,0]), float(bx[1,0])])
            neigh = [a for a in range(1, x.shape[1]) if not torch.isnan(x[0,a,-1]).any()
                     and int((~torch.isnan(x[0,a,:,0])).sum()) >= 30]
            if not neigh: continue
            xn = x[0].cpu().numpy()
            p0 = xn[0,-1]*sca+lo; v0 = (xn[0,-1]-xn[0,-2])*sca/DT
            Po = np.stack([xn[a,-1]*sca+lo for a in neigh]); Vo = np.stack([(xn[a,-1]-xn[a,-2])*sca/DT for a in neigh])
            _fo = eng.field(x, f, vt, neigh, loc_t, scale, return_dens=True, return_mp=True,
                            dims_m=(SDIMS.get(i) if SDIMS else None), return_raw_rf=ESWEEP)
            rf, dens, mp = _fo[0], _fo[1], _fo[2]
            s_o, pr_o = ours_peak(rf)
            s_mp, pr_mp = float(mp.reshape(-1, K).max(0).max()), mp.reshape(-1, K).max(0)
            s_p, pr_p = pora_style_score(dens[0], dens, K)
            sc["ours"].append(s_o); sc["ours_prob"].append(s_mp); sc["pora"].append(s_p)
            sc["ttc"].append(ttc_score(p0, v0, Po, Vo)[0]); sc["dsf"].append(dsf_score(p0, v0, Po, Vo))
            prof["ours"].append(pr_o); prof["ours_prob"].append(pr_mp); prof["pora"].append(pr_p)
            if MP_SMOOTH:                                     # deterministic spatial blur of the EXACT meeting prob (no sampling)
                for sg in SMSIG:
                    mps = gaussian_filter(mp, (sg, sg, 0))    # smooth each frame spatially; no temporal smoothing
                    pk = mps.reshape(-1, K).max(0)
                    sc["prob_s%d" % int(sg)].append(float(pk.max()))
                    prof["prob_s%d" % int(sg)].append(pk)
            if ESWEEP:                                        # energy read at rf-dilation sigma = 1/2/3 (es2 == ours)
                for sg in ESIG:
                    s_es, pr_es = ours_peak(gaussian_filter(_fo[3], (sg, sg, 0)))
                    sc["es%d" % int(sg)].append(s_es); prof["es%d" % int(sg)].append(pr_es)
            if MC_N:                                          # MC trajectory-sample occupancy (second forward, sampled densities)
                _mc = eng.field(x, f, vt, neigh, loc_t, scale, return_mp=True, mc_n=MC_N,
                                dims_m=(SDIMS.get(i) if SDIMS else None))
                mp_mc = _mc[1]
                sc["mc"].append(float(mp_mc.reshape(-1, K).max(0).max()))
                prof["mc"].append(mp_mc.reshape(-1, K).max(0))
            sidx.append(i); n += 1
            if MAX_SCENES and n >= MAX_SCENES: break
    sidx = np.array(sidx)
    np.savez(CACHE, sidx=sidx, **{m: np.array(sc[m]) for m in METHODS},
             **{"prof_"+m: np.array(prof[m]) for m in PROFILE})
    print(f"RESULT scored n={n} scenes (grid={S}) -> cached {CACHE}", flush=True)

    out = {"n_scenes": int(n), "grid": S, "labels": {}}
    for lf in LABEL_FILES:
        lf = lf.strip()
        if not os.path.exists(lf): print(f"RESULT [skip] {lf} missing"); continue
        z = np.load(lf, allow_pickle=True); D = z["data"]
        lab = {int(r[0]): (int(r[2]), int(r[3])) for r in D}
        keep = np.array([k for k, s in enumerate(sidx) if int(s) in lab])
        y = np.array([lab[int(sidx[k])][0] for k in keep])
        onk = np.array([lab[int(sidx[k])][1] for k in keep])
        rate = float(y.mean()); rec = {"rate": rate, "n": int(len(y))}
        print(f"RESULT === {lf}  rate={rate*100:.1f}%  n={len(y)} ===", flush=True)
        for m in METHODS:
            a = auroc(np.array(sc[m])[keep], y); p = ap(np.array(sc[m])[keep], y)
            rec[m] = {"AUROC": a, "AP": p}
            print(f"RESULT {m:10s} AUROC={a:.3f} AP={p:.3f}", flush=True)
        for m in PROFILE:
            P = np.array(prof[m])[keep]; safe = P[y == 0]
            thr = float(np.percentile(safe.max(1), 90)) if len(safe) else 0.0
            leads = [(onk[k] - np.where(P[k] > thr)[0][0]) * DT
                     for k in range(len(y)) if y[k] == 1 and (P[k] > thr).any()]
            lt = float(np.mean(leads)) if leads else float("nan")
            rec[m]["lead_time_s"] = lt; rec[m]["detect@FAR10"] = float(len(leads)/max((y==1).sum(),1))
            print(f"RESULT {m:10s} lead={lt:.2f}s detect@FAR10={rec[m]['detect@FAR10']*100:.0f}%", flush=True)
        out["labels"][lf] = rec
    with open("conflict_eval.json", "w") as fh: json.dump(out, fh, indent=2)
    print("saved conflict_eval.json + conflict_scores.npz")

if __name__ == "__main__":
    main()
