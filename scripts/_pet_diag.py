"""Verify PET for specific AD4CHE scenes straight from the ORIGINAL recording.
For each target scene: attach recorded geometry (RF_GT_GEOM), then for the ego
and its labelled partner print recorded dims, whether each matched the raw track,
the min same-frame centre gap, the true-box edge clearance at closest approach,
and PET computed with margin 0.3 (as labelled) vs margin 0.0 (true boxes)."""
import os, sys
os.environ.setdefault("RF_DATASET", "ad4che")
os.environ.setdefault("RF_GT_GEOM", "1")
os.environ.setdefault("RF_MARGIN", "0.3")
os.environ.setdefault("RF_DT", "0.0667")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import scripts.conflict_labels as CL

TARGETS = [int(x) for x in os.environ.get("RF_TARGETS", "1915,1252,2214,2238,2049").split(",")]
c = CL.default_dict()
reg = CL._reg
ind = reg["LoaderClass"](root=reg["root"], max_samples=c["maximum_samples"], train_ratio=c["train_ratio"],
        train_batch_size=c["train_batch_size"], test_batch_size=1, missing_rate=c["masked_data_ratio"],
        max_num_cars=c["max_num_cars"], max_empty_frames=c["max_empty_frames"], seq_len=c["seq_len"],
        moving_window=c["seq_len"]*2, sampling_step=c["sampling_step"], should_shuffle=False,
        include_future=c["include_future"])
site = ind.observation_site_by_scope("all")
Th = c["seq_len"]; sstep = int(c["sampling_step"])
boundaries_for_location = reg["boundaries_for_location"]

scenes = []
for i, b in enumerate(site.test_loader):
    if i not in TARGETS:
        continue
    x = b["input"]; fut = b["future"]; types = b["type"]
    loc = int(b["locationId"].view(-1)[0])
    if torch.isnan(x[:, 0, -2:, :]).any() or torch.isnan(fut[0, 0]).any():
        print("scene", i, "skipped (nan)"); continue
    bx = boundaries_for_location(loc)
    scale = np.array([float(bx[0,1]-bx[0,0]), float(bx[1,1]-bx[1,0])], np.float64)
    lo = np.array([float(bx[0,0]), float(bx[1,0])], np.float64)
    F = fut[0].numpy()
    valid = [a for a in range(F.shape[0]) if not np.isnan(F[a]).any()
             and int((~torch.isnan(x[0,a,:,0])).sum()) >= 30]
    neigh = [a for a in valid if a != 0]
    if not neigh:
        print("scene", i, "no neighbours"); continue
    scenes.append(dict(i=i, loc=loc, valid=valid, neigh=neigh, Pm=F*scale+lo,
                       types=types[0].numpy().reshape(-1),
                       ego_id=int(np.asarray(b["trackId"]).reshape(-1)[0]),
                       sf=int(np.asarray(b["startFrame"]).reshape(-1)[0])))
    if len(scenes) == len(TARGETS):
        break

CL._attach_gt_geom(scenes, ind, Th, sstep, scenes[0]["Pm"].shape[1])

def edge_clear(Pe, the, hel, hew, Pj, thj, hjl, hjw):
    """approx min edge-to-edge clearance over frames using SAT support along centre line."""
    best = np.inf; bk = -1
    for k in range(Pe.shape[0]):
        d = Pj[k]-Pe[k]; dist = np.hypot(d[0], d[1])
        if dist < 1e-6: return -1.0, k
        u = d/dist
        re = hel*abs(np.cos(the[k])*u[0]+np.sin(the[k])*u[1]) + hew*abs(-np.sin(the[k])*u[0]+np.cos(the[k])*u[1])
        rj = hjl*abs(np.cos(thj[k])*u[0]+np.sin(thj[k])*u[1]) + hjw*abs(-np.sin(thj[k])*u[0]+np.cos(thj[k])*u[1])
        clr = dist - re - rj
        if clr < best: best = clr; bk = k
    return best, bk

for s in scenes:
    i = s["i"]; Pm = s["Pm"]; geom = s.get("geom", {}); types = s["types"]
    TH = {a: CL._headings(Pm[a]) for a in s["valid"]}
    for a in geom: TH[a] = geom[a][2]
    hd0 = geom[0][:2] if 0 in geom else CL.half_dims(types[0])
    # find labelled partner = min PET neighbour
    best = None
    for j in s["neigh"]:
        hdj = geom[j][:2] if j in geom else CL.half_dims(types[j])
        pe, tt, gp, gpm, on = CL.pair_conflict(Pm[0], TH[0], hd0, Pm[j], TH[j], hdj, 0.3, 2.0)
        pe0, _, _, _, _ = CL.pair_conflict(Pm[0], TH[0], hd0, Pm[j], TH[j], hdj, 0.0, 2.0)
        if best is None or pe < best[1]:
            best = (j, pe, pe0, gp, gpm, hdj)
    j, pe, pe0, gp, gpm, hdj = best
    hdj_ = geom[j][:2] if j in geom else CL.half_dims(types[j])
    clr, ck = edge_clear(Pm[0], TH[0], hd0[0], hd0[1], Pm[j], TH[j], hdj_[0], hdj_[1])
    print("\n=== scene %d  loc=%d  partner_agent=%d ===" % (i, s["loc"], j))
    print("  ego matched=%s  recorded LxW=%.2f x %.2f m  (class-default %s)" % (
        0 in geom, hd0[0]*2, hd0[1]*2, CL.VEH_LW.get(int(types[0]))))
    print("  partner matched=%s  recorded LxW=%.2f x %.2f m  (class-default %s)" % (
        j in geom, hdj_[0]*2, hdj_[1]*2, CL.VEH_LW.get(int(types[j]))))
    print("  min same-frame centre gap = %.2f m" % gp)
    print("  min TRUE-box edge clearance = %.2f m  at frame %d (negative => boxes really overlap)" % (clr, ck))
    print("  PET(margin=0.3, as labelled) = %s s" % ("%.3f" % pe if np.isfinite(pe) else "inf"))
    print("  PET(margin=0.0, true boxes)  = %s s" % ("%.3f" % pe0 if np.isfinite(pe0) else "inf"))
