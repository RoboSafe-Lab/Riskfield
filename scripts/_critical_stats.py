"""Summarize GT-geometry conflict labels: counts per dataset + per-critical-scene
CSV (scene, site, location, ego/partner trackIds, PET, min gap, onset)."""
import sys
import numpy as np

SETS = [
    ("ind",    "conflict_labels_gt.npz",        "critical_scenes_ind.csv"),
    ("ad4che", "conflict_labels_ad4che_gt.npz", "critical_scenes_ad4che.csv"),
    ("round",  "conflict_labels_round_gt.npz",  "critical_scenes_round.csv"),
]
for name, npz, out in SETS:
    try:
        z = np.load(npz, allow_pickle=True)
    except FileNotFoundError:
        print(f"RESULT {name}: {npz} missing"); continue
    d = z["data"]; sites = z["sites"] if "sites" in z.files else np.array([""] * len(d))
    lab = d[:, 2] > 0
    matched = int((d[:, 8] >= 0).sum())
    print(f"RESULT {name}: scenes={len(d)} critical={int(lab.sum())} "
          f"rate={lab.mean()*100:.1f}%  geom-matched={matched}/{len(d)}")
    with open(out, "w") as f:
        f.write("scene,site,location,ego_trackId,partner_slot,partner_trackId,"
                "pet_s,min_gap_m,onset_k\n")
        for r, st in zip(d, sites):
            if r[2] < 1:
                continue
            f.write(f"{int(r[0])},{st},{int(r[1])},{int(r[8])},{int(r[6])},"
                    f"{int(r[9])},{r[4]:.3f},{r[5]:.2f},{int(r[3])}\n")
    print(f"RESULT wrote {out}")
