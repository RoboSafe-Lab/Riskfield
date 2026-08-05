import numpy as np
z = np.load("conflict_labels_honest.npz", allow_pickle=True)
cols = list(z["cols"]); D = z["data"]
g = lambda name: D[:, cols.index(name)]
pet, ttc, mg, gm = g("pet"), g("ttc"), g("min_gap"), g("min_gap_mot")
sc, loc, et, pt = g("scene"), g("loc"), g("ego_tid"), g("partner_tid")
mask = (pet > 1.0) & (gm < 3.0)                 # near in space, but NOT a PET crossing
idx = np.where(mask)[0]
print(f"RESULT n(pet>1s AND min_gap_mot<3m) = {len(idx)}  ({len(idx)/len(D)*100:.1f}% of scenes)")
order = idx[np.argsort(gm[idx])]               # closest first
for k in order[:14]:
    pets = "inf" if pet[k] > 1e5 else f"{pet[k]:.2f}"
    print(f"RESULT scene={int(sc[k])} loc={int(loc[k])} PET={pets}s ttc={ttc[k]:.2f}s "
          f"min_gap_mot={gm[k]:.2f}m min_gap={mg[k]:.2f}m ego_tid={int(et[k])} partner_tid={int(pt[k])}")
