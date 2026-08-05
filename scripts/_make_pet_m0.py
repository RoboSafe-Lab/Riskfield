"""Threshold the margin-0 honest labels to PET-only (pet<1.0s)."""
import numpy as np
for ds, f in [("InD", "conflict_labels_honest_m0.npz"),
              ("AD4CHE", "conflict_labels_ad4che_honest_m0.npz"),
              ("rounD", "conflict_labels_round_honest_m0.npz")]:
    z = np.load(f, allow_pickle=True)
    cols = [str(c) for c in z["cols"]]
    D = z["data"].copy()
    pet = D[:, cols.index("pet")].astype(float)
    lab = (pet < 1.0).astype(float)
    D[:, cols.index("label")] = lab
    out = f.replace("_honest_m0", "_pet_m0")
    meta = {k: z[k] for k in z.files if k not in ("data", "cols", "sites")}
    np.savez(out, data=D, cols=z["cols"], sites=z["sites"], **meta)
    print("RESULT %s margin0 PET<1.0s rate=%.2f%% conflicts=%d/%d -> %s" % (
        ds, lab.mean() * 100, int(lab.sum()), len(lab), out))
