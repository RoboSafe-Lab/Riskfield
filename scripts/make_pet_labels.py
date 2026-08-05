"""Derive PET-only conflict labels from the honest label files: label = (pet < tau).
The pet column is already the assumption-free post-encroachment time; this just drops
the min-distance trigger so a conflict = a real recorded crossing only."""
import numpy as np
FILES = [("InD", "conflict_labels_honest.npz"),
         ("AD4CHE", "conflict_labels_ad4che_honest.npz"),
         ("rounD", "conflict_labels_round_honest.npz")]
for tau, suffix in [(1.5, "_petloose"), (1.0, "_pet"), (0.5, "_petstrict")]:
    for ds, f in FILES:
        z = np.load(f, allow_pickle=True); cols = list(z["cols"]); D = z["data"].copy()
        pet = D[:, cols.index("pet")]
        lab = (pet < tau).astype(float)
        D[:, cols.index("label")] = lab
        out = f.replace("_honest", suffix)
        # carry through all source metadata (dt, margin, gt_geom, ...) so derived
        # files stay self-describing; only tau_pet/mode are overridden below.
        meta = {k: z[k] for k in z.files if k not in ("data", "cols", "sites", "tau_pet", "mode")}
        np.savez(out, data=D, cols=z["cols"], sites=z["sites"], tau_pet=tau, mode="pet_only", **meta)
        print(f"RESULT {ds} PET<{tau}s rate={lab.mean()*100:.2f}% "
              f"conflicts={int(lab.sum())}/{len(lab)} -> {out}")
