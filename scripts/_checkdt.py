import numpy as np
for f in ["conflict_labels_honest.npz", "conflict_labels_ad4che_honest.npz",
          "conflict_labels_round_honest.npz", "conflict_labels_ad4che_pet.npz",
          "conflict_labels_pet.npz", "conflict_labels_round_pet.npz"]:
    try:
        z = np.load(f, allow_pickle=True)
    except Exception as e:
        print(f, "ERR", e); continue
    dt = z["dt"] if "dt" in z.files else "MISSING"
    extra = [k for k in z.files if k not in ("data", "cols", "sites")]
    print("%-38s dt=%s  meta=%s" % (f, dt, extra))
