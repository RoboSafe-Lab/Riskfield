import numpy as np

def show(f, ids):
    try:
        z = np.load(f, allow_pickle=True)
    except Exception as e:
        print("==", f, "ERR", e); return
    D = z["data"]
    cols = [str(c) for c in z["cols"]] if "cols" in z.files else None
    print("==", f, "cols=", cols, "nrows=", len(D))
    rows = {int(r[0]): r for r in D}
    ip = cols.index("pet") if cols else 4
    for i in ids:
        if i in rows:
            r = rows[i]
            pet = float(r[ip])
            pets = "inf" if pet != pet or pet > 1e5 else "%.3f" % pet
            print("  scene %d: label=%d pet=%s onset=%d" % (i, int(r[2]), pets, int(r[3])))
        else:
            print("  scene %d: NOT in file" % i)

# AD4CHE hero figure scenes
show("conflict_labels_ad4che_honest.npz", [1252, 1915])
show("conflict_labels_ad4che_pet.npz", [1252, 1915])
# InD hero scenes against the honest + pet files too
show("conflict_labels_honest.npz", [44040, 56660])
show("conflict_labels_pet.npz", [44040, 56660])
