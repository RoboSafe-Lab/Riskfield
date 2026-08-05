import numpy as np

z = np.load("conflict_labels_ad4che_pet.npz", allow_pickle=True)
D = z["data"]
cols = [str(c) for c in z["cols"]]
ci = {c: i for i, c in enumerate(cols)}
rows = {int(r[0]): r for r in D}

# reference scene 1252 loc
r1252 = rows[1252]
loc1252 = r1252[ci["loc"]]
print("1252 loc=", loc1252, "pet=", float(r1252[ci["pet"]]), "label=", int(r1252[ci["label"]]))
print("1915 loc=", rows[1915][ci["loc"]], "pet=", float(rows[1915][ci["pet"]]))

# PET-positive scenes at the same loc, sorted by ttc ascending (fast closing first)
cand = []
for r in D:
    if int(r[ci["label"]]) != 1:
        continue
    if r[ci["loc"]] != loc1252:
        continue
    pet = float(r[ci["pet"]])
    ttc = float(r[ci["ttc"]])
    cand.append((int(r[0]), pet, ttc, int(r[ci["n_neigh"]])))

# valid finite ttc, sort ascending (fastest approach), exclude 1915/1252
cand = [c for c in cand if c[0] not in (1252, 1915) and 0 < c[2] < 1e5]
cand.sort(key=lambda x: x[2])
print("== PET-positive @ loc %s, sorted by TTC asc (n=%d) ==" % (loc1252, len(cand)))
for sid, pet, ttc, nn in cand[:20]:
    print("  scene %d  pet=%.3f  ttc=%.2f  n_neigh=%d" % (sid, pet, ttc, nn))
