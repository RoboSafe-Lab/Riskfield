import json
d = json.load(open("sweep_ind.json"))["labels"]
for lf in ["conflict_labels_petloose.npz", "conflict_labels_pet.npz", "conflict_labels_petstrict.npz"]:
    if lf not in d:
        print(lf, "MISSING; have:", list(d.keys())); continue
    r = d[lf]
    print("\n==", lf, " rate=%.1f%%  n=%s" % (r.get("rate", 0) * 100, r.get("n")))
    for m in ["dsf", "ttc", "pora", "ours", "ours_prob"]:
        if m in r:
            print("  %-10s AUROC=%.3f AP=%.3f lead=%s" % (
                m, r[m]["AUROC"], r[m]["AP"], r[m].get("lead_time_s")))
