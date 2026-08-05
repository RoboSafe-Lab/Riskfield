import json
CFG = [("InD", "sweep_ind.json", "conflict_labels_petloose.npz"),
       ("AD4CHE", "sweep_ad4che.json", "conflict_labels_ad4che_petloose.npz"),
       ("rounD", "sweep_round.json", "conflict_labels_round_petloose.npz")]
for name, f, lf in CFG:
    d = json.load(open(f))["labels"]
    r = d[lf]
    print("\n==== %s  loose(1.5s)  rate=%.1f%%  n=%s ====" % (name, r.get("rate", 0) * 100, r.get("n")))
    for m in ["dsf", "ttc", "pora", "ours", "ours_prob"]:
        if m in r:
            x = r[m]
            lead = x.get("lead_time_s")
            ls = "---" if lead is None else "%+.2f" % lead
            print("  %-10s AUROC=%.3f AP=%.3f lead=%s" % (m, x["AUROC"], x["AP"], ls))
