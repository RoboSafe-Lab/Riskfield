import json
for name, f, suf in [("AD4CHE", "sweep_ad4che.json", "ad4che"), ("rounD", "sweep_round.json", "round")]:
    d = json.load(open(f))["labels"]
    print("\n==== %s ====" % name)
    for tau, tag in [(1.5, "petloose"), (1.0, "pet"), (0.5, "petstrict")]:
        lf = "conflict_labels_%s_%s.npz" % (suf, tag)
        if lf not in d:
            print("  tau=%.1f MISSING" % tau); continue
        r = d[lf]
        oe = r.get("ours", {}); op = r.get("ours_prob", {})
        print("  tau=%.1f  ours(energy) lead=%+.2f (AUROC %.3f AP %.3f) | ours(prob) lead=%+.2f (AUROC %.3f AP %.3f)" % (
            tau, oe.get("lead_time_s", float('nan')), oe.get("AUROC", float('nan')), oe.get("AP", float('nan')),
            op.get("lead_time_s", float('nan')), op.get("AUROC", float('nan')), op.get("AP", float('nan'))))
