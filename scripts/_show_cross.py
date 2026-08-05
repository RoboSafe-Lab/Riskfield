import json, glob, os
# dump every label-file entry (rate + ours_prob AUROC/AP/lead) from each eval json
for f in sorted(glob.glob("sweep_*.json") + glob.glob("conflict_eval*.json")):
    try:
        d = json.load(open(f)).get("labels", {})
    except Exception as e:
        print(f, "ERR", e); continue
    print("\n==== %s ====" % f)
    for lf, r in d.items():
        op = r.get("ours_prob", {})
        print("  %-40s rate=%5.1f%% n=%-5s | ours_prob AUROC=%.3f AP=%.3f lead=%s" % (
            lf, r.get("rate", 0) * 100, r.get("n"),
            op.get("AUROC", float('nan')), op.get("AP", float('nan')), op.get("lead_time_s")))
