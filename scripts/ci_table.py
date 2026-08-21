"""Bootstrap confidence intervals for the conflict-detection table (tab:robust).

Reviewer: "many of the improvements are small, readers cannot determine whether
gains are meaningful. Table 3 reports only point estimates."

This runs entirely on the score cache that eval_conflict.py already writes, so it
needs no model, no dataset and no GPU -- AUROC, AP and the early-warning lead are
pure functions of (per-scene scores, labels), and the scores are label-independent
by construction.

**The interval is a PAIRED bootstrap over scenes.** Every method is scored on the
identical scene set, so the quantity that answers "is this gain meaningful" is the
DIFFERENCE, resampled jointly: between-scene variance dominates each marginal
AUROC and cancels in the difference. Reporting two overlapping marginal intervals
instead would systematically understate the evidence. Marginal intervals are
printed too, but the Delta block is the one to put in the paper.

``auroc`` and ``ap`` are copied verbatim from eval_conflict.py (and the lead-time
rule from its PROFILE block) so the point estimates reproduce the published table;
the script prints them first for exactly that check.

Env:
  RF_SCORES  score-cache npz (required)     RF_LABELS  label npz, PET (required)
  RF_B       bootstrap replicates (10000)   RF_REF     reference method (pora)
  RF_SEED    0                              RF_DT      parsed from RF_SCORES filename
"""
import os
import re
import sys
import time

import numpy as np

CI = (2.5, 97.5)                                  # 95% percentile interval


def auroc(s, y):                                  # verbatim from eval_conflict.py
    s = np.asarray(s, float); y = np.asarray(y, int)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0: return float("nan")
    u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); avg = (csum - cnt + csum + 1) / 2.0
    R = avg[inv][y == 1].sum()
    return float((R - npos * (npos + 1) / 2) / (npos * nneg))


def ap(s, y):                                     # verbatim from eval_conflict.py
    s = np.asarray(s, float); y = np.asarray(y, int)
    if y.sum() == 0: return float("nan")
    o = np.argsort(-s, kind="mergesort"); y = y[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / y.sum()
    return float(((rec - np.concatenate([[0], rec[:-1]])) * prec).sum())


def lead(Pm, y, onk, dt):
    """Mean early-warning lead at a 10% false-alarm rate.

    Same rule as eval_conflict.py's PROFILE block -- threshold = 90th percentile
    of the per-scene max over SAFE scenes -- but vectorized, because the original
    per-scene Python loop would run n*B times here. The threshold is recomputed
    inside every replicate, since it is itself an estimate from the resample.
    """
    safe = Pm[y == 0]
    thr = float(np.percentile(safe.max(1), 90)) if len(safe) else 0.0
    mask = Pm > thr
    first = mask.argmax(1)                        # first crossing (0 if none; gated below)
    sel = (y == 1) & mask.any(1)
    if not sel.any(): return float("nan"), 0.0
    leads = (onk[sel] - first[sel]) * dt
    return float(leads.mean()), float(sel.sum() / max(int((y == 1).sum()), 1))


def band(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if v.size == 0: return float("nan"), float("nan")
    return tuple(float(x) for x in np.percentile(v, CI))


def main():
    scores = os.environ["RF_SCORES"]; labels = os.environ["RF_LABELS"]
    B = int(os.environ.get("RF_B", "10000")); REF = os.environ.get("RF_REF", "pora")
    # Derive DT from the cache filename, which encodes it (eval_conflict.py writes
    # ..._dt0.0667.npz). Taking it from the env instead lets a caller point at an
    # AD4CHE cache while leaving RF_DT at 0.08, which would scale every lead time
    # and its bootstrap interval by 1.2 -- silently, in the table meant to
    # establish significance. An explicit RF_DT still wins.
    _m = re.search(r"_dt(\d+(?:\.\d+)?)", os.path.basename(scores))
    DT = float(os.environ.get("RF_DT", _m.group(1) if _m else "0.08"))
    rng = np.random.default_rng(int(os.environ.get("RF_SEED", "0")))

    z = np.load(scores)
    keys = list(z.files)
    METHODS = [k for k in keys if k != "sidx" and not k.startswith("prof_")]
    PROFILE = [k[5:] for k in keys if k.startswith("prof_")]
    sidx = z["sidx"]

    # align scores to labels exactly as eval_conflict.py:156-160
    D = np.load(labels, allow_pickle=True)["data"]
    lab = {int(r[0]): (int(r[2]), int(r[3])) for r in D}
    keep = np.array([k for k, s in enumerate(sidx) if int(s) in lab])
    y = np.array([lab[int(sidx[k])][0] for k in keep])
    onk = np.array([lab[int(sidx[k])][1] for k in keep])
    Sm = {m: np.asarray(z[m], float)[keep] for m in METHODS}
    Pr = {m: np.asarray(z["prof_" + m], float)[keep] for m in PROFILE}
    n = len(y); npos = int(y.sum())

    print("=" * 78)
    print(f"RESULT scores={os.path.basename(scores)} labels={os.path.basename(labels)}")
    print(f"RESULT n={n} positives={npos} ({100.0*npos/n:.1f}%) B={B} ref={REF}")
    print("--- point estimates (must match the published table) ---")
    pt_a = {m: auroc(Sm[m], y) for m in METHODS}
    pt_p = {m: ap(Sm[m], y) for m in METHODS}
    pt_l = {m: lead(Pr[m], y, onk, DT) for m in PROFILE}
    for m in METHODS:
        extra = f" lead={pt_l[m][0]:+.2f}s det@FAR10={100*pt_l[m][1]:.0f}%" if m in PROFILE else ""
        print(f"RESULT {m:10s} AUROC={pt_a[m]:.3f} AP={pt_p[m]:.3f}{extra}")

    t0 = time.time()
    A = {m: np.full(B, np.nan) for m in METHODS}
    P = {m: np.full(B, np.nan) for m in METHODS}
    L = {m: np.full(B, np.nan) for m in PROFILE}
    skipped = 0
    for b in range(B):
        ii = rng.integers(0, n, n)                # resample SCENES, shared by all methods
        yb = y[ii]
        if yb.sum() == 0 or yb.sum() == n:        # degenerate replicate: no ranking defined
            skipped += 1; continue
        for m in METHODS:
            A[m][b] = auroc(Sm[m][ii], yb); P[m][b] = ap(Sm[m][ii], yb)
        onb = onk[ii]
        for m in PROFILE:
            L[m][b] = lead(Pr[m][ii], yb, onb, DT)[0]
    print(f"--- {B} replicates in {time.time()-t0:.0f}s"
          f"{f' ({skipped} degenerate, dropped)' if skipped else ''} ---")

    print(f"--- marginal 95% CI (context only; do NOT use these to judge gaps) ---")
    for m in METHODS:
        la, ha = band(A[m]); lp, hp = band(P[m])
        print(f"RESULT {m:10s} AUROC={pt_a[m]:.3f} [{la:.3f},{ha:.3f}]"
              f"  AP={pt_p[m]:.3f} [{lp:.3f},{hp:.3f}]")

    print(f"--- PAIRED difference vs {REF}  (this is the reportable statistic) ---")
    for m in METHODS:
        if m == REF: continue
        for name, cur, ref, pt in (("dAUROC", A[m], A[REF], pt_a[m] - pt_a[REF]),
                                   ("dAP   ", P[m], P[REF], pt_p[m] - pt_p[REF])):
            d = cur - ref; d = d[np.isfinite(d)]
            lo, hi = band(d)
            p = 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean()))
            print(f"RESULT {m:10s} {name}={pt:+.3f} [{lo:+.3f},{hi:+.3f}] "
                  f"p={min(p,1.0):.4f} {'SIG' if lo*hi > 0 else 'ns '}")
    for m in PROFILE:
        if m == REF: continue
        d = L[m] - L[REF]; d = d[np.isfinite(d)]
        if d.size == 0: continue
        lo, hi = band(d)
        p = 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean()))
        print(f"RESULT {m:10s} dLEAD ={pt_l[m][0]-pt_l[REF][0]:+.3f}s [{lo:+.3f},{hi:+.3f}] "
              f"p={min(p,1.0):.4f} {'SIG' if lo*hi > 0 else 'ns '}")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
