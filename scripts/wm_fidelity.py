"""World-model quality + counterfactual-fidelity evaluation.

Answers "how good is the trained world model, and can it simulate realistic
surrounding-agent reactions to ego maneuvers?" -- measured on observational data.

B1  Factual prediction quality (well-trained?): neighbour conditional NLL and ego
    marginal NLL, conditioned on the REALIZED ego action; plus a MAP-trajectory ADE.
B3  Counterfactual-fidelity PROXY (key): on scenes where the ego GENUINELY maneuvered
    (top quartile of realized |a_ego|), does conditioning the joint on the TRUE ego
    action predict neighbours' realized futures better (lower NLL) than a WRONG
    (reversed) or autonomous (zero) action? Gap = NLL(wrong) - NLL(true); positive =
    the reaction term carries real causal signal. Bootstrap CI over scenes.
(B2 sensitivity vs maneuver magnitude is scripts/cf_react.py.)

Env: RF_MAX_SCENES (400), RF_MANEUVER_Q (0.75 quartile cut for "maneuver" scenes).
"""

import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from datasets.registry import get_dataset
_reg = get_dataset()                          # RF_DATASET env (ind|ad4che)
from model.RiskFlow import RiskFlow
from riskflow_config import default_dict

A_SCALE = 100.0
LOG2PI = math.log(2 * math.pi)
MAX_SCENES = int(os.environ.get("RF_MAX_SCENES", "400"))
MANEUVER_Q = float(os.environ.get("RF_MANEUVER_Q", "0.75"))

def build(sl, c, dev):
    return RiskFlow(seq_len=c["seq_len"], input_dim=c["input_dim"], feature_dim=c["feature_dim"],
        embedding_dim=c["embedding_dim"], hidden_dim=c["hidden_dim"], max_num_cars=c["max_num_cars"],
        num_classes=c["num_classes"], gru_layers=c["gru_layers"], num_heads=c["num_heads"], dropout=c["dropout"],
        norm_rotation=c["norm_rotate"], flow_layers=c["flow_layers"], flow_hidden_dim=c["flow_hidden_dim"],
        coupling_layers=c["coupling_layers"], use_cnf=c["use_cnf"], use_cgmm=c["use_cgmm"], gmm_modes=c["gmm_modes"],
        use_world_model=True, wm_state_dim=c["wm_state_dim"], action_dim=c["action_dim"], scene_level=sl,
        use_map=True, map_size=c["map_size"], map_data_dir=_reg["map_data_dir"],
        map_dataset=_reg["map_dataset"]).to(dev).eval()

def ego_action(ego_fut):                       # (1,K,2) -> (1,K,2) proxy
    a = ego_fut[:, 2:] - 2*ego_fut[:, 1:-1] + ego_fut[:, :-2]
    return torch.cat([a[:, :1], a, a[:, -1:]], 1) * A_SCALE

def neigh_nll(z, det, fut, D):
    per = 0.5*(z.pow(2).sum(-1) + D*LOG2PI) + det          # (1,N,K)
    valid = torch.isfinite(fut).all(-1).all(-1)            # (1,N)
    valid[:, 0] = False
    m = valid.unsqueeze(-1).expand_as(per) & torch.isfinite(per)
    return per[m].mean().item() if m.any() else float("nan")

def main():
    c = default_dict(); dev = "cuda" if torch.cuda.is_available() else "cpu"; D = c["input_dim"]
    ind = _reg["LoaderClass"](root=_reg["root"], max_samples=c["maximum_samples"], train_ratio=c["train_ratio"],
              train_batch_size=c["train_batch_size"], test_batch_size=1, missing_rate=c["masked_data_ratio"],
              max_num_cars=c["max_num_cars"], max_empty_frames=c["max_empty_frames"], seq_len=c["seq_len"],
              moving_window=c["seq_len"]*2, sampling_step=c["sampling_step"], should_shuffle=False,
              include_future=c["include_future"])
    site = ind.observation_site_by_scope("all"); K = c["seq_len"]
    ckpt_ego = os.environ.get("RF_CKPT_EGO", "serialized/riskflow_ind_8.pt")
    ckpt_joint = os.environ.get("RF_CKPT_JOINT", "serialized/riskflow_ind_7.pt")
    me = build(False, c, dev); me.load_state_dict(torch.load(ckpt_ego, map_location=dev), strict=False)
    mj = build(True, c, dev);  mj.load_state_dict(torch.load(ckpt_joint, map_location=dev), strict=False)
    nll_true, nll_ego, adem = [], [], []
    rows = []   # (maneuver_mag, nll_true, nll_rev, nll_zero)
    with torch.no_grad():
        n = 0
        for i, b in enumerate(site.test_loader):
            x = b["input"].to(dev); f = b["feature"].to(dev); t = b["type"].to(dev); fut = b["future"].to(dev)
            tg = b["target"].to(dev)
            if torch.isnan(x[:,0,-2:,:]).any() or torch.isnan(fut[0,0]).any(): continue
            valid = torch.isfinite(fut).all(-1).all(-1);
            if valid[0,1:].sum() == 0: continue
            loc = b.get("locationId"); loc = loc.to(dev) if loc is not None else None
            a_true = ego_action(fut[:, 0])
            mag = a_true.abs().mean().item()
            # B1: ego marginal NLL (single-target) + MAP-ADE
            ze, de, ee = me(x, tg, f, t, location_id=loc)
            nll_ego.append((0.5*(ze.pow(2).sum(-1)+D*LOG2PI)+de).mean().item())
            # B1/B3: joint neighbour NLL under true / reversed / zero ego action
            zt, dt_, _ = mj(x, fut, f, t, actions=a_true, location_id=loc)
            nt = neigh_nll(zt, dt_, fut, D); nll_true.append(nt)
            zr, dr, _ = mj(x, fut, f, t, actions=-a_true, location_id=loc)
            nr = neigh_nll(zr, dr, fut, D)
            zz, dz, _ = mj(x, fut, f, t, actions=torch.zeros_like(a_true), location_id=loc)
            nz = neigh_nll(zz, dz, fut, D)
            rows.append((mag, nt, nr, nz))
            n += 1
            if n >= MAX_SCENES: break
    R = np.array(rows)                                # (n,4)
    mag, nt, nr, nz = R[:,0], R[:,1], R[:,2], R[:,3]
    print(f"=== B1 factual prediction quality ({n} scenes) ===")
    print(f"RESULT ego marginal NLL = {np.nanmean(nll_ego):.4f}")
    print(f"RESULT neighbour conditional NLL (true action) = {np.nanmean(nt):.4f}")
    # B3: gap on maneuver scenes (high |a_ego|)
    cut = np.quantile(mag, MANEUVER_Q)
    mask = mag >= cut
    def boot(d, B=2000):
        d = d[np.isfinite(d)]
        if len(d) < 2: return float("nan"), float("nan")
        idx = np.random.randint(0, len(d), (B, len(d)))
        m = d[idx].mean(1); return float(d.mean()), (float(np.percentile(m,2.5)), float(np.percentile(m,97.5)))
    print(f"=== B3 counterfactual-fidelity proxy (maneuver scenes, |a|>=q{MANEUVER_Q}, n={int(mask.sum())}) ===")
    for name, wrong in [("reversed", nr), ("zero", nz)]:
        gap = (wrong - nt)[mask]                       # >0 => true action predicts neighbours better
        mean, ci = boot(gap)
        print(f"RESULT gap NLL({name})-NLL(true) = {mean:+.4f}  95%CI={ci}  "
              f"(>0 = true ego action helps; {'SIGNAL' if isinstance(ci,tuple) and ci[0]>0 else 'no sig'})")
    print(f"RESULT all-scenes gap NLL(reversed)-NLL(true) = {np.nanmean(nr-nt):+.4f}")

if __name__ == "__main__":
    main()
