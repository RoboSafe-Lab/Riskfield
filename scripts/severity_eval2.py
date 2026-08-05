"""Severity correlation (self-contained): for each GT conflict scene, compute each
method's score AND the GT collision energy at the near-miss (min-gap frame), then
Spearman-correlate. ours-energy (prob x severity) should track severity; ours-prob
and TTC should not."""
import os, sys, math; sys.path.insert(0, ".")
import numpy as np, torch
from scipy.stats import spearmanr
from datasets.registry import get_dataset
from model.RiskFlow import RiskFlow
from riskflow_config import default_dict
from scripts.joint_field import JointRiskField
SDIMS = (JointRiskField.load_scene_dims(os.environ["RF_DIMS"])
         if os.environ.get("RF_DIMS") else None)   # recorded per-agent dims sidecar
from scripts.baselines import ttc_score, dsf_score, pora_style_score, ours_peak

DT = float(os.environ.get("RF_DT", "0.08")); M_R = 750.0; S = int(os.environ.get("RF_GRID","48"))
STRIDE = int(os.environ.get("RF_STRIDE","20")); METHODS=["ours","ours_prob","pora","ttc","dsf"]
reg = get_dataset(); c = default_dict(); dev = "cuda" if torch.cuda.is_available() else "cpu"; K=c["seq_len"]
ind = reg["LoaderClass"](root=reg["root"], max_samples=c["maximum_samples"], train_ratio=c["train_ratio"],
    train_batch_size=c["train_batch_size"], test_batch_size=1, missing_rate=c["masked_data_ratio"],
    max_num_cars=c["max_num_cars"], max_empty_frames=c["max_empty_frames"], seq_len=c["seq_len"],
    moving_window=c["seq_len"]*2, sampling_step=c["sampling_step"], should_shuffle=False, include_future=c["include_future"])
site = ind.observation_site_by_scope("all"); bf = reg["boundaries_for_location"]
def build(sl):
    return RiskFlow(seq_len=c["seq_len"], input_dim=c["input_dim"], feature_dim=c["feature_dim"],
        embedding_dim=c["embedding_dim"], hidden_dim=c["hidden_dim"], max_num_cars=c["max_num_cars"],
        num_classes=c["num_classes"], gru_layers=c["gru_layers"], num_heads=c["num_heads"], dropout=c["dropout"],
        norm_rotation=c["norm_rotate"], flow_layers=c["flow_layers"], flow_hidden_dim=c["flow_hidden_dim"],
        coupling_layers=c["coupling_layers"], use_cnf=c["use_cnf"], use_cgmm=c["use_cgmm"], gmm_modes=c["gmm_modes"],
        use_world_model=True, wm_state_dim=c["wm_state_dim"], action_dim=c["action_dim"], scene_level=sl,
        use_map=True, map_size=c["map_size"], map_data_dir=reg["map_data_dir"], map_dataset=reg["map_dataset"]).to(dev).eval()
me=build(False); me.load_state_dict(torch.load(os.environ.get("RF_CKPT_EGO","serialized/riskflow_ind_8.pt"),map_location=dev),strict=False)
mj=build(True);  mj.load_state_dict(torch.load(os.environ.get("RF_CKPT_JOINT","serialized/riskflow_ind_7.pt"),map_location=dev),strict=False)
g1=torch.linspace(0.05,0.95,S);GX,GY=torch.meshgrid(g1,g1,indexing="ij");grid=torch.stack([GX.reshape(-1),GY.reshape(-1)],-1).to(dev)
eng=JointRiskField(me,mj,grid,S,K,dev,min_hist=30)
_z=np.load(os.environ.get("RF_LABELS","conflict_labels.npz"),allow_pickle=True)
L=_z["data"]; _cols=list(_z["cols"]) if "cols" in _z else None
_pi=_cols.index("partner") if _cols else 6      # column moved when ttc/min_gap_mot were added
lab={int(r[0]):(int(r[2]),int(r[3]),int(r[_pi])) for r in L}
sev=[]; meth={m:[] for m in METHODS}
with torch.no_grad():
    for i,b in enumerate(site.test_loader):
        if STRIDE>1 and i%STRIDE!=0: continue
        if i not in lab: continue
        label,onset,partner=lab[i]
        if label!=1 or partner<1: continue
        x=b["input"].to(dev);f=b["feature"].to(dev);vt=b["type"].to(dev);fut=b["future"].to(dev)
        if torch.isnan(x[:,0,-2:,:]).any() or torch.isnan(fut[0,0]).any(): continue
        loc=int(b["locationId"].view(-1)[0]); loc_t=torch.tensor([loc],device=dev); bx=bf(loc)
        scale=torch.tensor([float(bx[0,1]-bx[0,0]),float(bx[1,1]-bx[1,0])],device=dev); sca=scale.cpu().numpy()
        lo=np.array([float(bx[0,0]),float(bx[1,0])])
        neigh=[a for a in range(1,x.shape[1]) if not torch.isnan(x[0,a,-1]).any() and int((~torch.isnan(x[0,a,:,0])).sum())>=30]
        if not neigh or partner not in neigh: continue
        # GT severity = collision energy at min-gap frame (ego vs partner)
        pe=(fut[0,0].cpu().numpy())*sca+lo; pj=(fut[0,partner].cpu().numpy())*sca+lo
        if np.isnan(pe).any() or np.isnan(pj).any(): continue
        d=np.linalg.norm(pe-pj,axis=-1); kf=int(d.argmin())
        ve=np.gradient(pe,DT,axis=0); vj=np.gradient(pj,DT,axis=0)
        rs=float(np.linalg.norm(ve[kf]-vj[kf]))
        if not np.isfinite(rs): continue
        rf,dens,mp=eng.field(x,f,vt,neigh,loc_t,scale,return_dens=True,return_mp=True,dims_m=(SDIMS.get(i) if SDIMS else None),scene_idx=i)
        xn=x[0].cpu().numpy(); p0=xn[0,-1]*sca+lo; v0=(xn[0,-1]-xn[0,-2])*sca/DT
        Po=np.stack([xn[a,-1]*sca+lo for a in neigh]); Vo=np.stack([(xn[a,-1]-xn[a,-2])*sca/DT for a in neigh])
        meth["ours"].append(ours_peak(rf)[0]); meth["ours_prob"].append(float(mp.reshape(-1,K).max()))
        meth["pora"].append(pora_style_score(dens[0],dens,K)[0]); meth["ttc"].append(ttc_score(p0,v0,Po,Vo)[0]); meth["dsf"].append(dsf_score(p0,v0,Po,Vo))
        sev.append(0.5*M_R*rs*rs)
sev=np.array(sev)
print(f"=== severity correlation, {len(sev)} conflict scenes (GT collision energy at near-miss) ===")
for m in METHODS:
    rho,p=spearmanr(meth[m],sev); print(f"RESULT {m:10s} Spearman rho={rho:+.3f}  (p={p:.1e})")
