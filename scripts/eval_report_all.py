"""Standalone, authoritative evaluation of a saved checkpoint.

Prints metrics directly to stdout (no wandb) so values are never collapsed.
Configured via environment variables:

  RF_CKPT             path to .pt checkpoint   (default serialized/riskflow_ind_1.pt)
  RF_SITE_SCOPE       all|08|loc1..loc4        (default all)
  RF_USE_WORLD_MODEL  0/1                      (default 1)

Run from the project root:  python scripts/eval_report_all.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from datasets.InD import InD  # noqa: E402
from model.RiskFlow import RiskFlow  # noqa: E402
from evaluate import evaluate  # noqa: E402
from riskflow_config import default_dict  # noqa: E402

ckpt = os.environ.get("RF_CKPT", "serialized/riskflow_ind_1.pt")
scope = os.environ.get("RF_SITE_SCOPE", "all")
uwm = os.environ.get("RF_USE_WORLD_MODEL", "1").strip().lower() in ("1", "true", "yes")
# RF_SCENE_LEVEL=1 for ind_6+ (scene-level AR model); default False for ind_3..5.
scene_level = os.environ.get("RF_SCENE_LEVEL", "0").strip().lower() in ("1", "true", "yes")

c = default_dict()
ind = InD(
    root="data", max_samples=c["maximum_samples"], train_ratio=c["train_ratio"],
    train_batch_size=c["train_batch_size"], test_batch_size=c["test_batch_size"],
    missing_rate=c["masked_data_ratio"], max_num_cars=c["max_num_cars"],
    max_empty_frames=c["max_empty_frames"], seq_len=c["seq_len"],
    moving_window=c["seq_len"] * 2, sampling_step=c["sampling_step"],
    should_shuffle=False, include_future=c["include_future"],
)
site = ind.observation_site_by_scope(scope)
dev = "cuda" if torch.cuda.is_available() else "cpu"

m = RiskFlow(
    seq_len=c["seq_len"], input_dim=c["input_dim"], feature_dim=c["feature_dim"],
    embedding_dim=c["embedding_dim"], hidden_dim=c["hidden_dim"],
    max_num_cars=c["max_num_cars"], num_classes=c["num_classes"],
    gru_layers=c["gru_layers"], num_heads=c["num_heads"], dropout=c["dropout"],
    norm_rotation=c["norm_rotate"], flow_layers=c["flow_layers"],
    flow_hidden_dim=c["flow_hidden_dim"], coupling_layers=c["coupling_layers"],
    use_cnf=c["use_cnf"], use_cgmm=c["use_cgmm"], gmm_modes=c["gmm_modes"],
    use_world_model=uwm, wm_state_dim=c["wm_state_dim"], action_dim=c["action_dim"],
    scene_level=scene_level,
).to(dev)
# strict=False: the new MultiEncoder carries both legacy (mha) and
# scene-level (self_attn) sub-modules; an older checkpoint only has the keys
# its training used. Unused sub-modules stay at init and are not exercised.
_miss, _unexp = m.load_state_dict(torch.load(ckpt, map_location=dev), strict=False)
print(f"RESULT load: missing={len(_miss)} unexpected={len(_unexp)}")

n = sum(p.numel() for p in m.parameters() if p.requires_grad)
rmse, crps, ade, fde, nll = evaluate(
    observation_site=site, model=m, num_samples=c["evaluation_samples"], device=dev
)
print("=" * 60)
print(f"RESULT ckpt={ckpt} scope={scope} use_world_model={uwm}")
print(f"RESULT PARAMS={n}")
print(
    f"RESULT RMSE={rmse:.4f} CRPS={crps:.4f} minADE={ade:.4f} "
    f"minFDE={fde:.4f} NLL={nll:.4f}"
)
print("=" * 60)
