"""AD4CHE loader smoke: batch shapes/keys match InD, then a registration render."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.AD4CHE import AD4CHE

ROOT = os.environ.get("RF_AD4CHE_ROOT", "datasets/AD4CHE")
ind = AD4CHE(root=ROOT, max_samples=200, train_ratio=0.75, train_batch_size=8,
             test_batch_size=1, missing_rate=0.0, max_num_cars=8, max_empty_frames=0,
             seq_len=50, moving_window=100, sampling_step=3, should_shuffle=False,
             include_future=True)
site = ind.observation_site_by_scope("all")
b = next(iter(site.test_loader))
print("keys:", sorted(b.keys()))
print("input", tuple(b["input"].shape), "future", tuple(b["future"].shape),
      "type", tuple(b["type"].shape), "loc", int(b["locationId"].view(-1)[0]))
print("input nan-frac", round(float(b["input"].isnan().float().mean()), 3),
      "feat range", round(float(b["feature"][~b["feature"].isnan()].min()), 2),
      round(float(b["feature"][~b["feature"].isnan()].max()), 2))
