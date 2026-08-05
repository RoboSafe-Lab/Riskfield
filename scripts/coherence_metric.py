"""Temporal-coherence metric (the result the world-model claim rests on).

Reviewer 2's concern: the stack of per-step densities need not correspond to
ANY single coherent trajectory. We test this directly.

Mechanism (paper Sec. III-D): hold ONE base latent code z fixed and decode it
through the flow at every horizon step under that step's conditioning,
producing a "mode-consistent" trajectory. If the conditioning evolves
coherently (world model), this trajectory is smooth and physically plausible;
if each step is an independent function of a raw time index (no-WM), the
fixed-z decode jumps around.

We report, on fixed-z decoded trajectories (normalized units, so the WM vs
no-WM comparison is fair):

  ACCEL : mean ||p_{k+1}-2p_k+p_{k-1}||  (lower = smoother = more coherent)
  JERK  : mean ||third difference||
  DIRCOS: mean cos angle between consecutive displacement vectors
          (1 = perfectly consistent heading, 0 = teleporting)

Ground-truth future trajectories give the reference (real driving is smooth).
The model whose decoded trajectories are closest to GT is more temporally
coherent. Configured by env: RF_CKPT, RF_USE_WORLD_MODEL, RF_MAX_SCENES.

Run from project root:  python scripts/coherence_metric.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from datasets.InD import InD  # noqa: E402
from model.RiskFlow import RiskFlow  # noqa: E402
from riskflow_config import default_dict  # noqa: E402

ckpt = os.environ.get("RF_CKPT", "serialized/riskflow_ind_1.pt")
uwm = os.environ.get("RF_USE_WORLD_MODEL", "1").strip().lower() in ("1", "true", "yes")
max_scenes = int(os.environ.get("RF_MAX_SCENES", "1500"))
R = int(os.environ.get("RF_Z_SAMPLES", "16"))  # fixed-z trajectories per scene
# use_map MUST match the checkpoint. The ablation family (riskflow_ind_0..6,
# which the no-world-model control belongs to) was trained without a map
# encoder; the deployed models (ind_7/ind_8) carry one. RiskFlow defaults
# use_map=False, so loading ind_8 without this would drop its map weights as
# merely "unexpected" and measure a model that never sees the road -- silently,
# since load_state_dict(strict=False) reports nothing. Refused below.
umap = os.environ.get("RF_USE_MAP", "0").strip().lower() in ("1", "true", "yes")

c = default_dict()
ind = InD(
    root="data", max_samples=c["maximum_samples"], train_ratio=c["train_ratio"],
    train_batch_size=c["train_batch_size"], test_batch_size=1,
    missing_rate=c["masked_data_ratio"], max_num_cars=c["max_num_cars"],
    max_empty_frames=c["max_empty_frames"], seq_len=c["seq_len"],
    moving_window=c["seq_len"] * 2, sampling_step=c["sampling_step"],
    should_shuffle=False, include_future=c["include_future"],
)
site = ind.observation_site_by_scope("all")
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
    scene_level=os.environ.get("RF_SCENE_LEVEL", "0").strip().lower() in ("1","true","yes"),
    use_map=umap, map_size=c["map_size"], map_data_dir="data", map_dataset="ind",
).to(dev)
_res = m.load_state_dict(torch.load(ckpt, map_location=dev), strict=False)
_missing = list(getattr(_res, "missing_keys", []) or [])
_unexpected = list(getattr(_res, "unexpected_keys", []) or [])
# strict=False is deliberate here (the encoder carries both the legacy mha and
# the scene-level self_attn), but a MAP or WORLD-MODEL mismatch is never benign:
# it means the built model and the checkpoint disagree about which branches
# exist, and the un-matched branch runs at its random initialization.
for _tag, _keys in (("missing from checkpoint", _missing),
                    ("present in checkpoint but unused", _unexpected)):
    _bad = [k for k in _keys if "map" in k.lower() or k.startswith("world_model")]
    if _bad:
        raise SystemExit(
            f"ABORT {ckpt}: {len(_bad)} map/world-model params {_tag} "
            f"(e.g. {_bad[:3]}). Set RF_USE_MAP / RF_USE_WORLD_MODEL to match "
            f"this checkpoint -- otherwise the measurement is of a partly "
            f"randomly-initialized model."
        )
print(f"loaded {ckpt}: use_world_model={uwm} use_map={umap} "
      f"({len(_missing)} missing, {len(_unexpected)} unexpected, none structural)",
      flush=True)
m.eval()


def roughness(p):
    """p: (..., K, 2) -> (accel, jerk, dircos) scalars over the ... batch."""
    eps = 1e-8
    d1 = p[..., 1:, :] - p[..., :-1, :]                       # displacement
    acc = d1[..., 1:, :] - d1[..., :-1, :]                    # 2nd diff
    jrk = acc[..., 1:, :] - acc[..., :-1, :]                  # 3rd diff
    a = acc.norm(dim=-1).mean()
    j = jrk.norm(dim=-1).mean()
    u = d1[..., 1:, :]
    v = d1[..., :-1, :]
    cos = (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1) + eps)
    return a.item(), j.item(), cos.mean().item()


# The fixed-z codes are drawn from the global RNG, so without this the reported
# figure moves between runs. It is a paper number: pin it.
torch.manual_seed(int(os.environ.get("RF_SEED", "0")))

sum_a = sum_j = sum_c = 0.0
g_a = g_j = g_c = 0.0
n = 0
with torch.no_grad():
    for batch in site.test_loader:
        if n >= max_scenes:
            break
        x = batch["input"].to(dev)
        feat = batch["feature"].to(dev)
        vt = batch["type"].to(dev)
        tgt = batch["target"].to(dev)            # (1,K,2) GT future (normalized)
        K = tgt.shape[1]

        if m.norm_rotation:                       # config: norm_rotate=False
            x, ang = m._normalize_rotation(x)
            feat = m._rotate_features(feat, ang)
        if m.scene_level:
            agent_emb, _ = m.encoder(None, torch.cat([x, feat], dim=-1), vt, per_agent=True)
            emb = agent_emb[:, 0]                  # ego (B,E)
        else:
            emb, _ = m.encoder(None, torch.cat([x, feat], dim=-1), vt, per_agent=False)
        cond = m._flow_condition(emb, K, None)    # (1,K,E) WM or (1,E) no-WM
        if cond.dim() == 3:
            cond = cond.expand(R, K, cond.shape[-1])
        else:
            cond = cond.expand(R, cond.shape[-1])

        # ONE base code per trajectory, held fixed across all K steps.
        z0 = torch.randn(R, 1, m.input_dim, device=dev).expand(R, K, m.input_dim)
        traj, _ = m.flow(z0.contiguous(), cond, reverse=True, sampling_frequency=1)

        a, j, cc = roughness(traj)
        ga, gj, gc = roughness(tgt)
        sum_a += a; sum_j += j; sum_c += cc
        g_a += ga; g_j += gj; g_c += gc
        n += 1

inv = 1.0 / max(n, 1)
print("=" * 64)
print(f"RESULT ckpt={ckpt} use_world_model={uwm} scenes={n} z_per_scene={R}")
print(f"RESULT MODEL  accel={sum_a*inv:.5f} jerk={sum_j*inv:.5f} dircos={sum_c*inv:.4f}")
print(f"RESULT GTREF  accel={g_a*inv:.5f} jerk={g_j*inv:.5f} dircos={g_c*inv:.4f}")
print(f"RESULT GAP    accel_ratio(model/GT)={(sum_a/ max(g_a,1e-9)):.3f} "
      f"(closer to 1 = more coherent; >>1 = incoherent/jittery)")
print("=" * 64)
