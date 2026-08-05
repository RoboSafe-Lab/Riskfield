# Riskfield

Code for a **spatio-temporal probabilistic risk field** for autonomous-vehicle safety assessment.
Risk is factorized into occurrence likelihood × physical consequence,

```
Risk_k(g) = P_ego,k(g) · Σ_j M_j,k(g) · C_j,k(g)
```

where `P_ego` is the ego's own occupancy density, `M_j` the probability that the ego's footprint
meets agent *j* there (oriented-box overlap), and `C_j` a kinetic-energy-loss severity — so the
field carries physical units (Joules) rather than an arbitrary score. Occupancies come from two
coupled conditional normalizing flows evaluated exactly on a grid: an ego-marginal model and a
scene-level autoregressive joint conditioned on the ego.

The manuscript lives outside this repo, in `../WorldModel-Informed-RiskField/`.

## Repository layout

| path | contents |
|---|---|
| `model/` | `RiskFlow` (the flow + conditioning), `world_model/`, `decoder/`, `MapEncoder`, `flow/` |
| `datasets/` | **loader code only** — `InD.py`, `AD4CHE.py`, `RounD.py`, `registry.py`, `map_renderer.py`, `xodr_min/` |
| `scripts/` | evaluation, calibration, figure and animation scripts; `*.slurm` cluster jobs |
| `DMOG/` | one-off cluster job scripts; these hardcode cluster paths and are not portable |
| `main.py`, `train.py`, `riskflow_config.py` | training entry point, loop and config |

`datasets/` holds **code, not data.** No dataset and no checkpoint is committed.

## Data

Three loaders, selected by `RF_DATASET`, each reading a flat directory at the repo root:

| `RF_DATASET` | env var | default root | source |
|---|---|---|---|
| `ind` (default) | `RF_DATA_ROOT` | `data/` | inD — intersections, 25 Hz |
| `round` | `RF_ROUND_ROOT` | `data_round/` | rounD — roundabouts, 25 Hz |
| `ad4che` | `RF_AD4CHE_ROOT` | `data_ad4che/` | AD4CHE — congested highway, 30 Hz |

Each root holds the provider's `NN_tracks.csv`, `NN_tracksMeta.csv`, `NN_recordingMeta.csv` and
`NN_background.png`. Download them from the dataset providers; they are gitignored.

`sampling_step=2` means Δt = 2/25 = **0.08 s** on inD and rounD, and 2/30 = **0.0667 s** on AD4CHE
(`RF_DT`).

## Checkpoints

Not committed; they live in `serialized/` on the cluster. The `riskflow_ind_N` numbering is
historical, so here is what each actually contains — recoverable otherwise only by inspecting
parameter names:

| checkpoint | world model | map | level | role |
|---|---|---|---|---|
| `ind_0`, `ind_1` | naive rollout | – | single | **not loadable** — predate the residual head |
| `ind_2` | none | – | single | the no-world-model ablation control |
| `ind_3/4/5` | residual + dyn | – | single | residual world model (3 checkpoints of one variant) |
| `ind_6` | residual + dyn | – | scene | multi-agent action-conditioned |
| `ind_7` | residual + dyn | ✓ | scene | **deployed** ego-conditioned joint |
| `ind_8` | residual + dyn | ✓ | single | **deployed** ego marginal |

The risk field is the `ind_8` × `ind_7` pair. `ad4che_ego`/`ad4che_joint` and
`round_ego`/`round_joint` are the counterparts on the other datasets; `_vh` suffixes add the trained
velocity head, `_mc` are multi-checkpoint variants.

`use_map` and `use_world_model` **must match the checkpoint** — a mismatch is silently tolerated by
`load_state_dict(strict=False)` and leaves that branch randomly initialized. Scripts that guard
against this abort with an explicit message.

## Reproducing the results

Set `RF_DATASET` and the checkpoint env vars, then:

| script | produces |
|---|---|
| `main.py` | trains a model (`RF_SCENE_LEVEL=0` ego, `=1` joint) |
| `scripts/eval_report_all.py` | held-out NLL / CRPS / minADE / minFDE / RMSE for one checkpoint |
| `scripts/make_pet_labels.py` | PET conflict labels (`conflict_labels*.npz`) |
| `scripts/calibrate_joint.py` | `risk_calibration.json` — the global risk scale and `T_critical` |
| `scripts/eval_conflict.py` | conflict-detection comparison (ours / PORA-style / TTC / DSF) |
| `scripts/severity_eval2.py` | severity correlation against ground-truth collision energy |
| `scripts/cf_table.py` | counterfactual objective *J(A)* per ego maneuver |
| `scripts/wm_fidelity.py` | world-model and counterfactual fidelity |
| `scripts/coherence_dmog.slurm` | temporal-coherence measurement across model variants |
| `scripts/qualitative_map.py`, `animate_joint.py`, `animate_horizon.py` | figures and animations |

`risk_calibration.json` is committed because the figure scripts read it at runtime;
`critical_scenes_*.csv` record which scenes the qualitative figures use.

## Cluster workflow

Training and evaluation run on a Slurm cluster:

```bash
bash scripts/deploy_dmog.sh code     # push code (excludes datasets, gifs, checkpoints)
bash scripts/deploy_dmog.sh data     # push inD once (~1.8 GB)
sbatch scripts/train_dmog.slurm      # or eval_ / calibrate_joint_ / ablation_ / coherence_
```

Edit `KEY`, `HOST` and `REMOTE` at the top of `scripts/deploy_dmog.sh` for your own account.

## Environment

Python 3, PyTorch, numpy/scipy, pandas, matplotlib, wandb. See
`scripts/requirements_dmog.txt` for the cluster environment.
