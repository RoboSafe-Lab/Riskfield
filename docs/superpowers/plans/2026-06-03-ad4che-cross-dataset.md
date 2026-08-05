# AD4CHE Cross-Dataset Ablation — Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans (inline) to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.
>
> **Environment note:** This repo is **not a git repo** and has **no pytest suite**. So "tests" are **cluster smoke-scripts with expected output**, and "commit" steps are replaced by **CHECKPOINT: verify** gates (scp the changed file to the cluster + run the smoke). All training/eval runs on the DMOG cluster (`ssh dmog`, conda env `riskflow`, project dir `/users/cw3005/riskfield`). InD checkpoints are backed up at `backups/ind_2026-06-03/`.

**Goal:** Retrain the two-model risk field on AD4CHE (congested highway) and report prediction quality + conflict-detection efficacy, demonstrating cross-dataset generalization.

**Architecture:** A native `datasets/AD4CHE.py` subclasses `datasets/InD.py`, adapting AD4CHE's CSV schema to InD's and using **per-scene** normalization (scenes 14–24 = locations) + per-scene maps. A small `datasets/registry.py` makes the shared scripts dataset-aware (`RF_DATASET=ind|ad4che`). Then retrain `ind_8`/`ind_7`, recalibrate, label, and run the existing eval harness on AD4CHE.

**Tech Stack:** Python, PyTorch, pandas/numpy, scipy; existing `RiskFlow`, `JointRiskField`, `MapEncoder`, `conflict_labels.py`, `eval_conflict.py`, `wm_fidelity.py`, `calibrate_joint.py`.

**Spec:** `docs/superpowers/specs/2026-06-03-ad4che-ablation-design.md`

---

## File Structure

- **Create** `datasets/AD4CHE.py` — AD4CHE loader (subclass of `InD`); schema adapter + per-scene boundaries + scene→recordings map.
- **Create** `datasets/registry.py` — `get_dataset(name)` → `(LoaderClass, boundaries_for_location, feature_boundaries, veh_dims, location_recordings, map_glob)`.
- **Modify** `model/MapEncoder.py` — `build_location_maps` already arg-driven; add a `map_path_fn` so AD4CHE per-scene `maps/NN.jpg` can be used (InD uses `{rec}_background.png`).
- **Modify** `model/RiskFlow.py` — the `use_map` branch hardcodes InD imports; add `map_dataset` param selecting InD vs AD4CHE sources.
- **Modify** `riskflow_config.py` + `main.py` — add `RF_DATASET` env override (like `RF_SCENE_LEVEL`).
- **Modify** `scripts/joint_field.py` — `VEH_LW` becomes dataset-aware (AD4CHE adds `bus`).
- **Modify** `scripts/{conflict_labels,eval_conflict,calibrate_joint,wm_fidelity}.py` — build the loader + boundaries via `registry.get_dataset(os.environ.get("RF_DATASET","ind"))` instead of hardcoded `InD`/`boundaries_for_location`.
- **Create** `scripts/smoke_ad4che.py` — loader + registration-render smoke (the key early guard).

---

## Task 1: AD4CHE per-scene boundaries + schema constants

**Files:**
- Create: `datasets/AD4CHE.py` (boundaries + constants only this task)

- [ ] **Step 1: Write the boundaries computation as a runnable check**

Create `datasets/AD4CHE.py` with the scene list, class map, and a function that computes each scene's spatial box by pooling the metric `x,y` extents over all its recordings (with a small pad), plus a hardcoded fallback once computed:

```python
import os, glob, math
import numpy as np
import pandas as pd

AD4CHE_ROOT_DEFAULT = "datasets/AD4CHE"            # local; cluster: data_ad4che
SCENES = [str(s) for s in range(14, 25)]           # 14..24 = "locations"
# AD4CHE classes -> model class ids (reuse InD's 2-class FiLM head: car=0, big=1)
AD4CHE_CLASS_TO_ID = {"car": 0, "truck": 1, "bus": 1}
# heading feature is degrees in [0,360]; velocities/accels widened for highway.
AD4CHE_FEATURE_BOUNDARIES = np.array([[0, 360], [-40, 40], [-40, 40], [-6, 6], [-6, 6]])

def _recording_dirs(root, scene):
    return sorted(glob.glob(os.path.join(root, scene, "DJI_*")))

def compute_scene_boundaries(root=AD4CHE_ROOT_DEFAULT, pad=3.0):
    """Per-scene spatial box [[xlo,xhi],[ylo,yhi]] pooled over the scene's
    recordings (positions are already metric in tracks_pixel.csv)."""
    out = {}
    for sc in SCENES:
        xs_lo = xs_hi = ys_lo = ys_hi = None
        for rd in _recording_dirs(root, sc):
            f = glob.glob(os.path.join(rd, "*_tracks_pixel.csv"))
            if not f:
                continue
            df = pd.read_csv(f[0], usecols=["x", "y"])
            xlo, xhi = float(df["x"].min()), float(df["x"].max())
            ylo, yhi = float(df["y"].min()), float(df["y"].max())
            xs_lo = xlo if xs_lo is None else min(xs_lo, xlo)
            xs_hi = xhi if xs_hi is None else max(xs_hi, xhi)
            ys_lo = ylo if ys_lo is None else min(ys_lo, ylo)
            ys_hi = yhi if ys_hi is None else max(ys_hi, yhi)
        if xs_lo is None:
            continue
        out[int(sc)] = np.array([[xs_lo - pad, xs_hi + pad], [ys_lo - pad, ys_hi + pad]])
    return out

if __name__ == "__main__":   # smoke
    b = compute_scene_boundaries(os.environ.get("RF_AD4CHE_ROOT", AD4CHE_ROOT_DEFAULT))
    for k in sorted(b):
        (xlo, xhi), (ylo, yhi) = b[k]
        print(f"scene {k}: x[{xlo:.0f},{xhi:.0f}] ({xhi-xlo:.0f}m)  y[{ylo:.0f},{yhi:.0f}] ({yhi-ylo:.0f}m)")
```

- [ ] **Step 2: Run the boundaries smoke**

Run (local): `cd E:/Paper/riskfield/Riskfield && python datasets/AD4CHE.py`
Expected: 11 lines `scene 14..24` with x-ranges ~100–150 m and y-ranges ~25–35 m (long-thin highways). Scene 14 should read ~`x[-75,73] (148m) y[-20,14] (34m)`.

- [ ] **Step 3: CHECKPOINT** — boundaries look like highways (x ≫ y). If any scene missing, check `RF_AD4CHE_ROOT`.

---

## Task 2: AD4CHE loader (subclass of InD)

**Files:**
- Modify: `datasets/AD4CHE.py` (add `class AD4CHE(InD)`)
- Reference: `datasets/InD.py` (`_load_and_clean_data`, `_parse`, `_create_single_sample`, `_collate_results`, `class_to_id`)

The loader reuses InD's windowing by (a) overriding `_load_and_clean_data` to read AD4CHE CSVs and **rename columns to InD's schema**, and (b) overriding the per-location box + class map + collate to drop the InD-only `orthoPxToMeter`.

- [ ] **Step 1: Add the AD4CHE class**

Append to `datasets/AD4CHE.py`:

```python
import numpy as np
from datasets.InD import InD, normalize

class AD4CHE(InD):
    """AD4CHE loader: subclasses InD, adapts the tracks_pixel/tracksMeta schema and
    uses per-scene (14..24) normalization + the model's 2-class head."""
    def __init__(self, root="data_ad4che", **kw):
        super().__init__(root=root, **kw)
        self.target_classes = ["car", "truck", "bus"]
        self.input_cols = ["xCenter", "yCenter"]      # after rename
        self.feature_cols = ["heading", "xVelocity", "yVelocity",
                             "xAcceleration", "yAcceleration"]
        self._boxes = compute_scene_boundaries(root)
        # scene -> list of "site" ids; a site id encodes scene+recording dir name
        self.LOCATION_RECORDINGS = {
            int(sc): [f"{sc}/{os.path.basename(rd)}" for rd in _recording_dirs(root, sc)]
            for sc in SCENES if _recording_dirs(root, sc)
        }

    def boundaries_for_location(self, loc):
        return self._boxes[int(loc)]

    def _load_and_clean_data(self, site):
        """site = 'SCENE/DJI_xxxx'. Returns (meta, tracks, tracks_meta) in InD schema."""
        scene, rec = site.split("/")
        rd = os.path.join(self.root, scene, rec)
        prefix = glob.glob(os.path.join(rd, "*_tracks_pixel.csv"))[0].rsplit("_tracks_pixel.csv", 1)[0]
        meta = pd.read_csv(prefix + "_recordingMeta.csv")
        tracks = pd.read_csv(prefix + "_tracks_pixel.csv")
        tmeta = pd.read_csv(prefix + "_tracksMeta.csv")
        # --- rename to InD schema ---
        tracks = tracks.rename(columns={"id": "trackId", "x": "xCenter", "y": "yCenter"})
        tracks["heading"] = np.degrees(tracks["orientation"]) % 360.0
        # InD's consistency check uses trackLifetime = frames since track start:
        tracks["trackLifetime"] = tracks.groupby("trackId").cumcount()
        tmeta = tmeta.rename(columns={"id": "trackId"})
        # InD `_parse` reads locationId from meta; inject the SCENE number.
        meta["locationId"] = int(scene)
        meta["orthoPxToMeter"] = 1.0          # positions already metric; unused downstream
        tracks = tracks.drop_duplicates(subset=["trackId", "frame"])
        return meta, tracks, tmeta

    def _collate_results(self, samples, meta, spatial_box):
        res = super()._collate_results(samples, meta, spatial_box)
        res.pop("orthoPxToMeter", None)       # not meaningful for AD4CHE
        return res
```

- [ ] **Step 2: Make InD `_parse` use the instance boundary fn**

In `datasets/InD.py`, line ~383 currently calls module-level `boundaries_for_location(location_id)`. Change to use an overridable hook so the subclass box is honored:

```python
# datasets/InD.py  (in _parse, replace the spatial_box line)
spatial_box = self.boundaries_for_location(location_id)
```

And add to `class InD` (so InD behavior is unchanged):

```python
def boundaries_for_location(self, location_id):
    return boundaries_for_location(location_id)   # module-level InD default
```

- [ ] **Step 3: Run the loader smoke**

Create/append to `scripts/smoke_ad4che.py`:

```python
import os, sys; sys.path.insert(0, ".")
from datasets.AD4CHE import AD4CHE
ind = AD4CHE(root=os.environ.get("RF_AD4CHE_ROOT", "datasets/AD4CHE"),
             max_samples=200, train_ratio=0.75, train_batch_size=8, test_batch_size=1,
             missing_rate=0.0, max_num_cars=8, max_empty_frames=0, seq_len=50,
             moving_window=100, sampling_step=3, should_shuffle=False, include_future=True)
site = ind.observation_site_by_scope("all")
b = next(iter(site.test_loader))
print("keys:", sorted(b.keys()))
print("input", tuple(b["input"].shape), "future", tuple(b["future"].shape),
      "type", tuple(b["type"].shape), "loc", int(b["locationId"].view(-1)[0]))
print("input nan-frac", float(b["input"].isnan().float().mean()))
```

Run (local): `cd E:/Paper/riskfield/Riskfield && python scripts/smoke_ad4che.py`
Expected: `keys: ['feature','future','input','locationId','target','type','trackId','startFrame']`, `input (1,8,50,2) future (1,8,50,2) type (1,8) loc <14..24>`, nan-frac < 0.5.

- [ ] **Step 4: CHECKPOINT** — shapes/keys match InD's; `locationId` in 14..24. If `class_to_id` warns "unknown class", confirm `AD4CHE_CLASS_TO_ID` is wired (Task 5 makes `class_to_id` dataset-aware; for this smoke a transient warning is OK).

**Note (Δt):** AD4CHE is 30 fps. With `sampling_step=3` → Δt = 0.10 s (InD used 0.08 s). Pick `sampling_step=2` (Δt≈0.067) or `3` (0.10) so the K=50 horizon ≈ InD's ~4 s; document the choice. Default this plan: `sampling_step=3` (5 s horizon, closest to InD's coverage).

---

## Task 3: Registration smoke render (the key correctness guard)

**Files:**
- Modify: `scripts/smoke_ad4che.py` (add a render of GT boxes on the scene map)

- [ ] **Step 1: Add the render**

```python
# append to scripts/smoke_ad4che.py
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from datasets.AD4CHE import compute_scene_boundaries
loc = int(b["locationId"].view(-1)[0])
(xlo, xhi), (ylo, yhi) = ind._boxes[loc]
img = mpimg.imread(os.path.join(ind.root, "maps", f"{loc}.jpg"))
# de-normalize GT positions back to metres and overlay
inp = b["input"][0].numpy()                     # (N,T,2) normalized
pos = inp * np.array([xhi-xlo, yhi-ylo]) + np.array([xlo, ylo])
plt.figure(figsize=(10,3)); plt.imshow(img, extent=[xlo, xhi, ylo, yhi], origin="upper")
for a in range(pos.shape[0]):
    p = pos[a][~np.isnan(pos[a]).any(-1)]
    if len(p): plt.plot(p[:,0], p[:,1], ".-", ms=2)
plt.title(f"scene {loc} GT tracks on map"); plt.savefig("smoke_ad4che_reg.png", dpi=110); print("saved smoke_ad4che_reg.png")
```

- [ ] **Step 2: Run + inspect**

Run (local): `python scripts/smoke_ad4che.py` then open `smoke_ad4che_reg.png`.
Expected: agent tracks lie **on the highway lanes** of `maps/<loc>.jpg`. If they're rotated/flipped/off-road, the y-axis origin or a sign is wrong — fix the `extent`/`origin` (try `origin="lower"` and/or negating y) until tracks sit on the road. **Do not proceed past this gate until registration is correct.**

- [ ] **Step 3: CHECKPOINT** — registration verified visually. This is the single most important correctness check; map conditioning and the risk field both depend on it.

---

## Task 4: Per-scene map rasters for the encoder

**Files:**
- Modify: `model/MapEncoder.py` (`build_location_maps` gains a `map_path_fn` arg)

- [ ] **Step 1: Parameterize the map path**

In `model/MapEncoder.py::build_location_maps`, the InD path is `f"{rec}_background.png"` and uses `orthoPxToMeter`. AD4CHE uses one `maps/{scene}.jpg` per location and positions are already metric. Add a `map_path_fn(loc)` and a `meters_per_px` arg so both work:

```python
def build_location_maps(data_dir, location_recordings, boundaries, smap=64,
                        map_path_fn=None, metric_positions=False):
    ...
    for loc, recs in location_recordings.items():
        if map_path_fn is not None:
            img_path = map_path_fn(loc)                  # AD4CHE: maps/<loc>.jpg
            img = mpimg.imread(img_path)[..., :3].astype(np.float32)
            if img.max() > 1.5: img = img / 255.0
            (xlo, xhi), (ylo, yhi) = boundaries[loc]
            # map the [xlo,xhi]x[ylo,yhi] box directly onto the image pixels
            H, W = img.shape[:2]
            u = (np.arange(smap)+0.5)/smap
            UU, VV = np.meshgrid(u, u, indexing="ij")
            col = (UU*(xhi-xlo)+xlo - xlo)/(xhi-xlo) * (W-1)
            row = (1-(VV*(yhi-ylo)+ylo - ylo)/(yhi-ylo)) * (H-1)   # adjust per Task 3 finding
            out = np.zeros((3, smap, smap), np.float32)
            for c in range(3):
                out[c] = map_coordinates(img[..., c], [row.ravel(), col.ravel()], order=1,
                                         mode="constant", cval=0.0).reshape(smap, smap)
            maps[loc] = out
            continue
        # ... existing InD orthophoto path unchanged ...
```

**Important:** the `row`/`col` mapping MUST match the registration found correct in Task 3 (same axis orientation). Reuse the exact transform that put tracks on the road.

- [ ] **Step 2: Smoke the map build**

Append to `scripts/smoke_ad4che.py`:

```python
from model.MapEncoder import build_location_maps
mp = build_location_maps(ind.root, ind.LOCATION_RECORDINGS, ind._boxes, smap=64,
        map_path_fn=lambda L: os.path.join(ind.root, "maps", f"{L}.jpg"), metric_positions=True)
print("maps:", {k: (round(float(v.mean()),3)) for k,v in mp.items()})
```

Run (local): `python scripts/smoke_ad4che.py`
Expected: every scene 14..24 maps to a nonzero mean (~0.3–0.7), none all-zero.

- [ ] **Step 3: CHECKPOINT** — all scene rasters nonzero. Optionally `imsave` one to confirm it shows road.

---

## Task 5: Dataset registry + make scripts/model dataset-aware

**Files:**
- Create: `datasets/registry.py`
- Modify: `model/RiskFlow.py` (`map_dataset` param), `riskflow_config.py`+`main.py` (`RF_DATASET`), `scripts/joint_field.py` (`VEH_LW`), and the four eval scripts' loader/boundary construction.

- [ ] **Step 1: Write the registry**

```python
# datasets/registry.py
import os
def get_dataset(name=None):
    name = (name or os.environ.get("RF_DATASET", "ind")).lower()
    if name == "ad4che":
        from datasets.AD4CHE import AD4CHE, AD4CHE_FEATURE_BOUNDARIES, AD4CHE_CLASS_TO_ID
        root = os.environ.get("RF_AD4CHE_ROOT", "data_ad4che")
        loader = lambda **kw: AD4CHE(root=root, **kw)
        bf = lambda L, _r=root: __import__("datasets.AD4CHE", fromlist=["x"]).compute_scene_boundaries(_r)[int(L)]
        veh = {0: (4.5, 1.9), 1: (12.0, 2.6)}     # car, truck/bus (highway)
        return dict(name="ad4che", loader=loader, boundaries_for_location=bf,
                    feature_boundaries=AD4CHE_FEATURE_BOUNDARIES, veh_lw=veh, scope="all")
    from datasets.InD import InD, feature_boundaries, boundaries_for_location
    loader = lambda **kw: InD(root=os.environ.get("RF_DATA_ROOT", "data"), **kw)
    veh = {0: (4.5, 1.9), 1: (10.0, 2.6), 2: (1.8, 0.6), 3: (0.7, 0.7)}
    return dict(name="ind", loader=loader, boundaries_for_location=boundaries_for_location,
                feature_boundaries=feature_boundaries, veh_lw=veh, scope="all")
```

- [ ] **Step 2: `RiskFlow` map source by dataset**

In `model/RiskFlow.py` `__init__`, the `use_map` branch imports `InD.LOCATION_RECORDINGS`/`LOCATION_SPATIAL_BOUNDARIES`/builds `{rec}_background.png` maps. Add `map_dataset="ind"`; when `"ad4che"`, build `loc_maps` from `AD4CHE` recordings + `maps/<loc>.jpg` via the Task-4 `map_path_fn`. Keep InD path default so existing checkpoints load unchanged.

```python
# RiskFlow.__init__ (use_map branch), pseudocode of the change:
if map_dataset == "ad4che":
    from datasets.AD4CHE import AD4CHE, SCENES, compute_scene_boundaries, _recording_dirs
    root = map_data_dir
    boxes = compute_scene_boundaries(root)
    locrecs = {int(s): [s] for s in SCENES if _recording_dirs(root, s)}
    loc_maps = build_location_maps(root, locrecs, boxes, smap=map_size,
                  map_path_fn=lambda L: os.path.join(root, "maps", f"{L}.jpg"),
                  metric_positions=True)
    keys = sorted(loc_maps)            # scene ids 14..24 -> buffer rows 0..K-1
else:
    ... existing InD build ...
self._map_loc_index = {loc: i for i, loc in enumerate(keys)}   # map locationId -> buffer row
```

Update `_map_emb` to map `location_id` through `self._map_loc_index` (InD uses ids 1..4 directly; AD4CHE uses 14..24 → remap to buffer rows). Add `map_dataset` to the `main.py` constructor call, sourced from `RF_DATASET`.

- [ ] **Step 3: `RF_DATASET` in main/config**

In `main.py`, after the `RF_SCENE_LEVEL` block, add:
```python
_dataset = os.environ.get("RF_DATASET", getattr(run.config, "dataset", "ind"))
```
Build the dataset via `from datasets.registry import get_dataset; reg = get_dataset(_dataset); ind = reg["loader"](...)` (same kwargs as today). Pass `map_dataset=_dataset`, `map_data_dir=("data_ad4che" if _dataset=="ad4che" else "data")` to `RiskFlow`.

- [ ] **Step 4: `joint_field.VEH_LW` dataset-aware**

In `scripts/joint_field.py`, replace the module-level `VEH_LW` constant usage with a value read from `os.environ.get("RF_DATASET")` via `registry.get_dataset(...)["veh_lw"]` at import (one line), so highway truck/bus dims apply on AD4CHE.

- [ ] **Step 5: Eval scripts build loader via registry**

In `scripts/{conflict_labels,eval_conflict,calibrate_joint,wm_fidelity}.py`, replace `from datasets.InD import InD, boundaries_for_location` + `InD(root="data", ...)` with:
```python
from datasets.registry import get_dataset
_reg = get_dataset()                          # RF_DATASET env
ind = _reg["loader"](max_samples=c["maximum_samples"], ...)   # same kwargs
boundaries_for_location = _reg["boundaries_for_location"]
```
And the ego/joint checkpoints become env-driven: `RF_CKPT_EGO`/`RF_CKPT_JOINT` already exist — just point them at the AD4CHE checkpoints at run time.

- [ ] **Step 6: CHECKPOINT — InD regression**

Run (cluster): the existing InD conflict eval on a small slice to confirm no regression:
`RF_DATASET=ind RF_MAX_SCENES=60 RF_GRID=48 RF_STRIDE=20 python scripts/eval_conflict.py`
Expected: runs to completion; `ours`/`ttc` AUROCs within ~0.02 of the prior 60-scene numbers. (Guards the refactor.)

---

## Task 6: Deploy data + train ind_8 & ind_7 on AD4CHE

**Files:** none new (config/env only)

- [ ] **Step 1: Stage AD4CHE on the cluster**

Tar `datasets/AD4CHE` → cluster `data_ad4che/` (keep `maps/` + `<scene>/<DJI>/*tracks_pixel.csv,*tracksMeta.csv,*recordingMeta.csv`):
```bash
tar -czf - -C datasets/AD4CHE . | ssh dmog "mkdir -p ~/riskfield/data_ad4che && tar -xzf - -C ~/riskfield/data_ad4che && echo staged"
```
Expected: `staged`; `ssh dmog "ls ~/riskfield/data_ad4che/maps | wc -l"` → 11.

- [ ] **Step 2: Train ind_8 (single-target + map) on AD4CHE**

`sbatch --export=ALL,RF_DATASET=ad4che,RF_AD4CHE_ROOT=data_ad4che,RF_SCENE_LEVEL=0 scripts/train_dmog.slurm`
Expected: job runs; `grep '^epoch:' <log>` shows decreasing loss then plateau. Note the new checkpoint number (`serialized/riskflow_ind_<n>.pt`).

- [ ] **Step 3: Train ind_7 (scene-level + map) on AD4CHE**

`sbatch --export=ALL,RF_DATASET=ad4che,RF_AD4CHE_ROOT=data_ad4che,RF_SCENE_LEVEL=1 scripts/train_dmog.slurm`
Expected: decreasing loss; second new checkpoint.

- [ ] **Step 4: CHECKPOINT** — both AD4CHE checkpoints serialized; record their filenames as `AD4CHE_EGO=...`, `AD4CHE_JOINT=...`. (~10 h each; run concurrently.)

---

## Task 7: Recalibrate + conflict labels on AD4CHE

**Files:** none new

- [ ] **Step 1: AD4CHE conflict labels**

`RF_DATASET=ad4che RF_AD4CHE_ROOT=data_ad4che RF_STRIDE=10 RF_OUT=conflict_labels_ad4che.npz python scripts/conflict_labels.py`
Expected: prints conflict rate; sweep `RF_MARGIN`/`RF_VMIN` to land ~8–15% with a clean gap separation (conflict min-gap ≪ safe). Congested highway → expect more rear-end conflicts than InD; report the rate.

- [ ] **Step 2: Recalibrate T_critical on AD4CHE**

`sbatch --export=ALL,RF_DATASET=ad4che,RF_AD4CHE_ROOT=data_ad4che,RF_CKPT_EGO=serialized/<AD4CHE_EGO>,RF_CKPT_JOINT=serialized/<AD4CHE_JOINT> scripts/calibrate_joint_dmog.slurm`
Expected: `risk_calibration.json` (move/rename to `risk_calibration_ad4che.json`) with a finite `T_critical`; `complete:true`.

- [ ] **Step 3: CHECKPOINT** — labels + calibration produced for AD4CHE.

---

## Task 8: Evaluate — prediction quality + conflict detection

**Files:** none new (reuse `wm_fidelity.py`, `eval_conflict.py`)

- [ ] **Step 1: Prediction quality (NLL/ADE)**

`RF_DATASET=ad4che RF_AD4CHE_ROOT=data_ad4che RF_MAX_SCENES=300 python scripts/wm_fidelity.py` (B1 block prints ego marginal NLL + neighbour conditional NLL). Add minADE/minFDE by reusing `evaluate.py` on the AD4CHE loader (single-target, `RF_CKPT_EGO`).
Expected: finite, sharp NLLs (ego marginal clearly < 0); record them.

- [ ] **Step 2: Conflict-detection comparison**

`srun ... RF_DATASET=ad4che RF_AD4CHE_ROOT=data_ad4che RF_CKPT_EGO=serialized/<AD4CHE_EGO> RF_CKPT_JOINT=serialized/<AD4CHE_JOINT> RF_LABEL_FILES=conflict_labels_ad4che.npz RF_GRID=48 RF_STRIDE=10 python scripts/eval_conflict.py`
Expected: `conflict_eval.json` with AUROC/AP/lead for ours/ours-prob/pora/ttc/dsf on AD4CHE. Sanity: TTC AUROC > 0.5; **ours ≥ pora-style** (the same-backbone win should replicate).

- [ ] **Step 3: CHECKPOINT** — AD4CHE prediction + detection numbers collected.

---

## Task 9: Paper — AD4CHE generalization section

**Files:** Modify `WorldModel-Informed-RiskField/main.tex`

- [ ] **Step 1: Add a `\subsection{Cross-Dataset Generalization (AD4CHE)}`** under Results, with: dataset description (congested highway, 11 scenes, per-scene normalization), a prediction-quality mini-table (NLL/minADE/minFDE), and an AD4CHE conflict-detection table paralleling `tab:risk_compare`. Use only measured numbers; `\TODO{}` anything not yet run.
- [ ] **Step 2: Compile** `pdflatex … main.tex` ×2 → exit 0, no undefined refs.
- [ ] **Step 3: CHECKPOINT** — paper builds with the AD4CHE results.

---

## Self-Review notes

- **Spec coverage:** loader (T1–2), per-scene norm + map (T1,3,4), dataset-abstraction (T5), retrain (T6), calibrate+label (T7), eval prediction+detection (T8), paper (T9). Counterfactual/negative-result intentionally out of scope.
- **Key risk gated early:** Task 3 registration render is the make-or-break correctness check before any training.
- **InD regression guard:** Task 5 Step 6 re-runs an InD eval slice after the refactor.
- **Open impl decisions (each has a default):** `sampling_step=3` (Δt 0.10 s) for horizon parity; 2-class head (`bus→1`) to reuse InD's FiLM weights; per-scene box from pooled track extent.
