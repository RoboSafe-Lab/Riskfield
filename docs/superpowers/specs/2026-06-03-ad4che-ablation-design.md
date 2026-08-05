# AD4CHE Cross-Dataset Ablation Study — Design

**Date:** 2026-06-03
**Status:** approved design, pending implementation plan

## Context & Goal

The paper's risk field (two world-model-conditioned flows: `ind_8` ego marginal +
`ind_7` scene-level joint, both map-conditioned) is validated only on InD (four
intersections). To support a generalization claim, we retrain and evaluate the
method on a **second, structurally different dataset — AD4CHE** (congested Chinese
highway, drone-captured). Scope locked with the user:

- **AD4CHE only** (the other component ablations — map on/off, oriented box,
  prob-vs-energy — are dropped: the first two are obviously-required and already
  justified; prob-vs-energy is already in `tab:risk_compare` as `ours` vs
  `ours-prob`).
- **Train + evaluate on AD4CHE** (not zero-shot transfer — the map encoder is
  per-location and would be OOD across datasets).
- **Evaluate:** (1) prediction quality (held-out NLL + minADE/minFDE), and
  (2) the conflict-detection comparison (ours / ours-prob / PORA-style / TTC / DSF,
  AUROC/AP/lead-time). No counterfactual J(A) or negative-result replication.

The InD-trained checkpoints + InD artifacts are already backed up to
`backups/ind_2026-06-03/` on the cluster (md5-verified).

## AD4CHE data format (confirmed)

Per recording `datasets/AD4CHE/<scene 14..24>/<DJI_xxxx>/`:
- `NN_tracks_pixel.csv`: `frame,id,x,y,width,height,xVelocity,yVelocity,
  xAcceleration,yAcceleration,...,angle,orientation,...,x_pixel,y_pixel`.
  **`x,y` are already in metres; `xVelocity,yVelocity` in m/s; `orientation` in
  radians.** `width,height` = per-vehicle length,width (m).
- `NN_tracksMeta.csv`: `id,width,height,initialFrame,finalFrame,numFrames,class,
  drivingDirection,...` — `class` in {car, truck, bus}.
- `NN_recordingMeta.csv`: `id,frameRate(=30),locationId(=-1),...,scale` where
  `scale` is the string `"1 pixel = 0.0375 m"`. `locationId` is always -1 in the
  CSV, but the **directory scene number (14..24) is the physical location**:
  recordings within a scene share one registered coordinate frame (verified:
  scene 14's three recordings all span x≈[-71,70], y≈[-17,11]) and the per-scene
  map `maps/NN.jpg`.
- Maps: **per-scene** `maps/14.jpg … 24.jpg` (one canonical road per scene, the
  InD-orthophoto analogue). Per-recording `NN_highway.png`/`NN_lanePicture.png`
  also exist but the per-scene map is canonical.

This is parallel to InD but **simpler** (positions already metric). trajdata's
`dataset_specific/ad4che/ad4che_dataset.py` is the column-semantics reference; we
do **not** use trajdata's `AgentBatch` (a native loader is far less friction).

## Component 1 — `datasets/AD4CHE.py` (native loader)

Mirror `datasets/InD.py`'s public surface so the rest of the pipeline is unchanged:
produce batches with the exact keys `input (B,N,T,2)`, `feature (B,N,T,F)`,
`type (B,N)`, `future (B,N,K,2)`, `target (B,K,2)`, `carMask`, `locationId`,
plus an `observation_site_by_scope("all")` with `.train_loader/.test_loader` and
`denormalize_loc`.

Key mappings / decisions:
- **Positions/velocities:** use `x,y,xVelocity,yVelocity,xAcceleration,
  yAcceleration` directly (metric). Heading feature = `orientation` (rad) →
  degrees → `/360` to match InD's `feature_boundaries` channel-0 convention.
  Reuse `feature_boundaries`, widening velocity/accel ranges for highway speeds.
- **Normalization:** **per scene (14..24 = locations, like InD's 4 intersections).**
  All recordings of a scene share one spatial box (computed from the scene's
  pooled track extent, or the map extent) mapping positions to `[0,1]`;
  `locationId = scene number`. Expose `AD4CHE_SPATIAL_BOUNDARIES` +
  `boundaries_for_location(loc)`. Boxes are anisotropic (~140 m × 28 m highways);
  the metric `scale=(x_range,y_range)` carries the anisotropy as in InD.
- **Vehicle dimensions:** use per-vehicle `width,height` from tracksMeta where
  available; fall back to a class table `{car,truck,bus}`. (The risk field's
  `VEH_LW` becomes dataset-aware — see Component 2.)
- **Classes:** car/truck/bus → `num_classes=3` (or map bus→truck for 2 to reuse
  InD's FiLM head; decide at impl based on count). Filter to vehicles only.
- **Map:** resample the per-scene `maps/NN.jpg` into the scene's normalized box,
  same machinery as `MapEncoder.build_location_maps` but with AD4CHE
  scenes/boundaries (Component 2). One map per scene, shared by its recordings.
- **Windowing:** `T=50` history, `K=50` future, `sampling_step=2` (Δt≈0.067 s at
  30 fps; or downsample to match InD's 0.08 s — decide so horizons are comparable),
  `max_num_cars=8`. Reuse InD's sliding-window + carMask logic.

## Component 2 — Make the pipeline dataset-aware (the real work)

Several modules currently hardcode InD. Parameterize them so InD↔AD4CHE is a
config switch (`dataset: "ind" | "ad4che"`):

- `model/MapEncoder.py::build_location_maps` — currently takes InD
  `LOCATION_RECORDINGS` + `boundaries`. Pass AD4CHE's recordings/boundaries +
  highway-map filename pattern. (Already parameterized by args; the caller in
  `RiskFlow.__init__` hardcodes the InD imports — make those dataset-conditional.)
- `model/RiskFlow.py::__init__` — the `use_map` branch imports
  `InD.LOCATION_RECORDINGS` and `LOCATION_SPATIAL_BOUNDARIES`. Add a
  `map_dataset` param to select InD vs AD4CHE sources.
- `scripts/joint_field.py` — `VEH_LW` (class→dims) becomes dataset-aware (AD4CHE
  has buses + per-vehicle dims). `A_SCALE` unchanged.
- `scripts/conflict_labels.py`, `scripts/eval_conflict.py`,
  `scripts/calibrate_joint.py`, `scripts/wm_fidelity.py` — replace the InD
  `boundaries_for_location` import and the `InD(...)` construction with a small
  `get_dataset(name)` factory returning the right loader + boundaries fn. Drive
  via an env/config `RF_DATASET`.

Prefer a thin `datasets/registry.py` (`get_dataset(name)` →
`(LoaderClass, boundaries_for_location, feature_boundaries, veh_dims)`) over
editing every script ad hoc.

## Component 3 — Training

- Add AD4CHE to `riskflow_config.py` / `main.py` via `RF_DATASET=ad4che`.
- Retrain **`ind_8`** (`scene_level=False, use_map=True`) and **`ind_7`**
  (`scene_level=True, use_map=True`) on AD4CHE → new checkpoints (auto-incremented
  `ind_9`, `ind_10`, or explicit `riskflow_ad4che_{8,7}.pt`). ~10 h each on the
  cluster. InD checkpoints already backed up.

## Component 4 — Calibration, labels, evaluation

- Recalibrate `T_critical` on AD4CHE (`calibrate_joint.py` with `RF_DATASET=ad4che`).
- Generate AD4CHE conflict labels (`conflict_labels.py`, same OBB near-miss +
  motion gate; re-check the base rate, sweep margin/v_min).
- **Prediction quality:** held-out neighbour conditional NLL (`ind_7`) + ego
  marginal NLL (`ind_8`) + minADE/minFDE (reuse `wm_fidelity.py` B1 + `evaluate.py`).
- **Conflict-detection comparison:** run `eval_conflict.py` on AD4CHE (ours /
  ours-prob / PORA-style / TTC / DSF; AUROC/AP/lead-time across the label sweep).
- Paper: a new ablation/generalization subsection + an AD4CHE results table
  paralleling `tab:metrics` + `tab:risk_compare`.

## Verification

1. **Loader smoke:** load one AD4CHE recording, build a batch, assert shapes/keys
   match InD; render a scene's GT boxes on the highway map → **registration
   correct** (boxes sit on the road). Catches coord/scale/heading bugs early.
2. **Map smoke:** `build_location_maps` for AD4CHE returns nonzero rasters for all
   recordings.
3. **Training:** epoch NLL decreasing then plateau (as InD did).
4. **Eval sanity:** AD4CHE NLL finite/sharp; conflict base-rate plausible (few–15%);
   TTC baseline AUROC > 0.5; ours ≥ PORA-style (same-backbone) on AD4CHE.

## Risks

- **Coordinate/heading conventions** (radian orientation, driving-direction sign,
  per-recording frames) — the loader smoke render is the guard.
- **Frame rate (30 fps vs InD 25/sampling) / Δt** — pick sampling so the K-step
  horizon is comparable across datasets; document the chosen Δt.
- **Map style mismatch** (highway lane-painting vs InD orthophoto) — the MapEncoder
  is retrained on AD4CHE so this is fine; just confirm nonzero, road-aligned rasters.
- **Class head** (3 classes vs InD 2) — may need `num_classes=3`; small FiLM change.
- **Dataset-abstraction churn** touching shared scripts — guard InD behavior with a
  quick InD re-run of one metric after refactor to ensure no regression.

## Out of scope

Zero-shot transfer; counterfactual J(A) on AD4CHE; negative-result replication;
the dropped component ablations (map, box). The component value `prob vs energy`
is already reported (InD `tab:risk_compare`).
