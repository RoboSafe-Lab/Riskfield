"""Centralized default configuration for TrajFlow scripts.

Goal: keep dataset/model/training hyperparameters in ONE place so that
`main.py`, `train.py`, visualization scripts, and script-like tests share the
same defaults.

Notes
- This module intentionally uses plain dicts (WandB-friendly) and small helper
  functions. Scripts may still override a small number of run-mode flags.
- Keep key names stable: they are referenced by W&B sweeps (multi_sweep.yaml)
  and by scripts via `run.config.<key>`.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping


@dataclass(frozen=True)
class TrajFlowDefaults:
    # Repro
    seed: int = 42
    # seed: int = 43

    # Dataset
    sampling_step: int = 2
    maximum_samples: int = 10000
    # maximum_samples: int = 20000
    train_ratio: float = 0.75
    train_batch_size: int = 64
    test_batch_size: int = 1
    masked_data_ratio: float = 0.25
    seq_len: int = 50
    max_num_cars: int = 8
    max_empty_frames: int = 0
    should_shuffle: bool = True
    include_future: bool = True

    # Input/feature
    input_dim: int = 2
    feature_dim: int = 5

    # Model
    num_classes: int = 2
    embedding_dim: int = 256
    hidden_dim: int = 512
    gru_layers: int = 3
    num_heads: int = 16
    dropout: float = 0.1
    norm_rotate: bool = False

    # Flow
    use_cnf: bool = False
    # use_cgmm: bool = True
    use_cgmm: bool = False
    gmm_modes: int = 3
    flow_layers: int = 3
    flow_hidden_dim: int = 512
    coupling_layers: int = 10

    # Train/Eval
    training_epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.0
    gamma: float = 0.999
    verbose: bool = False
    evaluation_samples: int = 100

    # Loss shaping (optional; keep defaults speed-neutral)
    # If > 0, apply exponential decay weights w_t = exp(-loss_time_decay * t) to per-frame log-likelihood.
    loss_time_decay: float = 0.05

    # If > 0, add centroid smoothness regularization computed from model samples.
    # This adds an extra reverse-flow pass per batch with effective batch size B*centroid_samples.
    centroid_smooth_lambda: float = 0.2
    centroid_samples: int = 3
    # Sharpness for softmax weighting across samples to approximate a "mode-like" centroid.
    centroid_alpha: float = 10.0


def default_dict() -> dict[str, Any]:
    """Return the canonical defaults as a plain dict (safe to mutate externally)."""

    return dict(asdict(TrajFlowDefaults()))


WANDB_DEFAULTS: dict[str, Any] = default_dict()


def merged(base: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    out = dict(base)
    if overrides:
        out.update(dict(overrides))
    return out


# Script presets live here so you only edit *one* file when defaults change.
# The base is always WANDB_DEFAULTS; overrides cover known intentional differences.
PRESET_OVERRIDES: dict[str, dict[str, Any]] = {
    # main training/eval script
    "main": {},

    # train.py standalone historically used missing_rate=0 (no masking)
    "train": {
        "masked_data_ratio": 0.0,
        "test_batch_size": 1,
    },

    # visualize_px.py historically used a CNF model and slightly different class/empty-frame params
    "vis_px": {
        "use_cnf": True,
        "use_cgmm": False,
        "num_classes": 3,
        "max_empty_frames": 25,
        "flow_hidden_dim": 128,
        "should_shuffle": False,
    },

    # visualize_field.py uses deterministic ordering for index-based selection
    "vis_field": {
        "should_shuffle": False,
    },

    # script-like tests typically want deterministic ordering
    "test": {
        "should_shuffle": False,
    },
}


def preset(name: str) -> dict[str, Any]:
    """Return defaults merged with a named preset override."""

    overrides = PRESET_OVERRIDES.get(name)
    if overrides is None:
        raise KeyError(f"Unknown preset: {name}. Available: {sorted(PRESET_OVERRIDES)}")
    return merged(WANDB_DEFAULTS, overrides)


def set_wandb_defaults(run: Any, overrides: Mapping[str, Any] | None = None) -> None:
    """Apply shared defaults to a wandb run's config.

    This keeps `run.config.<key>` access working across scripts.
    """

    run.config.setdefaults(WANDB_DEFAULTS)

    if overrides:
        # Prefer WandB's update API if present.
        update = getattr(run.config, "update", None)
        if callable(update):
            try:
                update(dict(overrides), allow_val_change=True)
                return
            except TypeError:
                # Older wandb versions may not accept allow_val_change.
                update(dict(overrides))
                return

        # Fallback: set attributes (best-effort).
        for k, v in overrides.items():
            try:
                run.config[k] = v
            except Exception:
                setattr(run.config, k, v)


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and torch (best-effort)."""

    import random

    random.seed(int(seed))

    try:
        import numpy as np

        np.random.seed(int(seed))
    except ModuleNotFoundError:
        pass

    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except ModuleNotFoundError:
        pass
