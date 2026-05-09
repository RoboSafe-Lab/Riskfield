import argparse
import os
from typing import Optional

import numpy as np
import torch

from datasets.InD import InD
import risk_ranking_experiment as rre


def makedir(directory: str) -> None:
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def _count_env_cars_from_input(batch: dict, *, ego_index: int = 0) -> int:
    """Count non-empty environment cars in the observed input (history)."""
    inp = batch["input"]  # (B,N,H,2)
    if torch.is_tensor(inp):
        inp0 = inp[0].detach().cpu().numpy()
    else:
        inp0 = np.asarray(inp)[0]

    N = inp0.shape[0]
    cnt = 0
    for i in range(N):
        if i == ego_index:
            continue
        car = inp0[i]
        if np.all(car == 0) or np.isnan(car).all():
            continue
        cnt += 1
    return int(cnt)


def _resolve_model_path(path: str) -> str:
    if os.path.exists(path):
        return path

    candidates: list[str] = []
    base = os.path.basename(path)
    candidates.append(os.path.join("serialized", base))
    if not base.lower().endswith(".pt"):
        candidates.append(os.path.join("serialized", base + ".pt"))
        candidates.append(base + ".pt")
    candidates.append(os.path.join("serialized", path))

    for c in candidates:
        if os.path.exists(c):
            return c

    cand_msg = "\n".join([f"  - {c}" for c in candidates])
    raise FileNotFoundError(
        f"Model file not found: '{path}'. Tried:\n{cand_msg}\n"
        "Tip: pass '--model serialized/multi_trajflow_ind_2.pt'"
    )


def _summarize_array(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    xf = x[finite]
    out = {
        "count": int(x.size),
        "finite": int(xf.size),
        "nan": int(x.size - xf.size),
    }
    if xf.size == 0:
        out.update(
            {
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "p10": float("nan"),
                "p25": float("nan"),
                "median": float("nan"),
                "p75": float("nan"),
                "p90": float("nan"),
                "max": float("nan"),
            }
        )
        return out

    out.update(
        {
            "mean": float(np.mean(xf)),
            "std": float(np.std(xf)),
            "min": float(np.min(xf)),
            "p10": float(np.quantile(xf, 0.10)),
            "p25": float(np.quantile(xf, 0.25)),
            "median": float(np.quantile(xf, 0.50)),
            "p75": float(np.quantile(xf, 0.75)),
            "p90": float(np.quantile(xf, 0.90)),
            "max": float(np.max(xf)),
        }
    )
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Batch risk ranking test: randomly sample n eligible segments (>=min env cars) "
            "and compute per-t Spearman correlation between model risk field and objective risk field."
        )
    )
    p.add_argument("--n", type=int, required=True, help="Number of segments to sample")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    p.add_argument("--min-env", type=int, default=2, help="Minimum number of environment cars (excluding ego)")
    p.add_argument("--max-eligible", type=int, default=None, help="Optionally cap eligible pool size for speed")

    p.add_argument("--steps", type=int, default=64, help="Grid resolution per axis (use smaller for speed)")
    p.add_argument(
        "--radius",
        type=int,
        default=0,
        help="Optional px smoothing radius in grid cells (0 = no smoothing). Does not affect cache contents.",
    )
    p.add_argument("--site", type=str, default="08", help="InD site id, e.g. 08")
    p.add_argument("--model", type=str, required=True, help="Path to model .pt (state_dict)")
    p.add_argument("--device", type=str, default=None, help="cpu | cuda | auto")
    p.add_argument("--cost-step", type=int, default=2, help="dt = 0.04 * cost_step")

    p.add_argument("--sigma", type=float, default=1.5, help="Gaussian sigma (meters) for objective density")
    p.add_argument(
        "--objective",
        type=str,
        default="gaussian_relv",
        choices=["gaussian_relv", "gaussian_dist", "inv_dist", "min_dist", "ttc"],
        help="Objective risk definition",
    )
    p.add_argument("--eps", type=float, default=1e-6, help="Numerical epsilon")
    p.add_argument("--ttc-max", type=float, default=5.0, help="Max TTC seconds for objective=ttc")
    p.add_argument("--ttc-tau", type=float, default=2.0, help="TTC decay time constant (s) for objective=ttc")

    p.add_argument("--cache-dir", type=str, default=None, help="Cache dir for px tensors")
    p.add_argument("--out", type=str, default="videos/risk_ranking_batch", help="Output directory")
    p.add_argument("--save-per-seg", action="store_true", help="Save per-segment correlations (.npz)")
    p.add_argument("--dry-run", action="store_true", help="Only compute eligible indices and exit")
    return p


def main() -> None:
    from model.RiskFlow import RiskFlow
    from riskflow_config import preset, seed_everything

    args = build_arg_parser().parse_args()

    cfg = preset("vis_field")
    seed_everything(cfg["seed"])

    if args.device is None or args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    ind = InD(
        root="data",
        max_samples=cfg["maximum_samples"],
        train_ratio=cfg["train_ratio"],
        train_batch_size=cfg["train_batch_size"],
        test_batch_size=cfg["test_batch_size"],
        missing_rate=cfg["masked_data_ratio"],
        max_num_cars=cfg["max_num_cars"],
        max_empty_frames=cfg["max_empty_frames"],
        seq_len=cfg["seq_len"],
        moving_window=cfg["seq_len"] * 2,
        sampling_step=cfg["sampling_step"],
        should_shuffle=False,
        include_future=True,
    )

    site_id = str(args.site)
    obs = getattr(ind, f"observation_site_{site_id}", None)
    if obs is None:
        obs = ind._get_observation_site([site_id])
    observation_site = obs

    model_path = _resolve_model_path(args.model)

    traj_flow = RiskFlow(
        seq_len=cfg["seq_len"],
        input_dim=cfg["input_dim"],
        feature_dim=cfg["feature_dim"],
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        max_num_cars=cfg["max_num_cars"],
        num_classes=cfg["num_classes"],
        gru_layers=cfg["gru_layers"],
        num_heads=cfg["num_heads"],
        dropout=cfg["dropout"],
        norm_rotation=cfg["norm_rotate"],
        flow_layers=cfg["flow_layers"],
        flow_hidden_dim=cfg["flow_hidden_dim"],
        coupling_layers=cfg["coupling_layers"],
        use_cnf=cfg["use_cnf"],
        use_cgmm=cfg["use_cgmm"],
        gmm_modes=cfg["gmm_modes"],
    ).to(device)

    traj_flow.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    traj_flow.eval()

    out_dir = args.out
    makedir(out_dir)

    # Materialize once
    test_data = list(iter(observation_site.test_loader))

    eligible = []
    for idx, batch in enumerate(test_data):
        try:
            env_cnt = _count_env_cars_from_input(batch, ego_index=0)
        except Exception:
            continue
        if env_cnt >= int(args.min_env):
            eligible.append(idx)
        if args.max_eligible is not None and len(eligible) >= int(args.max_eligible):
            break

    eligible = np.array(eligible, dtype=np.int64)
    rng = np.random.default_rng(int(args.seed))

    if eligible.size == 0:
        raise RuntimeError("No eligible segments found (min-env too strict?)")

    if args.dry_run:
        print(f"eligible={eligible.size} / total={len(test_data)}")
        return

    n = int(args.n)
    if n > eligible.size:
        raise ValueError(f"Requested n={n} but only eligible={eligible.size}. Reduce n or min-env.")

    chosen = rng.choice(eligible, size=n, replace=False)

    spearman_means = np.full((n,), np.nan, dtype=np.float64)
    spearman_per_t_list: list[np.ndarray] = []
    seg_ids: list[int] = []

    for k, seg in enumerate(chosen.tolist()):
        batch = test_data[int(seg)]
        try:
            res = rre.run_experiment_for_batch(
                observation_site=observation_site,
                model=traj_flow,
                batch=batch,
                segment_index=int(seg),
                steps=int(args.steps),
                device=device,
                cost_step=int(args.cost_step),
                sigma=float(args.sigma),
                objective=str(args.objective),
                eps=float(args.eps),
                ttc_max_s=float(args.ttc_max),
                ttc_tau_s=float(args.ttc_tau),
                cache_dir=args.cache_dir,
                ego_index=0,
                radius=int(args.radius),
            )
        except Exception as e:
            print(f"segment={seg} failed: {type(e).__name__}: {e}")
            continue

        seg_ids.append(int(seg))
        spearman_means[len(seg_ids) - 1] = float(res["spearman_mean"])
        spearman_per_t_list.append(np.asarray(res["spearman_per_t"], dtype=np.float64))

        if args.save_per_seg:
            np.savez(
                os.path.join(out_dir, f"seg{int(seg)}_spearman.npz"),
                segment=int(seg),
                spearman_per_t=np.asarray(res["spearman_per_t"], dtype=np.float64),
                spearman_mean=float(res["spearman_mean"]),
                steps=int(args.steps),
                radius=int(args.radius),
                sigma=float(args.sigma),
                cost_step=int(args.cost_step),
                objective=str(args.objective),
                eps=float(args.eps),
                ttc_max_s=float(args.ttc_max),
                ttc_tau_s=float(args.ttc_tau),
            )

        print(f"[{k+1}/{n}] segment={seg} spearman_mean={float(res['spearman_mean']):.4f}")

    if len(seg_ids) == 0:
        raise RuntimeError("All sampled segments failed; try lowering min-env or checking dataset/model.")

    # Trim arrays to actual successes
    seg_ids_arr = np.array(seg_ids, dtype=np.int64)
    means_arr = spearman_means[: len(seg_ids)].copy()

    # Align per-t matrix by padding to max T
    T_max = int(max(x.shape[0] for x in spearman_per_t_list))
    per_t_mat = np.full((len(seg_ids), T_max), np.nan, dtype=np.float64)
    for i, v in enumerate(spearman_per_t_list):
        per_t_mat[i, : v.shape[0]] = v

    per_t_mean = np.nanmean(per_t_mat, axis=0)
    per_t_std = np.nanstd(per_t_mat, axis=0)

    stats_mean = _summarize_array(means_arr)

    # Additional useful stats
    frac_pos = float(np.nanmean(means_arr > 0.0))
    frac_02 = float(np.nanmean(means_arr > 0.2))
    frac_05 = float(np.nanmean(means_arr > 0.5))

    np.savez(
        os.path.join(out_dir, "summary.npz"),
        chosen_segments=seg_ids_arr,
        spearman_mean=means_arr,
        spearman_per_t=per_t_mat,
        spearman_per_t_mean=per_t_mean,
        spearman_per_t_std=per_t_std,
        eligible_count=int(eligible.size),
        total_count=int(len(test_data)),
        requested_n=int(args.n),
        sampled_n=int(n),
        succeeded_n=int(len(seg_ids)),
        min_env=int(args.min_env),
        steps=int(args.steps),
        radius=int(args.radius),
        sigma=float(args.sigma),
        cost_step=int(args.cost_step),
        objective=str(args.objective),
        eps=float(args.eps),
        ttc_max_s=float(args.ttc_max),
        ttc_tau_s=float(args.ttc_tau),
        frac_pos=frac_pos,
        frac_gt_02=frac_02,
        frac_gt_05=frac_05,
        stats_mean=stats_mean,
    )

    # Also emit a tiny CSV for quick scan
    csv_path = os.path.join(out_dir, "segments.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("segment,spearman_mean\n")
        for s, m in zip(seg_ids_arr.tolist(), means_arr.tolist()):
            f.write(f"{s},{m}\n")

    print("\nSummary:")
    print(f"eligible={eligible.size} total={len(test_data)} requested_n={n} succeeded={len(seg_ids)}")
    print(
        f"spearman_mean: mean={stats_mean['mean']:.4f} std={stats_mean['std']:.4f} "
        f"median={stats_mean['median']:.4f} p25={stats_mean['p25']:.4f} p75={stats_mean['p75']:.4f}"
    )
    print(f"fraction mean>0: {frac_pos:.3f}  >0.2: {frac_02:.3f}  >0.5: {frac_05:.3f}")
    print(f"wrote: {os.path.join(out_dir, 'summary.npz')}")


if __name__ == "__main__":
    main()
