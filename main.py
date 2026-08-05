import os
import time
import wandb
import torch
from datasets.InD import InD
from model.RiskFlow import RiskFlow
from train import train
from evaluate import evaluate
from riskflow_config import set_wandb_defaults, seed_everything

should_train = True
should_serialize = True
should_evaluate = True


last_model_name = "riskflow_ind_0.pt"  # Change this to the desired model name

with wandb.init(group="AFT") as run:
    # Centralized defaults (one place to edit): riskflow_config.py
    set_wandb_defaults(run)
    seed_everything(run.config.seed)
    verbose = bool(run.config.verbose)

    # Env overrides (for ablations / scripted runs without editing config):
    #   RF_USE_WORLD_MODEL=0/1   RF_SITE_SCOPE=all|08|loc1..loc4
    _uwm = bool(getattr(run.config, "use_world_model", False))
    if "RF_USE_WORLD_MODEL" in os.environ:
        _uwm = os.environ["RF_USE_WORLD_MODEL"].strip().lower() in ("1", "true", "yes")
    _scope = os.environ.get("RF_SITE_SCOPE", getattr(run.config, "site_scope", "08"))
    # RF_SCENE_LEVEL=0 forces the single-target (per-agent marginal) path used
    # by the risk field; RF_SCENE_LEVEL=1 the autoregressive joint. Defaults to
    # the config value when unset.
    _scene = bool(getattr(run.config, "scene_level", False))
    if "RF_SCENE_LEVEL" in os.environ:
        _scene = os.environ["RF_SCENE_LEVEL"].strip().lower() in ("1", "true", "yes")
    _map_local = bool(getattr(run.config, "map_local", False))
    if "RF_MAP_LOCAL" in os.environ:
        _map_local = os.environ["RF_MAP_LOCAL"].strip().lower() in ("1", "true", "yes")
    _epochs = int(os.environ.get("RF_EPOCHS", getattr(run.config, "training_epochs", 50)))
    print(f"[run config] use_world_model={_uwm} site_scope={_scope} scene_level={_scene}")
    wandb.log({"cfg/use_world_model": int(_uwm), "cfg/site_scope": str(_scope),
               "cfg/scene_level": int(_scene)})

    from datasets.registry import get_dataset
    _reg = get_dataset(os.environ.get("RF_DATASET", getattr(run.config, "dataset", "ind")))
    print(f"[run config] dataset={_reg['name']}")
    ind = _reg["LoaderClass"](
        root=_reg["root"],
        max_samples=run.config.maximum_samples,
        train_ratio=run.config.train_ratio,
        train_batch_size=run.config.train_batch_size,
        test_batch_size=run.config.test_batch_size,
        missing_rate=run.config.masked_data_ratio,
        max_num_cars=run.config.max_num_cars,
        max_empty_frames=run.config.max_empty_frames,
        seq_len=run.config.seq_len,
        moving_window=run.config.seq_len * 2,
        sampling_step=run.config.sampling_step,
        should_shuffle=run.config.should_shuffle,
        include_future=run.config.include_future,
    )
    observation_site = ind.observation_site_by_scope(_scope)

    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    traj_flow = RiskFlow(
        seq_len=run.config.seq_len,
        input_dim=run.config.input_dim,
        feature_dim=run.config.feature_dim,
        embedding_dim=run.config.embedding_dim,
        hidden_dim=run.config.hidden_dim,
        max_num_cars=run.config.max_num_cars,
        num_classes=run.config.num_classes,
        gru_layers=run.config.gru_layers,
        num_heads=run.config.num_heads,
        dropout=run.config.dropout,
        norm_rotation=run.config.norm_rotate,
        flow_layers=run.config.flow_layers,
        flow_hidden_dim=run.config.flow_hidden_dim,
        coupling_layers=run.config.coupling_layers,
        use_cnf=run.config.use_cnf,
        use_cgmm=run.config.use_cgmm,
        gmm_modes=run.config.gmm_modes,
        use_world_model=_uwm,
        wm_state_dim=getattr(run.config, "wm_state_dim", 256),
        action_dim=getattr(run.config, "action_dim", 2),
        scene_level=_scene,
        agent_ordering=getattr(run.config, "agent_ordering", "nearest_ego"),
        use_map=getattr(run.config, "use_map", False),
        map_size=getattr(run.config, "map_size", 64),
        map_data_dir=_reg["map_data_dir"],
        map_dataset=_reg["map_dataset"],
        map_local=_map_local,
        map_crop_m=getattr(run.config, "map_crop_m", 40.0),
        map_raster_res=getattr(run.config, "map_raster_res", 192),
    ).to(device)

    num_parameters = sum(p.numel() for p in traj_flow.parameters() if p.requires_grad)
    if verbose:
        print(f"parameters: {num_parameters}")
    wandb.log({"parameters": num_parameters})

    total_loss = []
    if should_train:
        train_start_time = time.time()
        total_loss = train(
            observation_site=observation_site,
            model=traj_flow,
            epochs=_epochs,
            lr=run.config.lr,
            weight_decay=run.config.weight_decay,
            gamma=run.config.gamma,
            verbose=verbose,
            device=device,
        )
        train_end_time = time.time()
        train_runtime = train_end_time - train_start_time
        if verbose:
            print(train_runtime)
        wandb.log({"train runtime": train_runtime})

        for loss in total_loss:
            wandb.log({"loss": loss})
        minimum_loss = min(total_loss) if len(total_loss) > 0 else 0
        wandb.log({"minimum loss": minimum_loss})

    traj_flow.eval()
    batch = next(iter(observation_site.test_loader))
    inputs, features, types, targets = (
        batch["input"],
        batch["feature"],
        batch["type"],
        batch["target"],
    )
    inputs = inputs.to(device)
    features = features.to(device)
    types = types.to(device)
    inference_start_time = time.time()
    traj_flow.sample(inputs, features, targets.shape[1], types, 100)
    inference_end_time = time.time()
    inference_runtime = inference_end_time - inference_start_time
    if verbose:
        print(inference_runtime)
    wandb.log({"inference runtime": inference_runtime})

    if should_serialize:
        serialize_dir = "serialized"
        if should_train:
            os.makedirs(serialize_dir, exist_ok=True)
            num = 0
            # If the model name exists, add a suffix number
            while os.path.exists(
                os.path.join(serialize_dir, f"riskflow_ind_{num}.pt")
            ):
                num += 1
            model_name = f"riskflow_ind_{num}.pt"
            _mc = "_mc" if _map_local else ""    # map-crop variant -> distinct name, keep baseline
            if _reg["name"] == "ad4che":
                # deterministic distinct names -> no race between concurrent runs
                model_name = f"riskflow_ad4che_{'joint' if _scene else 'ego'}{_mc}.pt"
            elif _reg["name"] == "round":
                model_name = f"riskflow_round_{'joint' if _scene else 'ego'}{_mc}.pt"
            elif _reg["name"] == "ind" and _map_local:
                # InD convention: ego=ind_8, joint=ind_7 (matches RF_CKPT defaults)
                model_name = f"riskflow_ind_{'7' if _scene else '8'}_mc.pt"
            torch.save(traj_flow.state_dict(), os.path.join(serialize_dir, model_name))
        else:
            model_name = last_model_name
            traj_flow.load_state_dict(
                torch.load(os.path.join(serialize_dir, model_name))
            )
        print("model loaded")

    if should_evaluate:
        rmse, crps, min_ade, min_fde, nll = evaluate(
            observation_site=observation_site,
            model=traj_flow,
            num_samples=run.config.evaluation_samples,
            device=device,
        )

        if verbose:
            print(f"rmse: {rmse}")
            print(f"crps: {crps}")
            print(f"min ade: {min_ade}")
            print(f"min fde: {min_fde}")
            print(f"nll: {nll}")
        wandb.log(
            {
                "rmse": rmse,
                "crps": crps,
                "min ade": min_ade,
                "min fde": min_fde,
                "nll": nll,
            }
        )
