import os
import sys
import shutil
import math
from pathlib import Path
from datetime import datetime

import ray
import pytorch_lightning as pl
from ray import train, tune, air
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

import argparse

# ---------- PROJECT IMPORTS ----------
MRLFADS_DIR = "/project/bdoiron/juliet/MR-LFADS"

if MRLFADS_DIR not in sys.path:
    sys.path.insert(0, MRLFADS_DIR)

import config.paths as path
from mrlfads.run import run


# ---------- USER SETTINGS ----------
SCRIPT_PATH = Path(__file__).resolve()
CUR_DIR = SCRIPT_PATH.parent
CONFIG_PATH = CUR_DIR / "main.yaml"

parser = argparse.ArgumentParser()
parser.add_argument(
    "area",
    choices=["MOp", "RSP", "SSp-tr", "VISp"],
    help="Area to run the hyperparameter search for",
)
args = parser.parse_args()

AREAS = [args.area]
# AREAS = ["MOp", "RSP", "SSp-tr", "VISp"]
KL_U_VALUES = [0.01, 0.1, 1.0, 10.0]
L2_VALUES = [1e3, 1e4, 1e5] # recurrent weight regularization
DROP_VALUES = [0.1, 0.3, 0.5] # drop out

HYPERPARAM_SPACE = {
    "area": tune.grid_search(AREAS),
    "kl_u": tune.grid_search(KL_U_VALUES),
    "l2_scale": tune.grid_search(L2_VALUES),
    "dropout_rate": tune.grid_search(DROP_VALUES),
    "seed": 0,
}

ALLOCATED_CPUS = int(os.environ["SLURM_CPUS_PER_TASK"])

CPUS_PER_TRIAL = 2
RESOURCES_PER_TRIAL = {"cpu": CPUS_PER_TRIAL, "gpu": 0.2}

MAX_CONCURRENT_TRIALS = min(ALLOCATED_CPUS // CPUS_PER_TRIAL,5)

if MAX_CONCURRENT_TRIALS < 1:
    raise ValueError("Request at least 2 CPUs for this configuration.")

# -----------------------------------


def trainable(trial_config, config_path):
    # Ensure project imports are available inside the Ray worker.
    if MRLFADS_DIR not in sys.path:
        sys.path.insert(0, MRLFADS_DIR)

    from mrlfads.run import run

    trial_dir = Path(train.get_context().get_trial_dir()).resolve()
    os.chdir(trial_dir)

    area = trial_config["area"]
    config_file = Path(config_path).resolve()

    # 1. Compose the original multi-area configuration.
    with initialize_config_dir(
        version_base="1.1",
        config_dir=str(config_file.parent),
    ):
        cfg = compose(config_name=config_file.stem)

    # 2. Convert this trial to a single-area model.
    # Update settings before resolving ${...} references.
    with open_dict(cfg):
        cfg.seed = int(trial_config["seed"])
        cfg.datamodule.area_names = [area]
        cfg.model.num_other_areas = 0
        cfg.model.detach_hn = True

        cfg.model.kl_co_scale = float(trial_config["kl_u"])
        cfg.model.l2_scale = float(trial_config["l2_scale"])
        cfg.model.dropout_rate = float(trial_config["dropout_rate"])
        cfg.model.kl_ic_scale = float(trial_config["kl_u"])

    # Resolve all cross-area references before removing any areas.
    resolved = OmegaConf.to_container(cfg, resolve=True)

    # Explicitly replace the dictionaries with single-area dictionaries.
    resolved["model"]["areas_params"] = {area: resolved["model"]["areas_params"][area]}
    resolved["model"]["hn_indices"] = {area: resolved["model"]["hn_indices"][area]}

    cfg = OmegaConf.create(resolved)

    area_cfg = cfg.model.areas_params[area]
    for key in ("ci_enc_dim", "con_dim", "co_dim"):
        if area_cfg[key] <= 0:
            raise ValueError(
                f"{area}: {key} must be positive to tune inferred-input KL"
            )

    # 3. Resolve interpolations and save a standalone trial config.
    # The composed config contains the selected settings from all
    # config groups, so run() no longer needs their original YAMLs.
    trial_config_path = trial_dir / "trial_config.yaml"
    OmegaConf.save(config=cfg, f=str(trial_config_path), resolve=True)

    # Verify the actual file that run() will load.
    saved = OmegaConf.load(str(trial_config_path))
    assert list(saved.model.areas_params.keys()) == [area]
    assert list(saved.datamodule.area_names) == [area]
    assert list(saved.model.hn_indices.keys()) == [area]

    print(
        f"Trial area={area}; "
        f"model areas={list(saved.model.areas_params.keys())}; "
        f"data areas={list(saved.datamodule.area_names)}; "
        f"config={trial_config_path}",
        flush=True,
    )

    # 4. Train 
    # model, datamodule, _ = run(config_path=str(trial_config_path), train=True,)
    
    import torch._inductor.config as inductor_config
    
    # Work around the failing matrix-padding optimization.
    inductor_config.shape_padding = False
    
    try:
        model, datamodule, _ = run(config_path=str(trial_config_path), train=True,)
    except Exception:
        import traceback

        error_text = traceback.format_exc()
        error_path = trial_dir / "original_training_error.txt"
        error_path.write_text(error_text)

        print(
            f"\nOriginal training failure:\n{error_text}"
            f"\nSaved to: {error_path}",
            file=sys.stderr,
            flush=True,
        )

        # Pass Ray a simple exception rather than the compiler exception.
        raise RuntimeError(f"Training failed; original traceback:\n{error_text}") from None

    # Check that this area actually has inferred inputs enabled.
    area_cfg = cfg.model.areas_params[area]
    for key in ("ci_enc_dim", "con_dim", "co_dim"):
        if area_cfg[key] <= 0:
            raise ValueError(
                f"{area}: {key} must be positive to tune inferred-input KL"
            )

    # 5. Evaluate the final trained weights on validation trials.
    evaluator = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    validation_results = evaluator.validate(model=model, datamodule=datamodule, verbose=False,)

    metric_name = f"valid/{area}/hn"
    metrics = validation_results[0]

    if metric_name not in metrics:
        raise RuntimeError(
            f"Missing metric {metric_name}. "
            f"Available metrics: {list(metrics)}"
        )

    holdout_loss = float(metrics[metric_name])

    train.report({"holdout_loss": holdout_loss})


if __name__ == "__main__":
    project_name = CUR_DIR.name
    run_name = (
        f"{project_name}_sr_{args.area}_"
        f"{datetime.now():%y%m%d%H%M%S}_"
        f"{os.environ.get('SLURM_JOB_ID', 'local')}"
    )

    run_dir = Path(path.resultpath).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    shutil.copyfile(SCRIPT_PATH, run_dir / SCRIPT_PATH.name)

    run_config = air.RunConfig(storage_path=str(run_dir.parent), name=run_dir.name)
    
    ray.init(
        address="local",
        num_cpus=int(os.environ["SLURM_CPUS_PER_TASK"]),
        num_gpus=1,
        include_dashboard=False,
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(trainable,config_path=str(CONFIG_PATH)),
            resources=RESOURCES_PER_TRIAL),
        param_space=HYPERPARAM_SPACE,
        tune_config=tune.TuneConfig(
            metric="holdout_loss",
            mode="min",
            max_concurrent_trials=MAX_CONCURRENT_TRIALS,
        ),
        run_config=run_config,
    )

    try:
        results = tuner.fit()
        best_by_area = {}

        for area in AREAS:
            candidates = [result for result in results if result.error is None and result.config["area"] == area and "holdout_loss" in result.metrics]

            if not candidates:
                print(f"{area}: no successful trials")
                continue
            
            best = min(candidates, key=lambda result: result.metrics["holdout_loss"])

            best_by_area[area] = {
                "kl_co_scale": float(best.config["kl_u"]),
                "l2_scale": float(best.config["l2_scale"]),
                "dropout_rate": float(best.config["dropout_rate"]),
                "kl_ic_scale": float(best.config["kl_u"]),
                "holdout_loss": float(best.metrics["holdout_loss"]),
                "trial_dir": str(best.path),
            }

            print(
                f"{area}: "
                f"KL(u)={best.config['kl_u']:g}, "
                f"holdout loss={best.metrics['holdout_loss']:.6f}\n"
                f"  Trial directory: {best.path}"
            )

        OmegaConf.save(
            config=OmegaConf.create(best_by_area),
            f=str(run_dir / "best_kl_by_area.yaml"),
        )

    finally:
        ray.shutdown()