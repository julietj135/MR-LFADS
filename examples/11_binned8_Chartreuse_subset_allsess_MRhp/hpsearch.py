"""Joint MR-LFADS search: independent area KL(m) scales, shared KL(m) schedule."""

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

AREAS = ["MOp", "RSP", "SSp-tr", "VISp"]
KL_COM_SCALES = [0.001, 0.01]
KL_STARTS = [100, 150]
KL_INC = [100, 200]

HYPERPARAM_SPACE = {
    "kl_start": tune.grid_search(KL_STARTS),
    "kl_inc": tune.grid_search(KL_INC),
    **{f"kl_com_{area}": tune.grid_search(KL_COM_SCALES) for area in AREAS},
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

    config_file = Path(config_path).resolve()

    # 1. Compose the original multi-area configuration.
    with initialize_config_dir(
        version_base="1.1",
        config_dir=str(config_file.parent),
    ):
        cfg = compose(config_name=config_file.stem)

    # 2. Configure the joint four-area model.
    with open_dict(cfg):
        cfg.seed = int(trial_config["seed"])
        cfg.datamodule.area_names = list(AREAS)
        cfg.model.num_other_areas = len(AREAS) - 1
        cfg.model.detach_hn = True
        
        cfg.model.kl_start_epoch_com = int(trial_config["kl_start"])
        cfg.model.kl_increase_epoch_com = int(trial_config["kl_inc"])
        
    # Hold out the first 10% of neurons in each session.
    heldout = {}
    for area in AREAS:
        area_cfg = cfg.model.areas_params[area]
        area_cfg.kl_com_scale = float(trial_config[f"kl_com_{area}"])

        neuron_counts = area_cfg.num_neurons
        heldout[area] = []
        for session_idx in range(len(neuron_counts)):
            n_neurons = int(neuron_counts[f"n{session_idx}"])
            n_holdout = max(1, int(0.10 * n_neurons))
            heldout[area].append(list(range(n_holdout)))

    resolved = OmegaConf.to_container(cfg, resolve=True)
    resolved["model"]["areas_params"] = {area: resolved["model"]["areas_params"][area] for area in AREAS}
    resolved["model"]["hn_indices"] = heldout

    cfg = OmegaConf.create(resolved)

    # 3. Resolve interpolations and save a standalone trial config.
    # The composed config contains the selected settings from all
    # config groups, so run() no longer needs their original YAMLs.
    trial_config_path = trial_dir / "trial_config.yaml"
    OmegaConf.save(config=cfg, f=str(trial_config_path), resolve=True)

    # Verify the actual file that run() will load.
    saved = OmegaConf.load(str(trial_config_path))
    assert list(saved.model.areas_params.keys()) == AREAS
    assert list(saved.datamodule.area_names) == AREAS
    assert list(saved.model.hn_indices.keys()) == AREAS
    assert saved.model.num_other_areas == 3

    print(
        f"Joint trial areas={AREAS}; "
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

    # 5. Evaluate the final trained weights on validation trials.
    evaluator = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    validation_results = evaluator.validate(model=model, datamodule=datamodule, verbose=False,)

    metrics = validation_results[0]
    report = {}

    for area in AREAS:
        metric_name = f"valid/{area}/hn"

        if metric_name not in metrics:
            raise RuntimeError(
                f"Missing {metric_name}. Available metrics: {list(metrics)}"
            )

        value = float(metrics[metric_name])
        if not math.isfinite(value):
            raise RuntimeError(f"Non-finite holdout loss for {area}: {value}")

        report[f"holdout_loss_{area}"] = value

    report["holdout_loss"] = sum(report.values()) / len(AREAS)
    train.report(report)    


if __name__ == "__main__":
    project_name = CUR_DIR.name
    run_name = (
        f"{project_name}_mr_klcom_"
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

        candidates = [
            result for result in results
            if result.error is None
            and "holdout_loss" in result.metrics
            and math.isfinite(float(result.metrics["holdout_loss"]))
        ]

        if not candidates: raise RuntimeError("No successful joint trials. Check the trial error files.")
        best = min(candidates, key=lambda result: result.metrics["holdout_loss"])

        summary = {
            "kl_start_epoch_com": int(best.config["kl_start"]),
            "kl_increase_epoch_com": int(best.config["kl_inc"]),
            "kl_com_scale_by_area": {
                area: float(best.config[f"kl_com_{area}"])
                for area in AREAS
            },
            "holdout_loss": float(best.metrics["holdout_loss"]),
            "holdout_loss_by_area": {
                area: float(best.metrics[f"holdout_loss_{area}"])
                for area in AREAS
            },
            "trial_dir": str(best.path),
        }

        summary_cfg = OmegaConf.create(summary)
        OmegaConf.save(
            config=summary_cfg,
            f=str(run_dir / "best_joint_communication.yaml"),
        )
        print(OmegaConf.to_yaml(summary_cfg), flush=True)

    finally:
        ray.shutdown()