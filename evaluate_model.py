import os
import warnings

import hydra
import pandas as pd
import torch
import wandb
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from scripts.data_loader import train_val_test_loader
from scripts.exceptions import (
    EvaluateFreshInitializedModelException,
    UnknownModeException,
)
from scripts.model_config import model_selection
from scripts.utilis import model_path


def save_results_to_csv(results: list[dict], filename: str) -> None:
    """Save the evaluation results to a CSV file.

    Parameters
    ----------
    results : list[dict]
        List of dictionaries containing the evaluation results.
    filename : str, optional
        The filename or path where the results should be saved.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    base, ext = os.path.splitext(filename)
    if ext.lower() != ".csv":
        warnings.warn(
            f"Evaluation results file '{filename}',"
            f" extension '{ext}' is not '.csv'. Changing it to '.csv'."
        )
        filename = f"{base}.csv"

    counter = 1
    new_filename = filename
    while os.path.exists(new_filename):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1

    df = pd.DataFrame(results)
    df.to_csv(new_filename, index=False)
    print(f"Evaluation results saved to {new_filename}")


def log_bar_charts_to_wandb(
    results: list[dict],
    mode: str,
    logger: WandbLogger,
) -> None:
    """
    Log bar charts to wandb using wandb.Table and wandb.plot.bar().

    Parameters
    ----------
    results : list[dict]
        List of dictionaries containing the evaluation results.
    mode : str
        The evaluation mode.
    logger : WandbLogger
        Wandb logger instance.
    """
    if not results:
        return

    df = pd.DataFrame(results)
    
    if mode == "all":
        df['config'] = df['posterior'] + '_steps' + df['step'].astype(str)
        for metric in df['metric'].unique():
            metric_df = df[df['metric'] == metric]
            if not metric_df.empty:
                table = wandb.Table(dataframe=metric_df[['config', 'value']])
                logger.experiment.log({
                    f"chart/{metric}": wandb.plot.bar(table, "config", "value", title=f"Evaluation Results for {metric}")
                })
    else:
        key_col = "step" if mode == "steps" else "posterior"
        for metric in df['metric'].unique():
            metric_df = df[df['metric'] == metric]
            if not metric_df.empty:
                table = wandb.Table(dataframe=metric_df[[key_col, 'value']])
                logger.experiment.log({
                    f"chart/{metric}": wandb.plot.bar(table, key_col, "value", title=f"Evaluation by {key_col.capitalize()} for {metric}")
                })


def evaluate_model(cfg, model, trainer: Trainer, test_loader, logger: WandbLogger) -> None:
    """Evaluate the model based on the specified configuration.

    Parameters
    ----------
    cfg : OmegaConf
        Configuration object containing evaluation settings.
    model : pl.LightningModule
        The model to be evaluated.
    trainer : Trainer
        PyTorch Lightning Trainer object.
    test_loader : torch.utils.data.DataLoader
        DataLoader for the test dataset.
    logger: WandbLogger
        The logger for recording results.
    """
    results = []

    if cfg.evaluation.mode in ["steps", "all"]:
        for interference_step in cfg.evaluation.steps:
            model.diffusion.set_timesteps(interference_step)
            trainer.test(model, test_loader, ckpt_path=None)
            metrics = trainer.callback_metrics
            for metric_name, metric_value in metrics.items():
                if "test/" in metric_name:
                    clean_metric_name = metric_name.replace("test/", "")
                    results.append({
                        "step": interference_step,
                        "posterior": cfg.diffusion.posterior_type,
                        "metric": clean_metric_name,
                        "value": metric_value.item(),
                    })

    if cfg.evaluation.mode in ["posterior", "all"]:
        for posterior in cfg.evaluation.posteriors:
            model.diffusion.set_posterior_type(posterior)
            # Use default timesteps when evaluating posteriors
            model.diffusion.set_timesteps(cfg.diffusion.timesteps)
            trainer.test(model, test_loader, ckpt_path=None)
            metrics = trainer.callback_metrics
            for metric_name, metric_value in metrics.items():
                if "test/" in metric_name:
                    clean_metric_name = metric_name.replace("test/", "")
                    results.append({
                        "step": cfg.diffusion.timesteps,
                        "posterior": posterior,
                        "metric": clean_metric_name,
                        "value": metric_value.item(),
                    })
    
    if not results:
        raise ValueError("No results were generated. Check your evaluation configuration.")

    log_bar_charts_to_wandb(results, cfg.evaluation.mode, logger)

    if cfg.evaluation.save_results:
        save_results_to_csv(results, cfg.evaluation.results_file)


@hydra.main(version_base=None, config_path="conf", config_name="config_supresdiffgan_evaluation")
def main(cfg) -> None:
    """Main function to evaluate the model based on the provided configuration."""

    if cfg.model.load_model is None:
        raise EvaluateFreshInitializedModelException("The 'load_model' path in the config cannot be empty for evaluation.")

    final_model_path = model_path(cfg)
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Use the correct logger config path
    logger = WandbLogger(
        project=cfg.wandb_logger.project,
        entity=cfg.wandb_logger.entity,
        name=f"evaluation_{final_model_path.split('/')[-1].split('.')[0]}",
        config=config_dict,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus = torch.cuda.device_count() if device == "cuda" else 0

    model = model_selection(cfg=cfg, device=device)
    _, _, test_loader = train_val_test_loader(cfg=cfg)

    model = model.to(device)

    # Simplify trainer strategy for single/multi GPU
    trainer = Trainer(
        accelerator='gpu' if num_gpus > 0 else 'cpu',
        devices=num_gpus if num_gpus > 0 else 1,
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=False) if num_gpus > 1 else 'auto',
    )
    
    print(f"Starting evaluation for model: {cfg.model.load_model}")
    evaluate_model(cfg, model, trainer, test_loader, logger)
    print("Evaluation finished.")


if __name__ == "__main__":
    main()

