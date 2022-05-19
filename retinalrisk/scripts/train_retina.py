#!/usr/bin/env python3

from pathlib import Path

from socket import gethostname

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from retinalrisk.training import setup_training
from retinalrisk.utils.callbacks import (
    WriteFeatureAttributions,
    WritePredictionsDataFrame,
    EncoderFreezeUnfreeze
)

from retinalrisk.utils.helpers import extract_metadata

@hydra.main("../../config", config_name="config")
def main(args: DictConfig):
    seed_everything(0)

    print(OmegaConf.to_yaml(args))

    host = gethostname()
    cluster = "charite-hpc" if host.startswith("s-sc") else "eils-hpc"
    output_root = f"{args.setup.root[cluster]}/{args.setup.output_path}"

    datamodule, model, tags = setup_training(args)

    wandb_logger = WandbLogger(
        entity=args.setup.entity,
        project=args.setup.project,
        group=args.setup.group,
        name=args.setup.name,
        tags=tags,
        config=args,
    )
    wandb_logger.watch(model, log_graph=True)

    callbacks = [
        ModelCheckpoint(mode="min", monitor="valid/loss", save_top_k=1, save_last=True),
        EarlyStopping(
            monitor="valid/loss",
            min_delta=0.00000001,
            patience=args.training.patience,
            verbose=False,
            mode="min",
        ),
    ]

    if args.training.warmup_period > 0:
        callbacks.append(EncoderFreezeUnfreeze(args.training.warmup_period))

    if args.training.write_predictions:
        callbacks.append(WritePredictionsDataFrame())

    if args.training.write_embeddings:
        callbacks.append(WriteRecordNodeEmbeddingsDataFrame())

    if args.training.write_attributions:
        callbacks.append(
            WriteFeatureAttributions(
                batch_size=args.datamodule.batch_size,
                baseline_mode=args.training.attribution_baseline_mode,
            )
        )

    trainer = Trainer(
        default_root_dir=output_root,
        logger=wandb_logger,
        callbacks=callbacks,
        **args.trainer,
    )

    trainer.fit(model, datamodule=datamodule)

    if hasattr(trainer, "checkpoint_callback"):
        OmegaConf.save(
            config=args, f=f"{Path(trainer.checkpoint_callback.dirpath).parent}/config.yaml"
        )
        wandb_logger.experiment.config[
            "best_checkpoint"
        ] = trainer.checkpoint_callback.best_model_path
        wandb_logger.experiment.config["best_score"] = trainer.checkpoint_callback.best_model_score


if __name__ == "__main__":
    main()
