#!/usr/bin/env python3

from socket import gethostname

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

from retinalrisk.training import setup_training
from retinalrisk.utils.callbacks import WritePredictionsDataFrame


@hydra.main("../../config", config_name="config")
def main(args: DictConfig):
    seed_everything(0)

    print(OmegaConf.to_yaml(args))

    host = gethostname()
    cluster = "charite-hpc" if host.startswith("s-sc") else "eils-hpc"
    output_root = f"{args.setup.root[cluster]}/{args.setup.output_path}"

    # setup training
    datamodule, module, tags = setup_training(args)

    # wandb
    api = wandb.Api()
    runs = api.runs(args.setup.entity + "/" + args.setup.project)

    run = None
    for r in runs:
        if r.id == str(args.setup.restore_id):
            run = r

    assert run is not None, f'Run {args.setup.restore_id} not found.'

    if 'best_checkpoint' in run.config.keys():
        args.model.restore_from_ckpt = run.config.best_checkpoint
    else:
        args.model.restore_from_ckpt = '/sc-projects/sc-proj-ukb-cvd/results/models/retina/2af9tvdp/checkpoints/epoch=40-step=27962.ckpt'

    cb = WritePredictionsDataFrame()
    cb.manual(args, datamodule, module)

    #TODO: log to wandb

    print('DONE')


if __name__ == '__main__':
    main()


