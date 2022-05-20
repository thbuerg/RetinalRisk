import os
import json
from pathlib import Path
import neptune.new as neptune
import wandb
import pathlib

import hydra
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import pretrainedmodels
import kornia as K

from pprint import pprint
from collections import OrderedDict
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, MultiStepLR
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging
from tqdm import tqdm

from riskiano.source.callbacks.general import WriteSupervisedPredictionsDataFrameRetina
from riskiano.source.datamodules.retina import *
from riskiano.source.losses.general import BinaryFocalLossWithLogits, weighted_mse_loss, powered_mae_loss
from riskiano.source.tasks.supervised import *
from riskiano.source.tasks.survival import *
from riskiano.source.transforms.general import *
from riskiano.source.utils.general import set_up_neptune, get_default_callbacks
import torchmetrics

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# globals:
import pandas as pd
pd.options.mode.use_inf_as_na = True
import pytorch_lightning as pl
pl.seed_everything(23)  #the number of doom

from neptune.new.types import File


config_path = "config"

def prepare_wandb_runs(tag='test_experiment'):
    # wandb
    api = wandb.Api()
    entity, project = "ll-cha", "retina"
    runs = api.runs(entity + "/" + project)
    run_list = []
    for run in tqdm(runs):
        if tag in run.tags:
            run_list.append(
                {
                    "id": run.path[-1],
                    "name": run.name,
                    "tags": run.tags,
                    "config": {k: v for k, v in run.config.items() if not k.startswith('_')},
                    "summary": run.summary._json_dict,
                    "path": None if "best_checkpoint" not in run.config.keys() else str(
                        pathlib.Path(run.config["best_checkpoint"]).parent.parent)
                }
            )
    return run_list, runs

def write_and_log(output_path, partition, predictions_df, wandb_id, runs_obj, tag='test_experiment' ):
    # write the prediction files
    os.makedirs(output_path, exist_ok=True)
    feather_filename = str(wandb_id) + "_predictions.feather"
    csv_filename = str(wandb_id) + "_predictions.csv"
    predictions_df.to_feather(os.path.join(output_path, feather_filename))
    #predictions_df.to_csv(os.path.join(output_path, csv_filename))

    # write path to wandb run
    for run in runs_obj:
        if run.id == str(wandb_id):
            run.config.update({
                                'predictions_path': f'/sc-projects/sc-proj-ukb-cvd/results/projects/22_retina_phewas/data/{tag}/predictions/{run.id}_predictions.feather'
                              })
            run.update()


def predict(FLAGS):
    #assert FLAGS.evaluation.output_dir is not None, 'specify output directory for predictions!'
    #assert FLAGS.evaluation.checkpoint_path is not None, 'specify checkpoint to model!'
    #assert FLAGS.evaluation.cv_partition is not None, 'specify cv partition!'
    # output_dir = FLAGS.evaluation.output_dir
    # model_checkpoint = FLAGS.evaluation.checkpoint_path
    # cv_partition = FLAGS.evaluation.cv_partition

    # TODO: instead of this, call prepare_wandb_runs, use run_list and forward wandb_runs object to write_and_log
    runs = [10, 21, 0]
    ids  = ['k13j26o7', '3km6lat3', '3opii8oq']
    partitions = [10, 21, 0]
    ckpts = {
        10: '/sc-projects/sc-proj-ukb-cvd/results/models/retina/k13j26o7/checkpoints/last.ckpt',
        21: '/sc-projects/sc-proj-ukb-cvd/results/models/retina/3km6lat3/checkpoints/last.ckpt',
        0: '/sc-projects/sc-proj-ukb-cvd/results/models/retina/3opii8oq/checkpoints/last.ckpt'
    }

    for idx, run in enumerate(runs):
        print('########## RUN:', run)
        output_dir = f'/home/loockl/retina_phewas/retina_phewas_croprange_TTA/predictions/{partitions[idx]}'
        model_checkpoint = ckpts[run]
        cv_partition = f'/home/loockl/retina_phewas/retina_phewas_croprange_TTA/predictions/{partitions[idx]}'

        # labels and datamodule
        with open(FLAGS.setup.phewas_endpoint_list) as f:
            phewas_labels = [line.rstrip() for line in f]

        events = [f'{label}_event' for label in phewas_labels]
        durations = [f'{label}_event_time' for label in phewas_labels]

        FLAGS.experiment.datamodule_kwargs.cv_partition = partitions[idx]
        print('PARTITION:', FLAGS.experiment.datamodule_kwargs.cv_partition)
        print('NUM TESTTIME VIEWS:', FLAGS.experiment.datamodule_kwargs.num_testtime_views)

        datamodule = RetinaUKBBSurvivalDataModule_CropInRange_withExclusions(**FLAGS.experiment.datamodule_kwargs,
                                                                             event=events, duration=durations)
        datamodule.prepare_data()
        datamodule.setup("fit")

        # module and checkpoint

        valid_transforms = [
            # K.augmentation.CenterCrop((FLAGS.experiment.datamodule_kwargs.image_size, FLAGS.experiment.datamodule_kwargs.image_size), return_transform=False),
            K.augmentation.Normalize(
                mean=torch.Tensor([0.5380, 0.2636, 0.0910]),
                std=torch.Tensor([0.1658, 0.0843, 0.0449])
            )
        ]
        transforms = TransformsFromList(train_transforms_list=valid_transforms,
                                        valid_transforms_list=valid_transforms,
                                        cut_mix=FLAGS.experiment.cut_mix,
                                        mix_up=FLAGS.experiment.mix_up)

        encoder = None
        if FLAGS.experiment.architecture in tv.models.__dict__.keys():
            encoder = tv.models.__dict__[FLAGS.experiment.architecture](pretrained=True)
            encoder.fc = nn.Linear(encoder.fc.weight.shape[1], FLAGS.experiment.latent_dim)
        else:
            encoder = pretrainedmodels.__dict__[FLAGS.experiment.architecture](num_classes=1001,
                                                                               pretrained='imagenet+background')  # imagenet init
            # encoder = pretrainedmodels.__dict__[FLAGS.experiment.architecture](num_classes=1000) # no imagenet init
            encoder.last_linear = nn.Linear(encoder.last_linear.weight.shape[1], FLAGS.experiment.latent_dim)
        assert encoder is not None, f'FLAGS.architecture not found in {tv.models.__dict__} and {pretrainedmodels.__dict__}'

        if FLAGS.experiment.predictor_module is not None:
            LatentModule = eval(FLAGS.experiment.predictor_module)
        else:
            LatentModule = None

        if LatentModule is not None:
            latent_module = LatentModule(**FLAGS.experiment.predictor_module_kwargs,
                                         )
        else:
            latent_module = nn.Identity()

        ft_extractor = nn.Sequential(
            encoder,
            nn.SiLU(),
            latent_module
        )

        cause_specific = nn.Identity()

        mts = MultiTaskSurvivalTraining(feature_extractor=ft_extractor,
                                        latent_mlp=cause_specific,
                                        transforms=transforms,
                                        task_names=phewas_labels,
                                        **FLAGS.experiment.task_kwargs)

        # try:
        # model = mts.load_from_checkpoint(model_ckpt2)
        # except RuntimeError:
        cp = torch.load(model_checkpoint, map_location=torch.device('cuda'))
        mts.load_state_dict(cp['state_dict'])
        module = mts
        module.eval()
        module.to(torch.device('cuda'))

        # collect predictions!
        #time_max = 26  # effective real time max is time_max-2 -> 25 years
        #times = [e for e in range(1, time_max, 1)]
        times = [10]

        predictions = {}
        for ds_idx, (ds, ds_name) in enumerate([     (datamodule.train_ds, 'train'),
                                                     (datamodule.valid_ds, 'valid'),
                                                     (datamodule.test_ds["left"], 'test_left'),
                                                     (datamodule.test_ds["right"], 'test_right'),
                                               ]):
            print('current ds:', ds_name)
            predictions[ds_name] = module.predict_dataset(ds, times)
            predictions[ds_name]['eid'] = ds.datasets[0].eid_map.index.values # crash: ValueError: Length of values (6087) does not match length of index (5607)
            predictions[ds_name]["split"] = ds_name

        predictions_df = pd.concat([*predictions.values()]).reset_index(drop=True)
        predictions_df["partition"] = datamodule.cv_partition
        predictions_df["module"] = type(module).__name__
        try:
            predictions_df["net"] = type(module.net).__name__
        except AttributeError:
            pass
        predictions_df["datamodule"] = type(datamodule).__name__
        predictions_df["event_names"] = str(datamodule.event)
        #predictions_df["feature_names"] = str(datamodule.features)

        write_and_log(output_path=output_dir,
                      partition=cv_partition,
                      predictions_df=predictions_df,
                      wandb_id=ids[idx]
                      )

@hydra.main(config_path, config_name="retina_prediction_template")
def main(FLAGS: DictConfig):
    OmegaConf.set_struct(FLAGS, False)
    print(OmegaConf.to_yaml(FLAGS))
    FLAGS.setup.config_path = config_path

    return predict(FLAGS)

if __name__ == '__main__':
    main()