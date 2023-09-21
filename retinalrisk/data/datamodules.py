import os
import glob
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from pytorch_lightning import LightningDataModule
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader

from retinalrisk.data.data import WandBBaselineData
from retinalrisk.data.collate import ImgCollator
from retinalrisk.data.sample import RepeatedSampler
from retinalrisk.data.datasets import RetinalFundusDataset


class RetinaDataModule(LightningDataModule):
    def __init__(
        self,
        wandb_data: WandBBaselineData,
        img_root: str,
        partition: int,
        batch_size: int,
        num_workers: int,
        task: str, # TODO: need this??
        task_kwargs: dict, # TODO: need this??

        label_definition: dict,

        augmentation: Optional[dict] = {},
        img_size_to_gpu: Optional[int] = 299,
        img_crop_ratio: Optional[float] = 0.7,
        img_file_extension: Optional[str] = '.png',
        img_visit: Optional[int] = 0,
        img_n_testtime_views: Optional[int] = 1,

        covariates: List[str] = [],
        **kwargs,
    ):
        super().__init__()
        self.img_root = img_root
        self.img_file_extension = img_file_extension
        self.img_visit = img_visit
        self.img_size_to_gpu = img_size_to_gpu
        self.img_crop_ratio = img_crop_ratio
        self.img_n_testtime_views = img_n_testtime_views

        self.collator = None
        self.augmentation_pipeline = None

        self.prepare_augmentation_pipeline(augmentation)

        self.partition = partition
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task = task
        self.task_kwargs = task_kwargs
        self.label_definition = label_definition
        self.covariate_cols = covariates

        self.data = wandb_data

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    @property
    def num_classes(self) -> int:
        return self.labels[0].shape[1]

    def setup_covariates_preprocessor(self):
        covariates_df = self.data.covariates[self.covariate_cols]

        numeric_features = [
            f
            for f in covariates_df.columns
            if pd.api.types.is_numeric_dtype(covariates_df[f].dtype)
        ]
        categorical_features = [
            f
            for f in covariates_df.columns
            if pd.api.types.is_categorical_dtype(covariates_df[f].dtype)
        ]

        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        self.covariate_preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        self.covariate_preprocessor.fit(covariates_df.loc[self.eids["train"]])

    def prepare_labels(self):
        # This is a minimal and naive version of prepare_labels that
        # assumes all cols provided in the outcome file should be used.

        self.labels = []
        if self.label_definition is None:
            self.labels += ['_'.join(c.split('_')[:-1])
                           for c in self.data.outcomes.columns if 'event' in c]
        else:
            if len(self.label_definition["custom"]) > 0:
                if isinstance(self.label_definition["custom"], list):
                    self.labels += self.label_definition["custom"]
                elif isinstance(self.label_definition["custom"], str):
                    custom_label_df = pd.read_csv(self.label_definition["custom"])
                    self.labels += custom_label_df.endpoint.apply(
                        lambda s: s.replace(".", "-").replace("/", "-")
                    ).values.tolist()
                else:
                    assert False

        print('Labels are...')
        print(self.labels)

        phecode_lookup = self.data.phecode_definitions.assign(
            phecode=self.data.phecode_definitions.phecode.apply(lambda s: s.replace(".", "-").replace("/", "-"))
        ).set_index('phecode')

        def map_label(l):
            if "phecode" in l:
                definition = phecode_lookup.loc[l.split("phecode_")[1]]
                return f"{l} - {definition.phecode_string}"
            return l

        fix_str = lambda s: s.replace(".", "-").replace("/", "-")

        self.label_mapping = {fix_str(l): fix_str(map_label(l)) for l in self.labels}

    def prepare_data(self):
        if self.train_dataset is not None:
            print('Data already prepared.')
            return
        
        self.eids = self.data.eid_dict[self.partition]

        # now get eid->img_path map
        if self.img_visit is not None:
            files = [fp for fp in sorted(
                glob.glob(os.path.join(self.img_root,
                                       f'*{self.img_file_extension}' if self.img_file_extension is not None else '*')))
                     if f'_{self.img_visit}_' in fp]
        else:
            files = [fp for fp in sorted(
                glob.glob(os.path.join(self.img_root,
                                       f'*{self.img_file_extension}' if self.img_file_extension is not None else '*')))]
        eids = np.asarray([os.path.basename(fp).split('_')[0] for fp in files]).astype(int)
        img_eid_map = pd.DataFrame(np.stack([eids, np.asarray(files)], axis=-1),
                               columns=['eid', 'file_path'])
        img_eid_map = img_eid_map.astype({'eid': 'int32'})

        self.img_map_by_split = dict()
        for split in ("train", "valid", "test"):
            # select eids by outcomes and split:
            outcomes_eids = set(self.data.outcomes.reset_index().eid.values)
            img_eids = set(img_eid_map.eid.values)
            split_eids = list(set(self.eids[split]) & outcomes_eids & img_eids)
            self.eids[split] = split_eids
            subsplit_img_eid_map = img_eid_map.query('eid in @split_eids')
            self.img_map_by_split[split] = subsplit_img_eid_map

        self.prepare_labels()

        self.setup_covariates_preprocessor()

        print("Generating train dataset...")
        self.train_dataset = self.get_retina_dataset(set="train")

        print("Generating valid dataset...")
        self.valid_dataset = self.get_retina_dataset(set="valid")

        self.collator = ImgCollator()

    def prepare_augmentation_pipeline(self, augmentation_pipeline):
        """
        Prepare the augmentations
        :return:
        """
        # instantiate the list of augmentations
        self.augmentation_pipeline = dict()

        if augmentation_pipeline:
            for split, pipeline in augmentation_pipeline.items():
                self.augmentation_pipeline[split] = []
                if len(pipeline) > 0: # empty ListConfig would cause error on .items() call
                    for aug_str, kwargs in pipeline.items():
                        self.augmentation_pipeline[split].append(transforms.__dict__[aug_str](**kwargs))

    def get_retina_dataset(self, set="train"):
        exclusions = self.data.outcomes[[c for c in self.data.outcomes.columns if c.replace('_prev', '') in self.labels]].loc[self.eids[set]]

        covariates = self.data.covariates.loc[self.eids[set]]
        print(covariates.head())
        fts = self.covariate_preprocessor.transform(covariates[self.covariate_cols])
        covariates = pd.DataFrame(fts,
                                  columns=[f'ft_{i}' for i in range(fts.shape[1])],
                                  index = covariates.index)

        labels_events = self.data.outcomes[[c for c in self.data.outcomes.columns if 'event' in c]].loc[self.eids[set]]
        labels_times = self.data.outcomes[[c for c in self.data.outcomes.columns if 'time' in c]].loc[self.eids[set]]

        # select correct labels:
        labels_events = labels_events[[f"{c}_event" for c in self.labels]]
        labels_times = labels_times[[f"{c}_time" for c in self.labels]]

        labels_times = labels_times.where(labels_events.values != 0, 0)

        censorings = self.data.outcomes[[c for c in self.data.outcomes.columns if 'time' in c]].max(axis=1).to_frame(
            name='cens_time').loc[self.eids[set]]

        dataset = RetinalFundusDataset(
            img_map=self.img_map_by_split[set],
            augmentations=self.augmentation_pipeline[set],
            exclusions=exclusions,
            labels_events=labels_events,
            labels_times=labels_times,
            covariates=covariates,
            censorings=censorings,
            eids=self.eids[set],
            img_crop_ratio=self.img_crop_ratio[set],
            img_size_to_gpu = self.img_size_to_gpu
        )

        return dataset

    # trainer_flag: reload_dataloaders_every_epoch=True
    def train_dataloader(self, shuffle=True, drop_last=True, testtime=False):
        sampler = None
        batch_size = self.batch_size

        if testtime and self.img_n_testtime_views > 1:
            sampler = RepeatedSampler(n_times=self.img_n_testtime_views,
                                      data_source=self.train_dataset)
            batch_size = int((batch_size // self.img_n_testtime_views) * self.img_n_testtime_views)

        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            pin_memory=False,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            collate_fn=self.collator,
            persistent_workers=self.num_workers > 0,
            sampler=sampler,
        )

    def val_dataloader(self, testtime=False):
        sampler = None
        batch_size = self.batch_size

        if testtime and self.img_n_testtime_views > 1:
            sampler = RepeatedSampler(n_times=self.img_n_testtime_views,
                                      data_source=self.valid_dataset)
            batch_size = int((batch_size // self.img_n_testtime_views) * self.img_n_testtime_views)

        return DataLoader(
            self.valid_dataset,
            num_workers=self.num_workers,
            pin_memory=False,
            batch_size=batch_size,
            drop_last=False,
            shuffle=False,
            collate_fn=self.collator,
            persistent_workers=self.num_workers > 0,
            sampler=sampler
        )

    def test_dataloader(self, testtime=True):
        if self.test_dataset is None:
            print("Generating test dataset...")
        self.test_dataset = self.get_retina_dataset(set="test")

        sampler = None
        batch_size = self.batch_size
        if testtime and self.img_n_testtime_views > 1:
            sampler = RepeatedSampler(n_times=self.img_n_testtime_views,
                                      data_source=self.test_dataset)
            batch_size = int((batch_size // self.img_n_testtime_views) * self.img_n_testtime_views)

        return DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            pin_memory=False,
            batch_size=batch_size,
            drop_last=False,
            shuffle=False,
            collate_fn=self.collator,
            persistent_workers=self.num_workers > 0,
            sampler=sampler
        )
