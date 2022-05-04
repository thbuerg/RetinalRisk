import copy
import datetime
import gc
import os
import glob
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import scipy
import torch
from pytorch_lightning import LightningDataModule
from scipy.sparse import coo_matrix
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader

from retinalrisk.data.data import WandBBaselineData
from retinalrisk.data.datasets import RetinalFundusDataset

# TODO:
'''
[x] remove the GNN/embedding parts
[ ] make sure to retain functions such as eid lists etc
[ ] define interface for img batches
[ ] read targets as files??
[ ] assess how to best do record encodings // how much of that we really need since its only phecode outputs.
'''
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

        img_file_extension: Optional[str] = '.png',
        img_visit: Optional[int] = 0,

        use_top_n_phecodes: int = 200,
        covariates: List[str] = [],
        filter_phecode_categories: Iterable[str] = ("Cong", "Dev", "Neonate"),
        filter_phecode_strings: Iterable[str] = ("history of",),
        filter_input_origins: Iterable[str] = [], #todo: need this??
        **kwargs,
    ):
        super().__init__()
        self.img_root = img_root
        self.img_file_extension = img_file_extension
        self.img_visit = img_visit

        self.partition = partition
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task = task
        self.task_kwargs = task_kwargs
        self.label_definition = label_definition
        self.use_top_n_phecodes = use_top_n_phecodes
        self.covariate_cols = covariates

        self.data = wandb_data

        # todo: do we need those?
        # self.le_eids, self.cat_type_eid = self.data.record_encodings["eid"]
        # self.le_concepts, self.cat_type_concepts = self.data.record_encodings["concept_id"]
        # self.le_concepts_cols = self.le_concepts.classes_

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    @property
    def num_classes(self) -> int:
        return self.labels[0].shape[1]

    # TODO: a) adapt to receive input from baseline file directly
    # TODO: b) need this?
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
        self.labels = ['_'.join(c.split('_')[:-1])
                       for c in self.data.outcomes.columns if 'event' in c]

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
            idxs = np.where(np.in1d(self.eids[split].astype('int32'),
                                    self.data.outcomes.reset_index().eid.unique(), assume_unique=True))[0]
            idxs = np.where(np.in1d(self.eids[split].astype('int32')[idxs],
                                    img_eid_map.eid.unique(), assume_unique = True))[0]
            split_eids = self.eids[split][idxs]
            subsplit_img_eid_map = img_eid_map.query('eid in @split_eids')
            self.img_map_by_split[split] = subsplit_img_eid_map

        self.prepare_labels()

        self.setup_covariates_preprocessor()

        print("Generating train dataset...")
        self.train_dataset = self.get_retina_dataset(set="train")

        print("Generating valid dataset...")
        self.valid_dataset = self.get_retina_dataset(set="valid")

    def get_records_data(self, t0, split):
        """
        Generate a records dataset with t0
        :param t0:  ['recruitment', 'random_age', 'random_censoring']
        :return:
        """
        covariates_df = self.data.covariates
        covariates = self.covariate_preprocessor.transform(covariates_df[self.covariate_cols])

        eid_df = self.data.eid_df.copy()
        eid_df.sort_values("eid", ascending=True, inplace=True)
        eid_df["t0_date"] = eid_df["recruitment_date"]
        censorings = ((eid_df.exit_date - eid_df.t0_date) / np.timedelta64(1, "Y")).to_frame(
            name="cens_time"
        )

        # TODO: get records from wandb baseline file
        records_events = None
        records_times = None

        return covariates, records_events, records_times, censorings

    def get_retina_dataset(self, set="train"):
        # todo: get this information from baseline file
        exclusions = self.data.outcomes[[c for c in self.data.outcomes.columns if 'prev' in c]]

        covariates_df = self.data.covariates
        covariates = self.covariate_preprocessor.transform(covariates_df[self.covariate_cols])

        labels_events = self.data.outcomes[[c for c in self.data.outcomes.columns if 'event' in c]]
        labels_times = self.data.outcomes[[c for c in self.data.outcomes.columns if 'time' in c]]
        labels_times = labels_times.where(labels_events.values != 0, 0)
        censorings = self.data.outcomes[[c for c in self.data.outcomes.columns if 'time' in c]].max(axis=1).to_frame(
            name='cens_time')

        dataset = RetinalFundusDataset(
            img_map=self.img_map_by_split[set],
            exclusions=exclusions,
            labels_events=labels_events,
            labels_times=labels_times,
            covariates=covariates,
            censorings=censorings,
            eids=self.eids[set],
        )

        return dataset

    # trainer_flag: reload_dataloaders_every_epoch=True
    def train_dataloader(self, shuffle=True, drop_last=True):
        if self.t0_mode != "recruitment":
            self.train_dataset = None
            gc.collect()
            self.train_dataset = self.get_retina_dataset(set="train")
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            pin_memory=False,
            batch_size=self.batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            collate_fn=self.graph_collator,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            num_workers=self.num_workers,
            pin_memory=False,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            collate_fn=self.graph_collator,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            print("Generating test dataset...")
            self.test_dataset = self.get_retina_dataset(set="test")
        return DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            pin_memory=False,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            collate_fn=self.graph_collator,
            persistent_workers=self.num_workers > 0,
        )
