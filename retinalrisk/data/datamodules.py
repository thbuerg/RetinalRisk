import copy
import datetime
import gc
from typing import Iterable, List

import numpy as np
import pandas as pd
import scipy
import torch
import torch_geometric.transforms as T
from pytorch_lightning import LightningDataModule
from scipy.sparse import coo_matrix
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader

from retinalrisk.data.collate import GraphCollator
from retinalrisk.data.data import WandBData
from retinalrisk.data.datasets import RetinalFundusDataset
from retinalrisk.data.sampling import SubgraphNeighborSampler

class EHRGraphDataModule(LightningDataModule):
    def __init__(
            self,
            wandb_data: WandBGraphData,
            t0_mode: str,
            partition: int,
            batch_size: int,
            num_workers: int,
            task: str,
            task_kwargs: dict,
            label_definition: dict,
            graph_sampler: object = SubgraphNeighborSampler,
            graph_sampler_kwargs: dict = dict(
                num_neighbors=[2] * 9,
            ),
            use_top_n_phecodes: int = 200,
            covariates: List[str] = [],
            heterogeneous_graph: bool = True,
            sparse_tensor_graph: bool = True,
            min_edge_type_fraction: float = 0.01,
            filter_phecode_categories: Iterable[str] = ("Cong", "Dev", "Neonate"),
            filter_phecode_strings: Iterable[str] = ("history of",),
            filter_input_origins: Iterable[str] = [],
            buffer_years: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        self.partition = partition
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.t0_mode = t0_mode
        self.task = task
        self.task_kwargs = task_kwargs
        self.label_definition = label_definition
        self.graph_sampler = graph_sampler
        self.graph_sampler_kwargs = graph_sampler_kwargs
        self.graph_collator = None
        self.use_top_n_phecodes = use_top_n_phecodes
        self.covariate_cols = covariates
        self.heterogeneous_graph = heterogeneous_graph
        self.sparse_tensor_graph = sparse_tensor_graph
        self.min_edge_type_fraction = min_edge_type_fraction
        self.filter_phecode_categories = filter_phecode_categories
        self.filter_phecode_strings = filter_phecode_strings
        self.filter_input_origins = filter_input_origins
        self.buffer_period = datetime.timedelta(days=365 * buffer_years)

        self.data = wandb_data
        self.graph = wandb_data.graph
        self.edge_types = wandb_data.edge_types

        self.le_eids, self.cat_type_eid = self.data.record_encodings["eid"]
        self.le_concepts, self.cat_type_concepts = self.data.record_encodings["concept_id"]
        self.le_concepts_cols = self.le_concepts.classes_

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    @property
    def num_features(self) -> int:
        return self.graph.x_dict["0"].shape[1]

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

    def preprocess_graph(self):
        num_features = self.graph.num_features

        if self.min_edge_type_fraction < 1:
            edge_codes, num_counts = np.unique(self.graph.edge_code, return_counts=True)
            fraction = num_counts / len(self.graph.edge_code)
            valid_edge_codes = edge_codes[fraction >= self.min_edge_type_fraction]
            valid_edges = np.isin(self.graph.edge_code, valid_edge_codes)
            self.graph.edge_index = self.graph.edge_index[:, valid_edges]
            self.graph.edge_code = self.graph.edge_code[valid_edges]
            self.graph.edge_weight = self.graph.edge_weight[valid_edges]
            self.edge_types = self.edge_types[valid_edge_codes]

            print(f"Using edge types: {self.edge_types}")

        self.graph_node_ids = copy.deepcopy(self.graph.node_ids)

        if self.heterogeneous_graph:
            self.graph = self.graph.to_heterogeneous(
                torch.zeros(len(self.graph.x), dtype=int),
                self.graph.edge_code,
            )

        if self.sparse_tensor_graph:
            # transform = T.Compose([T.AddSelfLoops(), T.ToSparseTensor()])
            transform = T.Compose([T.ToSparseTensor()])

            self.graph = transform(self.graph)

        self.graph.num_features = num_features

    def prepare_labels(self):
        # prepare labels
        assert len(self.label_definition) > 0
        self.labels = []

        if self.label_definition["all_cause_death"] == True:
            self.labels += ["OMOP_4306655"]  # concept_id for "all_cause_death"
        if self.label_definition["phecodes"] == True:
            concept_idxer = self.data.concept_df.concept_id.apply(
                lambda cid: cid.startswith("phecode")
            )
            # get phecodes and corresponding concept_id_idxs
            phecode_concepts = self.data.concept_df[concept_idxer].copy().reset_index()
            phecode_concepts["phecode"] = phecode_concepts.concept_id.apply(
                lambda s: s.split("_")[1]
            )
            # merge with phecode_definitions to get phecode_categories
            phecode_concepts = phecode_concepts.merge(
                self.data.phecode_definitions, left_on="phecode", right_on="phecode"
            )
            phecode_concepts = phecode_concepts[
                ~phecode_concepts.phecode_category.isin(self.filter_phecode_categories)
            ]
            phecode_concepts = phecode_concepts[
                ~phecode_concepts.phecode_string.apply(
                    lambda s: any(
                        [
                            c.lower().strip() in s.lower().strip()
                            for c in self.filter_phecode_strings
                        ]
                    )
                )
            ]

            # filtered list of concept_id_idxs
            phecode_concept_id_idxes = set(phecode_concepts.concept_id_idx.values)

            top_n_phecodes_idxes = (
                self.data.record_df[
                    self.data.record_df["concept_id_idx"].isin(phecode_concept_id_idxes)
                ]["concept_id_idx"]
                    .value_counts()[: self.use_top_n_phecodes]
                    .index.values.tolist()
            )
            top_n_phecodes = self.data.concept_df.loc[
                top_n_phecodes_idxes
            ].concept_id.values.tolist()
            self.labels += top_n_phecodes
        if len(self.label_definition["custom"]) > 0:
            self.labels += self.label_definition["custom"]

        phecode_lookup = self.data.phecode_definitions.set_index("phecode")

        def map_label(l):
            if "phecode" in l:
                definition = phecode_lookup.loc[l.split("phecode_")[1]]
                return f"{l} - {definition.phecode_string}"
            return l

        fix_str = lambda s: s.replace(".", "-").replace("/", "-")

        self.label_mapping = {fix_str(l): fix_str(map_label(l)) for l in self.labels}

    def prepare_data(self):
        self.eids = self.data.eid_dict[self.partition]

        self.record_df_by_split = dict()
        for split in ("train", "valid", "test"):
            split_eid_idxs = self.data.eid_df[self.data.eid_df.eid.isin(self.eids[split])].index
            subsplit_df = self.data.record_df[self.data.record_df.eid_idx.isin(split_eid_idxs)]
            self.record_df_by_split[split] = subsplit_df

        self.prepare_labels()

        # make sure records are sorted by date s.t. we can easily drop recurrent events later on
        assert self.data.record_df.date[~self.data.record_df.date.isna()].is_monotonic

        # build index table for nodes with records:
        node_ids = list(self.graph.node_ids)
        self.node_ids_set = set(node_ids)
        self.record_cols = [c for c in self.le_concepts_cols if c in self.node_ids_set]
        node_ids_lookup = dict(list((c, i) for i, c in enumerate(node_ids)))
        self.record_node_indices = np.array([node_ids_lookup[c] for c in self.record_cols])

        labels = [l for l in self.labels if l in self.le_concepts_cols]
        assert set(labels) == set(self.labels)
        self.label_node_indices = np.array([node_ids_lookup[c] for c in labels])

        self.setup_covariates_preprocessor()

        # TODO: this is called again in preprocess_graph, potential bug if using graph subsampling
        self.graph_node_ids = copy.deepcopy(self.graph.node_ids)

        # get list of concept_id_idxs either not in filter_input_origins or from phecode vocabulary
        # i.e. only filter out input records, not endpoints
        valid_input_concept_idxs = (
            self.data.concept_df[
                self.data.concept_df.origin.apply(
                    lambda s: all([origin not in s for origin in self.filter_input_origins])
                )
            ]
                .query('vocabulary != "phecode"')
                .index
        )
        valid_concept_idxs = valid_input_concept_idxs.union(
            set(self.data.concept_df.query('vocabulary == "phecode"').index)
        )
        self.data.record_df[self.data.record_df.concept_id_idx.isin(valid_concept_idxs)]

        # TODO: unfortunately we have to drop duplicates here (instead of data.py) if we want to be
        # able to filter out hes/gp records in the datamodule. This requires merging in the origin
        # column which is rather slow. This could be optimized if necessary.
        self.data.record_df = self.data.record_df.merge(
            self.data.concept_df["origin"], left_on="concept_id_idx", right_index=True
        )
        self.data.record_df = self.data.record_df.drop_duplicates(
            subset=[c for c in self.data.record_df.columns if c != "origin"]
        ).drop("origin", axis=1)

        # generate static datasets
        # train dataset is required for incidence calculation for classifier init
        print("Generating train dataset...")
        # temporally set t0_mode to recruitment for consistent incidence calculation and phecode
        # selection
        t0_mode = self.t0_mode
        self.t0_mode = "recruitment"
        self.train_dataset = self.get_graph_records_dataset(set="train")
        self.t0_mode = t0_mode
        print("Generating valid dataset...")
        self.valid_dataset = self.get_graph_records_dataset(set="valid")

        self.preprocess_graph()

        self.graph_sampler_kwargs["subgraph_nodes"] = self.record_node_indices
        self.graph_sampler_kwargs["label_nodes"] = self.label_node_indices
        self.graph_collator = GraphCollator(
            graph=self.graph,
            graph_sampler=self.graph_sampler,
            graph_sampler_kwargs=self.graph_sampler_kwargs,
        )

    def sparsify_records_df(self, df, data_col="event", suffix=""):
        data = df[data_col].values
        if data_col == "event":
            data = data.astype(int)
            assert data.max() == 1
        else:
            data = data.astype(float)

        coo = coo_matrix(
            (data, (df["eid_idx"].values, df["concept_id_idx"].values)),
            shape=(len(self.le_eids.classes_), len(self.le_concepts.classes_)),
        )

        cols = [f"{c}{suffix}" for c in self.le_concepts_cols]
        df_sparse = pd.DataFrame.sparse.from_spmatrix(
            coo, columns=cols, index=self.le_eids.classes_
        )
        df_sparse.index.rename("eid", inplace=True)
        return df_sparse

    def sample_from_age_dist(self):
        ages = np.random.choice(
            self.data.covariates["age_at_recruitment_f21022_0_0"], size=(len(self.data.eid_df),)
        ).astype("timedelta64[Y]")
        eid_df = self.data.eid_df.copy()
        eid_df["age"] = ages
        eid_df["t0_date"] = eid_df["birth_date"].values + eid_df["age"].values

        return eid_df

    def sample_from_censoring_dist(self, apply_constraints=False):
        censoring_dist = (self.data.eid_df.exit_date - self.data.eid_df.recruitment_date).values
        censoring_sampled = np.random.choice(censoring_dist, size=(len(self.data.eid_df),))

        eid_df = self.data.eid_df.copy()
        eid_df["t0_date"] = self.data.eid_df.exit_date - censoring_sampled

        if apply_constraints:
            eid_df = self.apply_age_constraint(eid_df)

        eid_df["age"] = (eid_df["t0_date"] - eid_df["birth_date"]).astype("timedelta64[Y]")

        return eid_df

    def shift_t0(self, by=5, apply_constraints=False, forward_only=False):
        # get t0_date -> apply random shit -> correct if out of bounds
        if forward_only:
            shifts = np.random.uniform(low=0, high=by, size=(len(self.data.eid_df),)).astype("timedelta64[Y]")
        else:
            shifts = np.random.uniform(low=-by, high=by, size=(len(self.data.eid_df),)).astype("timedelta64[Y]")
        shifted_t0 = self.data.eid_df["recruitment_date"] + shifts

        eid_df = self.data.eid_df.copy()
        eid_df["t0_date"] = shifted_t0

        if apply_constraints:
            eid_df = self.apply_age_constraint(eid_df)

        eid_df["age"] = (eid_df["t0_date"] - eid_df["birth_date"]).astype("timedelta64[Y]")

        return eid_df

    def apply_age_constraint(self, eid_df):
        eid_df["age"] = eid_df["t0_date"] - eid_df["birth_date"]
        eid_df["original_age"] = eid_df["recruitment_date"] - eid_df["birth_date"]
        # reset those that fall out of the OG age dist:
        min_age = eid_df.original_age.min()
        max_age = eid_df.original_age.max()
        idxs = eid_df.query("age < @min_age | age > @max_age").index.values
        eid_df.loc[idxs, 't0_date'] = eid_df.loc[idxs, 'recruitment_date']
        eid_df.drop(['original_age'], axis=1, inplace=True)
        return eid_df

    def get_records_data(self, t0, split):
        """
        Generate a records dataset with t0
        :param t0:  ['recruitment', 'random_age', 'random_censoring']
        :return:
        """
        records_tte = self.record_df_by_split[split].assign(event=1)

        if t0 == "recruitment":
            records_tte = records_tte.merge(
                self.data.eid_df["recruitment_date"], left_on="eid_idx", right_index=True
            )
            records_tte.rename(columns={"recruitment_date": "t0_date"}, inplace=True)
        elif t0.startswith("random"):
            if t0 == "random_age":
                eid_df = self.sample_from_age_dist()
            elif t0 == "random_censoring":
                eid_df = self.sample_from_censoring_dist(apply_constraints=False)
            elif t0 == "random_censoring_with_constraints":
                eid_df = self.sample_from_censoring_dist(apply_constraints=True)
            elif t0.startswith("random_shift"):
                by = int(t0.split('_')[-1])
                if 'forward' in t0:
                    eid_df = self.shift_t0(by, apply_constraints=True, forward_only=True)
                else:
                    eid_df = self.shift_t0(by, apply_constraints=True)
            else:
                assert False

            # attention: it's important to use pd.join here (instead of merge) because joins keeps
            # the order of records_tte and we will assume later that records_tte is ordered by date
            records_tte = records_tte.join(eid_df["t0_date"], on="eid_idx", how="left", sort=False)

            # remove records from individuals with t0 after exit date
            valid_eids = set(eid_df.index)
            records_tte = records_tte[records_tte.eid_idx.isin(valid_eids)]

            assert np.min(np.diff(records_tte.date[~records_tte.date.isna()].values)) >= 0
        elif isinstance(t0, datetime.date):
            records_tte["t0_date"] = t0
        else:
            raise ValueError("t0 needs to be  [`recruitment`, `random`] or a datetime obj")

        records_tte["buffered_t0_date"] = records_tte.t0_date - self.buffer_period

        records_t0 = records_tte.query("date<=buffered_t0_date").drop(
            ["t0_date", "buffered_t0_date"], axis=1
        )
        records_later = records_tte.query("date>t0_date")
        records_later = records_later.drop_duplicates(
            subset=("eid_idx", "concept_id_idx"), keep="first", inplace=False
        )

        times = records_later["date"].values - records_later["t0_date"].values
        times = (times / np.timedelta64(1, "D")) / 365.25
        records_later["time"] = times
        records_later.drop(["t0_date", "buffered_t0_date"], axis=1, inplace=True)

        records_t0 = self.sparsify_records_df(records_t0, data_col="event", suffix="")
        records_times = self.sparsify_records_df(
            records_later, data_col="time", suffix="_event_time"
        )

        if self.task == "binary":
            records_events = self.tte_to_binary(records_times)
        else:
            records_events = self.sparsify_records_df(
                records_later, data_col="event", suffix="_event"
            )[[f"{c}_event" for c in self.labels]]

        records_times = records_times[[f"{c}_event_time" for c in self.labels]]

        assert (records_t0.index == records_times.index).all()
        assert (records_t0.index == records_events.index).all()

        covariates_df = self.data.covariates.loc[records_t0.index].copy()

        age_col = "age_at_recruitment_f21022_0_0"
        if age_col in self.covariate_cols and t0 == "random":
            eid_df.sort_values("eid", ascending=True, inplace=True)
            assert (records_t0.index == eid_df.eid.values).all()
            covariates_df[age_col] = (eid_df["age"] / np.timedelta64(1, "Y")).values

        covariates = self.covariate_preprocessor.transform(covariates_df[self.covariate_cols])

        records = records_t0[self.record_cols].copy()

        # record_extra_cols for exclusions
        record_extra_cols = [c for c in records_t0.columns if c not in self.node_ids_set]
        records_extra = records_t0[record_extra_cols].copy()

        if t0 == "recruitment":
            eid_df = self.data.eid_df.copy()
            eid_df.sort_values("eid", ascending=True, inplace=True)
            assert (records_t0.index == eid_df.eid).all()
            eid_df["t0_date"] = eid_df["recruitment_date"]
        else:
            # use eid_df with random t0 from before
            pass

        censorings = ((eid_df.exit_date - eid_df.t0_date) / np.timedelta64(1, "Y")).to_frame(
            name="cens_time"
        )

        return records, records_extra, covariates, records_events, records_times, censorings

    def tte_to_binary(self, records_later):
        labels_sparse = scipy.sparse.csr_matrix(
            records_later[[f"{c}_event_time" for c in self.labels]].sparse.to_coo(), dtype=float
        )
        labels_sparse.data = (labels_sparse.data <= self.task_kwargs["delta_years"]).astype(int)
        labels_sparse.eliminate_zeros()

        return pd.DataFrame.sparse.from_spmatrix(
            labels_sparse, index=records_later.index, columns=[f"{c}_event" for c in self.labels]
        )

    def prepare_exclusion_matrix(self, records, records_extra):
        # TODO: this merge is probably not necessary
        temp_exclusions = records.merge(records_extra, left_index=True, right_index=True)
        exclusions = temp_exclusions[self.labels]
        return exclusions

    def get_graph_records_dataset(self, set="train"):
        (
            records,
            records_extra,
            covariates,
            labels_events,
            labels_times,
            censorings,
        ) = self.get_records_data(t0="recruitment" if set != "train" else self.t0_mode, split=set)

        exclusions = self.prepare_exclusion_matrix(records, records_extra)

        # map from eids to sequential indices of selected split
        iloc_indices = records.index.get_indexer(self.eids[set])

        # csr for fast row indexing
        records_csr = scipy.sparse.csr_matrix(records.sparse.to_coo(), dtype=float)
        exclusions_csr = scipy.sparse.csr_matrix(exclusions.sparse.to_coo(), dtype=float)
        labels_events_csr = scipy.sparse.csr_matrix(labels_events.sparse.to_coo(), dtype=float)
        labels_times_csr = scipy.sparse.csr_matrix(labels_times.sparse.to_coo(), dtype=float)

        # scipy.sparse / np arrays from here on
        valid_cens_times = (censorings.cens_time > 0).values

        # only use eids with positive censoring times (filter out eids with t0 after exit_date in
        # random t0 sampling)
        iloc_indices = [i for i in iloc_indices if valid_cens_times[i]]

        records = records_csr[iloc_indices]
        exclusions = exclusions_csr[iloc_indices]
        labels_events = labels_events_csr[iloc_indices]
        labels_times = labels_times_csr[iloc_indices]
        covariates = covariates[iloc_indices]
        censorings = censorings.cens_time.values[iloc_indices]

        dataset = RecordsDataset(
            records=records,
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
            self.train_dataset = self.get_graph_records_dataset(set="train")
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
            self.test_dataset = self.get_graph_records_dataset(set="test")
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


# TODO:
'''
[ ] remove the GNN/embedding parts
[ ] define interface for img batches
[ ] make sure to retain functions such as eid lists etc
[ ] read targets as files??
[ ] assess how to best do record encodings // how much of that we really need since its only phecode outputs.
'''
class RetinaDataModule(LightningDataModule):
    def __init__(
        self,
        wandb_data: WandBData,
        partition: int,
        batch_size: int,
        num_workers: int,
        task: str, # TODO: need this??
        task_kwargs: dict, # TODO: need this??

        label_definition: dict,

        use_top_n_phecodes: int = 200,

        covariates: List[str] = [],
        filter_phecode_categories: Iterable[str] = ("Cong", "Dev", "Neonate"),
        filter_phecode_strings: Iterable[str] = ("history of",),
        filter_input_origins: Iterable[str] = [], #todo: need this??
        **kwargs,
    ):
        super().__init__()
        self.partition = partition
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task = task
        self.task_kwargs = task_kwargs
        self.label_definition = label_definition
        self.graph_collator = None
        self.use_top_n_phecodes = use_top_n_phecodes
        self.covariate_cols = covariates
        self.filter_phecode_categories = filter_phecode_categories
        self.filter_phecode_strings = filter_phecode_strings
        self.filter_input_origins = filter_input_origins

        self.data = wandb_data # todo: replace by baseline info + covariates directly

        # todo: do we need those?
        self.le_eids, self.cat_type_eid = self.data.record_encodings["eid"]
        self.le_concepts, self.cat_type_concepts = self.data.record_encodings["concept_id"]
        self.le_concepts_cols = self.le_concepts.classes_

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
        # prepare labels
        assert len(self.label_definition) > 0
        self.labels = []

        if self.label_definition["all_cause_death"] == True:
            self.labels += ["OMOP_4306655"]  # concept_id for "all_cause_death"
        if self.label_definition["phecodes"] == True:
            concept_idxer = self.data.concept_df.concept_id.apply(
                lambda cid: cid.startswith("phecode")
            )
            # get phecodes and corresponding concept_id_idxs
            phecode_concepts = self.data.concept_df[concept_idxer].copy().reset_index()
            phecode_concepts["phecode"] = phecode_concepts.concept_id.apply(
                lambda s: s.split("_")[1]
            )
            # merge with phecode_definitions to get phecode_categories
            phecode_concepts = phecode_concepts.merge(
                self.data.phecode_definitions, left_on="phecode", right_on="phecode"
            )
            phecode_concepts = phecode_concepts[
                ~phecode_concepts.phecode_category.isin(self.filter_phecode_categories)
            ]
            phecode_concepts = phecode_concepts[
                ~phecode_concepts.phecode_string.apply(
                    lambda s: any(
                        [
                            c.lower().strip() in s.lower().strip()
                            for c in self.filter_phecode_strings
                        ]
                    )
                )
            ]

            # filtered list of concept_id_idxs
            phecode_concept_id_idxes = set(phecode_concepts.concept_id_idx.values)

            top_n_phecodes_idxes = (
                self.data.record_df[
                    self.data.record_df["concept_id_idx"].isin(phecode_concept_id_idxes)
                ]["concept_id_idx"]
                .value_counts()[: self.use_top_n_phecodes]
                .index.values.tolist()
            )
            top_n_phecodes = self.data.concept_df.loc[
                top_n_phecodes_idxes
            ].concept_id.values.tolist()
            self.labels += top_n_phecodes
        if len(self.label_definition["custom"]) > 0:
            self.labels += self.label_definition["custom"]

        phecode_lookup = self.data.phecode_definitions.set_index("phecode")

        def map_label(l):
            if "phecode" in l:
                definition = phecode_lookup.loc[l.split("phecode_")[1]]
                return f"{l} - {definition.phecode_string}"
            return l

        fix_str = lambda s: s.replace(".", "-").replace("/", "-")

        self.label_mapping = {fix_str(l): fix_str(map_label(l)) for l in self.labels}

    def prepare_data(self):
        self.eids = self.data.eid_dict[self.partition]

        self.record_df_by_split = dict()
        for split in ("train", "valid", "test"):
            split_eid_idxs = self.data.eid_df[self.data.eid_df.eid.isin(self.eids[split])].index
            # todo make intersection w/ retina here??
            subsplit_df = self.data.record_df[self.data.record_df.eid_idx.isin(split_eid_idxs)]
            self.record_df_by_split[split] = subsplit_df

        self.prepare_labels()

        labels = [l for l in self.labels if l in self.le_concepts_cols]
        assert set(labels) == set(self.labels)

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
        # todo: get this information directly from file
        (
            records,
            records_extra,
            covariates,
            labels_events,
            labels_times,
            censorings,
        ) = self.get_records_data(t0="recruitment" if set != "train" else self.t0_mode, split=set)

        # todo: get this information from baseline file
        exclusions = self.prepare_exclusion_matrix(records, records_extra)

        # map from eids to sequential indices of selected split
        # todo: edit to get idx of baseline file
        iloc_indices = records.index.get_indexer(self.eids[set])

        # csr for fast row indexing
        records_csr = scipy.sparse.csr_matrix(records.sparse.to_coo(), dtype=float)
        exclusions_csr = scipy.sparse.csr_matrix(exclusions.sparse.to_coo(), dtype=float)
        labels_events_csr = scipy.sparse.csr_matrix(labels_events.sparse.to_coo(), dtype=float)
        labels_times_csr = scipy.sparse.csr_matrix(labels_times.sparse.to_coo(), dtype=float)

        # scipy.sparse / np arrays from here on
        valid_cens_times = (censorings.cens_time > 0).values

        # only use eids with positive censoring times (filter out eids with t0 after exit_date in
        # random t0 sampling)
        iloc_indices = [i for i in iloc_indices if valid_cens_times[i]]

        records = records_csr[iloc_indices]
        exclusions = exclusions_csr[iloc_indices]
        labels_events = labels_events_csr[iloc_indices]
        labels_times = labels_times_csr[iloc_indices]
        covariates = covariates[iloc_indices]
        censorings = censorings.cens_time.values[iloc_indices]

        dataset = RetinalFundusDataset(
            records=records,
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
