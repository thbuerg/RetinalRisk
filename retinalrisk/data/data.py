import os
import glob
import pathlib
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import wandb
import yaml
import zstandard


def load_wandb_artifact(
    identifier, run: wandb.sdk.wandb_run.Run = None, project: str = None, entity: str = None
):
    assert run is not None or (project is not None and entity is not None)

    if run is None:
        api = wandb.Api()
        artifact = api.artifact(f"{entity}/{project}/{identifier}")
    else:
        artifact = run.use_artifact(identifier)

    return artifact


def get_path_from_wandb(reference: str, data_root: Optional[str] = None):
    if data_root is not None:
        stub = pathlib.Path(reference.split("file://")[1]).name
        path = pathlib.Path(data_root, stub)
    else:
        path = pathlib.Path(reference.split("file://")[1])
    print(path)

    assert path.exists(), f"Path not found: {path}"
    return path


class WandBBaselineData:
    def __init__(
        self,
        eid_dict_name: str = "eids:latest",
        covariates_name: str = "baseline_covariates:latest",
        outcomes_name: str = "baseline_outcomes:latest",
        phecode_definitions_name: str = "phecode_definitions:latest",
        data_root: Optional[str] = None,
        wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
        wandb_entity: Optional[str] = "cardiors",
        wandb_project: Optional[str] = "Retina",
    ):

        self.data_root = data_root

        self.eid_dict_name = eid_dict_name
        self.covariates_name = covariates_name
        self.outcomes_name = outcomes_name
        self.phecode_definitions_name = phecode_definitions_name

        self.wandb_run = wandb_run
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project

        self.phecode_definitions = self.load_phecode_definitions()
        self.eid_dict = self.load_eid_dict(valid_eids=None)
        self.covariates = self.load_covariates()
        self.outcomes = self.load_outcomes()


    def load_phecode_definitions(self):
        phecode_artifact = self._load_artifact(self.phecode_definitions_name)

        phecodes_entries = phecode_artifact.manifest.entries
        phecodes_reference = phecodes_entries["phecode_definitions"]
        phecodes_path = get_path_from_wandb(phecodes_reference.ref, data_root=self.data_root)

        phecode_df = pd.read_feather(phecodes_path).set_index("index")
        return phecode_df

    def load_outcomes(self):
        artifact = self._load_artifact(self.outcomes_name)
        entry = artifact.manifest.entries[self.outcomes_name.split(":")[0]]
        path = get_path_from_wandb(entry.ref, data_root=self.data_root)
        outcomes_df = pd.read_feather(path).set_index("eid")

        # drop all object cols for now
        # TODO fix encoding for object columns:
        outcomes_df = outcomes_df.apply(pd.to_numeric, errors="ignore")

        # todo revise and delete
        # outcomes_df = outcomes_df[
        #     [c for c in outcomes_df.columns.to_list() if "date" not in c]
        # ]
        #
        # for c in outcomes_df.columns:
        #     if outcomes_df[c].dtype != "float64":
        #         outcomes_df[c] = outcomes_df[c].astype("category")
        #
        return outcomes_df

    def load_eid_dict(self, valid_eids=None):
        artifact = self._load_artifact(self.eid_dict_name)
        entry = artifact.manifest.entries[self.eid_dict_name.split(":")[0]]
        path = get_path_from_wandb(entry.ref, data_root=self.data_root)
        eid_dict = yaml.load(open(path), Loader=yaml.CLoader)

        if valid_eids is not None:
            for partition, sets in eid_dict.items():
                for split, eids in sets.items():
                    eid_dict[partition][split] = np.array(
                        [eid for eid in eids if eid in valid_eids]
                    )

        return eid_dict

    def load_covariates(self):
        artifact = self._load_artifact(self.covariates_name)
        entry = artifact.manifest.entries[self.covariates_name.split(":")[0]]
        path = get_path_from_wandb(entry.ref, data_root=self.data_root)
        covariates_df = pd.read_feather(path).set_index("eid")

        # drop all object cols for now
        # TODO fix encoding for object columns:
        covariates_df = covariates_df.apply(pd.to_numeric, errors="ignore")

        covariates_df = covariates_df[
            [c for c in covariates_df.columns.to_list() if "date" not in c]
        ]

        for c in covariates_df.columns:
            if covariates_df[c].dtype != "float64":
                covariates_df[c] = covariates_df[c].astype("category")

        return covariates_df

    def _load_artifact(self, identifier: str):
        return load_wandb_artifact(
            identifier,
            self.wandb_run,
            self.wandb_project,
            self.wandb_entity,
        )


def get_or_load_wandbdataobj(
    data_root, identifier="WandBBaselineData:latest", run=None, project=None, entity=None, **kwargs
):
    # log this as artifact:
    artifact = load_wandb_artifact(identifier, run=run, project=project, entity=entity)
    ref = artifact.manifest.entries["WandBBaselineData"].ref
    stub = pathlib.Path(ref.split("file://")[1]).name
    artifact_path = pathlib.Path(data_root, stub)
    print(artifact_path)

    try:
        with open(artifact_path, "rb") as fh:
            dctx = zstandard.ZstdDecompressor()
            with dctx.stream_reader(fh) as decompressor:
                data = pickle.loads(decompressor.read())
    except Exception as err:
        print(err)
        data = WandBBaselineData(
            data_root=data_root, wandb_run=run, wandb_entity=entity, wandb_project=project, **kwargs
        )
        # then pickle it:
        with open(artifact_path, "wb") as fh:
            cctx = zstandard.ZstdCompressor()
            with cctx.stream_writer(fh) as compressor:
                compressor.write(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))

    return data


def get_retina_eid_map(data_root, visit=0, extension='.png'):
    """
    Get a eid->fp mapping for all retina pics in dir data_root.
    :param data_root:
    :param visit:
    :return:
    """
    # build eid to file map:
    if visit is not None:
        files = [fp for fp in sorted(glob.glob(os.path.join(data_root,
                                                            f'*{extension}' if extension is not None else '*')))
                 if f'_{visit}_' in fp]
    else:
        files = [fp for fp in sorted(glob.glob(os.path.join(data_root,
                                                            f'*{extension}' if extension is not None else '*')))]

    eids = [os.path.basename(fp).split('_')[0] for fp in files]
    eids = np.asarray(eids).astype(int)

    eid_map = pd.DataFrame(np.stack([eids, np.asarray(files)], axis=-1),
                           columns=['eid', 'file_path'])
    eid_map = eid_map.astype({'eid': 'int32'})
    return eid_map

