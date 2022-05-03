import os
import glob

from typing import Optional

import numpy as np
import pandas as pd
import scipy
import scipy.sparse
import torch

from PIL import Image

class RecordsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        records: scipy.sparse.csr_matrix,
        exclusions: scipy.sparse.csr_matrix,
        labels_events: scipy.sparse.csr_matrix,
        labels_times: scipy.sparse.csr_matrix,
        covariates: Optional[scipy.sparse.csr_matrix] = None,
        censorings: Optional[np.array] = None,
        eids: Optional[np.array] = None # TODO: are these real eids?
    ):
        self.records = records
        self.exclusions = exclusions
        self.covariates = covariates
        self.labels_events = labels_events
        self.labels_times = labels_times
        self.censorings = censorings
        self.eids = eids

    def __len__(self):
        return self.records.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        records = torch.Tensor(self.records[idx].todense())
        exclusions = torch.Tensor(self.exclusions[idx].todense())
        labels_events = torch.Tensor(self.labels_events[idx].todense())
        labels_times = torch.Tensor(self.labels_times[idx].todense())

        eids = torch.LongTensor([self.eids[idx]])

        covariates = None
        if self.covariates is not None:
            if not isinstance(idx, list):
                idx = [idx]
            covariates = torch.Tensor(self.covariates[idx])

        censorings = None
        if self.censorings is not None:
            if not isinstance(idx, list):
                idx = [idx]
            censorings = torch.Tensor(self.censorings[idx])

        data_tuple = (records, covariates)
        labels_tuple = (labels_events, labels_times, exclusions, censorings, eids)

        return data_tuple, labels_tuple


# TODO:
'''
[ ] add riskiano retina dataset -> be minimal in transfer and keep only the parts that have the img loading functions.
[ ] research on efficient pytorch img loading pipelines (with augmentations??)
[ ] 
'''
class RetinalFundusDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            retina_map: pd.DataFrame,
            exclusions: scipy.sparse.csr_matrix,
            labels_events: scipy.sparse.csr_matrix,
            labels_times: scipy.sparse.csr_matrix,
            covariates: Optional[scipy.sparse.csr_matrix] = None,
            censorings: Optional[np.array] = None,
            eids: Optional[np.array] = None,
            visit=0,
            extension='.png',
    ):
        super().__init__()
        self.retina_map = retina_map

        self.exclusions = exclusions
        self.covariates = covariates
        self.labels_events = labels_events
        self.labels_times = labels_times
        self.censorings = censorings

        self.eids = eids

        self.visit = visit
        self.extension = extension

    def _png_loader(self, path):
        return self._RGBA_png_loader(path)

    @staticmethod
    def _RGBA_png_loader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGBA')

    def __getitem__(self, idx):
        # todo: check if this could work from multiple paths...
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = self.retina_map['file_path'].values[idx]
        img = self.loader(path)

        exclusions = torch.Tensor(self.exclusions[idx].todense())
        labels_events = torch.Tensor(self.labels_events[idx].todense())
        labels_times = torch.Tensor(self.labels_times[idx].todense())

        eids = torch.LongTensor([self.eids[idx]])

        covariates = None
        if self.covariates is not None:
            if not isinstance(idx, list):
                idx = [idx]
            covariates = torch.Tensor(self.covariates[idx])

        censorings = None
        if self.censorings is not None:
            if not isinstance(idx, list):
                idx = [idx]
            censorings = torch.Tensor(self.censorings[idx])

        data_tuple = (img, covariates)
        labels_tuple = (labels_events, labels_times, exclusions, censorings, eids)

        return data_tuple, labels_tuple

    def __len__(self):
        return self.retina_map.shape[0]

