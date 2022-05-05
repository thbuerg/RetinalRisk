import os
import glob

from typing import Optional

import numpy as np
import pandas as pd
import scipy
import scipy.sparse
import torch
import PIL

from torchvision import transforms

from retinalrisk.transforms.transforms import AdaptiveRandomCropTransform


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


class RetinalFundusDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            img_map: pd.DataFrame,
            exclusions: scipy.sparse.csr_matrix,
            labels_events: scipy.sparse.csr_matrix,
            labels_times: scipy.sparse.csr_matrix,
            covariates: Optional[scipy.sparse.csr_matrix] = None,
            censorings: Optional[np.array] = None,
            eids: Optional[np.array] = None,
            visit=0,
            crop_ratio: Optional[float] = 0.3,
            img_size_to_gpu: Optional[int] = 512,
            extension='.png',
    ):
        super().__init__()
        self.retina_map = img_map
        self.crop_ratio = crop_ratio
        self.img_size_to_gpu = img_size_to_gpu
        self.eids = eids

        # build idx matching table to be faster in get_item:
        self.retina_map = self.retina_map.merge(censorings.loc[self.eids].reset_index()[['eid']].reset_index(),
                                                how='left', on='eid') \
            .rename({'index': 'unique_eid_idx'}, axis=1)

        # filter all data:
        eid_idx = self.retina_map['unique_eid_idx'].unique()
        self.exclusions = exclusions.iloc[eid_idx]
        self.covariates = covariates[eid_idx]
        self.labels_events = labels_events.iloc[eid_idx]
        self.labels_times = labels_times.iloc[eid_idx]
        self.censorings = censorings.iloc[eid_idx]

        self.visit = visit
        self.extension = extension

        # set up transforms
        self.transforms = transforms.Compose([
                # AdaptiveRandomCropTransform(crop_ratio=self.crop_ratio,
                #                             out_size=self.img_size_to_gpu,
                #                             interpolation=PIL.Image.BICUBIC),
                transforms.Resize(self.img_size_to_gpu*1.1),
                transforms.CenterCrop(self.img_size_to_gpu),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def loader(self, path):
        return self._RGBA_png_loader(path)

    @staticmethod
    def _RGBA_png_loader(path):
        with open(path, 'rb') as f:
            img = PIL.Image.open(f)
            return img.convert('RGB')

    def _transforms_dummy(self, img):
        return img

    def __getitem__(self, idx):
        # todo: check if this could work from multiple paths...
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = self.retina_map['file_path'].values[idx]
        eid_idx = self.retina_map['unique_eid_idx'].values[idx]
        img = self.loader(path)
        img = self.transforms(img)

        exclusions = torch.Tensor(self.exclusions.values[eid_idx])
        labels_events = torch.Tensor(self.labels_events.values[eid_idx])
        labels_times = torch.Tensor(self.labels_times.values[eid_idx])

        eids = torch.LongTensor([self.eids[eid_idx]])

        covariates = None
        if self.covariates is not None:
            covariates = torch.Tensor(self.covariates[eid_idx])

        censorings = None
        if self.censorings is not None:
            censorings = torch.Tensor(self.censorings.values[eid_idx])

        data_tuple = (img, covariates)
        labels_tuple = (labels_events, labels_times, exclusions, censorings, eids)

        return data_tuple, labels_tuple

    def __len__(self):
        return self.retina_map.shape[0]

