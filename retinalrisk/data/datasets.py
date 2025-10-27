from typing import Optional

import numpy as np
import pandas as pd
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
            augmentations: Optional[list] = [],
            covariates: Optional[scipy.sparse.csr_matrix] = None,
            censorings: Optional[np.array] = None,
            eids: Optional[np.array] = None,
            img_crop_ratio: Optional[float] = 0.9,
            img_size_to_gpu: Optional[int] = 324,
            extension='.png',
    ):
        super().__init__()
        self.retina_map = img_map
        self.img_crop_ratio = img_crop_ratio
        self.img_size_to_gpu = img_size_to_gpu
        self.eids = eids

        self.exclusions = exclusions
        self.covariates = covariates
        self.labels_events = labels_events
        self.labels_times = labels_times
        self.censorings = censorings

        self.extension = extension

        # set up transforms
        self.transforms = transforms.Compose([
                AdaptiveRandomCropTransform(crop_ratio=self.img_crop_ratio,
                                            out_size=self.img_size_to_gpu,
                                            interpolation=PIL.Image.BICUBIC),
                                             ] + augmentations + [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def loader(self, path):
        return self._RGB_png_loader(path)

    @staticmethod
    def _RGB_png_loader(path):
        with open(path, 'rb') as f:
            img = PIL.Image.open(f)
            return img.convert('RGB')

    def _transforms_dummy(self, img):
        return img

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = self.retina_map['file_path'].values[idx]
        eid = self.retina_map['eid'].values[idx]
        img = self.loader(path)
        img = self.transforms(img)

        exclusions = torch.Tensor(self.exclusions.loc[eid].values)
        labels_events = torch.Tensor(self.labels_events.loc[eid].values)
        labels_times = torch.Tensor(self.labels_times.loc[eid].values)

        covariates = None
        if self.covariates is not None:
            covariates = torch.Tensor(self.covariates.loc[eid].values)

        censorings = None
        if self.censorings is not None:
            censorings = torch.Tensor(self.censorings.loc[eid].values)

        return (img, covariates, labels_events, labels_times, exclusions, censorings, eid)

    def __len__(self):
        return self.retina_map.shape[0]

