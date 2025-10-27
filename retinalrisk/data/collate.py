from collections import abc as container_abcs
from dataclasses import dataclass

import torch
#from torch._six import string_classes


@dataclass
class Batch:
    data: torch.Tensor
    covariates: torch.Tensor
    exclusions: torch.Tensor
    events: torch.Tensor
    times: torch.Tensor
    censorings: torch.Tensor
    eids: torch.Tensor


class ImgCollator:
    def __init__(
            self,
    ):
        super().__init__()

    @staticmethod
    def default_collate(batch):
        return torch.utils.data.default_collate(batch)

    def __call__(self, batch):
        # collate what's in batch
        batch = self.default_collate(batch)

        (imgs, covariates, events, times, exclusions, censorings, eids) = batch

        return Batch(
            data=imgs,
            covariates=covariates,
            exclusions=exclusions,
            events=events,
            times=times,
            censorings=censorings,
            eids=eids,
        )

