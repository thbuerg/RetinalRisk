from collections import abc as container_abcs
from dataclasses import dataclass

import torch
from torch._six import string_classes


@dataclass
class Batch:
    data: torch.Tensor
    covariates: torch.Tensor
    exclusions: torch.Tensor
    events: torch.Tensor
    times: torch.Tensor
    censorings: torch.Tensor
    eids: torch.Tensor


# TODO:
'''
[x] understand what is happening in graph collator 
[x] adapt for img input!
[x] make sure batch class is populated and returned
[ ] WHERE TO ADD THE COLLATOR???
[ ] test
'''
class ImgCollator:
    def __init__(
            self,
    ):
        super().__init__()

    @staticmethod
    def default_collate(batch):
        return torch.utils.data.default_collate(batch)
        # r"""Puts each datamodules field into a tensor with outer dimension batch size"""
        # elem = batch[0]
        # # elem_type = type(elem)
        # if isinstance(elem, torch.Tensor):
        #     return torch.stack(batch, 0)
        # else:
        #     print(type(elem))
        #     raise NotImplementedError()
        # elif (
        #         elem_type.__module__ == "numpy"
        #         and elem_type.__name__ != "str_"
        #         and elem_type.__name__ != "string_"
        # ):
        #     elem = batch[0]
        #     if elem_type.__name__ == "ndarray":
        #
        #         return ImgCollator.default_collate([torch.as_tensor(b) for b in batch])
        #     elif elem.shape == ():  # scalars
        #         return torch.as_tensor(batch)
        # elif isinstance(elem, float):
        #     return torch.tensor(batch, dtype=torch.float64)
        # elif isinstance(elem, int):
        #     return torch.tensor(batch)
        # elif isinstance(elem, string_classes):
        #     return batch
        # elif isinstance(elem, container_abcs.Mapping):
        #     return {key: ImgCollator.default_collate([d[key] for d in batch]) for key in elem}
        # elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        #     return elem_type(*(ImgCollator.default_collate(samples) for samples in zip(*batch)))
        # if isinstance(batch, container_abcs.Sequence):
        #     it = iter(batch)
        #     elem_size = len(next(it))
        #     if not all(len(elem) == elem_size for elem in it):
        #         raise RuntimeError("each element in list of batch should be of equal size")
        #     transposed = zip(*batch)
        #     return [ImgCollator.default_collate(samples) for samples in transposed]

    def __call__(self, batch):
        # collate what's in batch
        batch = self.default_collate(batch)

        # if isinstance(batch, container_abcs.Sequence):
        #     return self.graph_sampler(), *batch
        # else:
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

