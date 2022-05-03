from collections import abc as container_abcs
from dataclasses import dataclass

import torch
from torch._six import string_classes
from torch_geometric.data import Data


@dataclass
class Batch:
    graph: Data
    record_indices: torch.Tensor
    label_indices: torch.Tensor
    records: torch.Tensor
    covariates: torch.Tensor
    exclusions: torch.Tensor
    events: torch.Tensor
    times: torch.Tensor
    censorings: torch.Tensor
    eids: torch.Tensor


class GraphCollator:
    def __init__(
        self,
        graph,
        graph_sampler=None,
        graph_sampler_kwargs: dict = {},
    ):
        super().__init__()
        self.graph = graph
        self.graph_sampler = graph_sampler(graph, **graph_sampler_kwargs)

    @staticmethod
    def default_collate(batch):
        r"""Puts each datamodules field into a tensor with outer dimension batch size"""
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in batch)
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.cat(batch, 0, out=out)
        elif (
            elem_type.__module__ == "numpy"
            and elem_type.__name__ != "str_"
            and elem_type.__name__ != "string_"
        ):
            elem = batch[0]
            if elem_type.__name__ == "ndarray":

                return GraphCollator.default_collate([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            return {key: GraphCollator.default_collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
            return elem_type(*(GraphCollator.default_collate(samples) for samples in zip(*batch)))
        if isinstance(batch, container_abcs.Sequence):
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError("each element in list of batch should be of equal size")
            transposed = zip(*batch)
            return [GraphCollator.default_collate(samples) for samples in transposed]

    def __call__(self, batch):
        # collate what's in batch
        batch = self.default_collate(batch)

        # if isinstance(batch, container_abcs.Sequence):
        #     return self.graph_sampler(), *batch
        # else:
        (records, covariates), (events, times, exclusions, censorings, eids) = batch
        graph, record_indices, label_indices = self.graph_sampler()

        return Batch(
            graph=graph,
            record_indices=record_indices,
            label_indices=label_indices,
            records=records,
            covariates=covariates,
            exclusions=exclusions,
            events=events,
            times=times,
            censorings=censorings,
            eids=eids,
        )
