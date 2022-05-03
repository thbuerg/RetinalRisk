# %%
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader.utils import (
    edge_type_to_str,
    filter_data,
    filter_hetero_data,
    to_csc,
    to_hetero_csc,
)
from torch_geometric.typing import EdgeType, NodeType

NumNeighbors = Union[List[int], Dict[EdgeType, List[int]]]


class SubgraphNeighborSampler:
    def __init__(
        self,
        data: Data,
        num_neighbors: NumNeighbors,
        subgraph_nodes: Union[Optional[Tensor], NodeType, Tuple[NodeType, Optional[Tensor]]] = None,
        label_nodes: Union[Optional[Tensor], NodeType, Tuple[NodeType, Optional[Tensor]]] = None,
        replace: bool = False,
        directed: bool = False,
        input_node_type: Optional[str] = "0",
    ):
        self.data_cls = data.__class__
        self.data = data
        self.num_neighbors = num_neighbors
        self.replace = replace
        self.directed = directed
        self.subgraph_nodes = torch.LongTensor(subgraph_nodes)
        self.label_nodes = torch.LongTensor(label_nodes)

        if isinstance(data, Data):
            # Convert the graph data into a suitable format for sampling.
            self.colptr, self.row, self.perm = to_csc(data)
            assert isinstance(num_neighbors, (list, tuple))

        elif isinstance(data, HeteroData):
            # Convert the graph data into a suitable format for sampling.
            # NOTE: Since C++ cannot take dictionaries with tuples as key as
            # input, edge type triplets are converted into single strings.
            out = to_hetero_csc(data, device="cpu")
            self.colptr_dict, self.row_dict, self.perm_dict = out

            self.node_types, self.edge_types = data.metadata()
            if isinstance(num_neighbors, (list, tuple)):
                num_neighbors = {key: num_neighbors for key in self.edge_types}
            assert isinstance(num_neighbors, dict)
            self.num_neighbors = {
                edge_type_to_str(key): value for key, value in num_neighbors.items()
            }

            self.num_hops = max(len(v) for v in self.num_neighbors.values())

            assert isinstance(input_node_type, str)
            self.input_node_type = input_node_type

        else:
            raise TypeError(f"NeighborLoader found invalid type: {type(data)}")

    def __call__(self):
        # TODO: add label_nodes
        if issubclass(self.data_cls, Data):
            node, row, col, edge = torch.ops.torch_sparse.neighbor_sample(
                self.colptr,
                self.row,
                self.subgraph_nodes,
                self.num_neighbors,
                self.replace,
                self.directed,
            )
            data = filter_data(self.data, node, row, col, edge, self.perm)

            return data

        elif issubclass(self.data_cls, HeteroData):
            sample_fn = torch.ops.torch_sparse.hetero_neighbor_sample
            node_dict, row_dict, col_dict, edge_dict = sample_fn(
                self.node_types,
                self.edge_types,
                self.colptr_dict,
                self.row_dict,
                {self.input_node_type: self.subgraph_nodes},
                self.num_neighbors,
                self.num_hops,
                self.replace,
                self.directed,
            )
            data = filter_hetero_data(
                self.data, node_dict, row_dict, col_dict, edge_dict, self.perm_dict
            )

            return data

            # TODO: fix add record_indices


class DummySampler:
    def __init__(
        self,
        data: HeteroData,
        subgraph_nodes: Union[Optional[Tensor], NodeType, Tuple[NodeType, Optional[Tensor]]] = None,
        label_nodes: Union[Optional[Tensor], NodeType, Tuple[NodeType, Optional[Tensor]]] = None,
        *args,
        **kwargs,
    ):
        self.data = data
        self.subgraph_nodes = subgraph_nodes
        self.label_nodes = label_nodes

    def __call__(self):
        return self.data, self.subgraph_nodes, self.label_nodes
