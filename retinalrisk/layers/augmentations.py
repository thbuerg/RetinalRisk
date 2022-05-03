import torch
from GCL.augmentors.functional import drop_feature
from torch_geometric.data import HeteroData
from torch_sparse import index_select_nnz


class HeteroEdgeRemoving:
    def __init__(self, pe: float):
        super().__init__()
        self.pe = pe

    def augment(self, g: HeteroData) -> HeteroData:
        g_aug = HeteroData()

        for node_key, x in g.x_dict.items():
            g_aug[node_key].x = x.clone()
            g_aug[node_key]["node_ids"] = g[node_key]["node_ids"].copy()

        for edge_key, adj in g.adj_t_dict.items():
            keep_edges = (
                (torch.rand((adj.nnz(),), device=adj.device()) < self.pe).nonzero().flatten()
            )
            adj_dropout = index_select_nnz(adj, keep_edges, layout="coo")
            g_aug[edge_key].adj_t = adj_dropout

        return g_aug


class HeteroFeatureMasking:
    def __init__(self, pf: float):
        super().__init__()
        self.pf = pf

    def augment(self, g: HeteroData) -> HeteroData:
        g_aug = g.clone()

        for node_key, x in g.x_dict.items():
            g_aug[node_key].x = drop_feature(x.clone(), self.pf)

        for edge_key, adj in g.adj_t_dict.items():
            g_aug[edge_key].adj_t = adj

        return g_aug
