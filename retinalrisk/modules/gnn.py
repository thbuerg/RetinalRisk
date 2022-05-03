from typing import Optional

import torch
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T

from retinalrisk.layers.hetero import HeteroConv
from retinalrisk.layers.wgat import WGATv2Conv


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_hidden, num_outputs, num_blocks):
        super().__init__()

        self.blocks = []

        for _ in range(num_blocks):
            self.blocks.append((torch_geometric.nn.GraphNorm(num_hidden), "x -> x"))
            self.blocks.append(
                (torch_geometric.nn.GCNConv(num_hidden, num_hidden), "x, edge_index -> x")
            )
            self.blocks.append(torch.nn.LeakyReLU(inplace=True))

        self.model = torch_geometric.nn.Sequential(
            "x, edge_index",
            [
                (torch_geometric.nn.GraphNorm(num_node_features), "x -> x"),
                (torch_geometric.nn.GCNConv(num_node_features, num_hidden), "x, edge_index -> x"),
                torch.nn.LeakyReLU(inplace=True),
            ]
            + self.blocks
            + [
                (torch_geometric.nn.GraphNorm(num_hidden), "x -> x"),
                (torch_geometric.nn.GCNConv(num_hidden, num_outputs), "x, edge_index -> x"),
            ],
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        xhat = self.model(x, edge_index)

        return xhat


class GATN(torch.nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_hidden: int,
        num_outputs: int,
        num_heads: int,
        num_blocks: int,
        dropout: float = 0,
        edge_embedder: Optional[torch.nn.Embedding] = None,
        gat_layer: torch.nn.Module = WGATv2Conv,
    ):
        super().__init__()

        self.num_node_features = num_node_features
        self.num_outputs = num_outputs
        self.edge_embedder = edge_embedder

        num_gat_hidden = num_hidden // num_heads
        num_gat_outputs = num_outputs // num_heads

        gat_inputs = "x, edge_index, edge_weight"

        edge_dim = None
        if self.edge_embedder is not None:
            gat_inputs += ", edge_attr"
            edge_dim = self.edge_embedder.embedding_dim

        assert num_blocks >= 2
        self.blocks = []
        for _ in range(num_blocks - 2):
            self.blocks.append((torch_geometric.nn.GraphNorm(num_hidden), "x -> x"))
            self.blocks.append(
                (
                    gat_layer(
                        num_hidden,
                        num_gat_hidden,
                        heads=num_heads,
                        dropout=dropout,
                        edge_dim=edge_dim,
                    ),
                    f"{gat_inputs} -> x",
                )
            )
            self.blocks.append(torch.nn.LeakyReLU(inplace=True))

        self.model = torch_geometric.nn.Sequential(
            f"{gat_inputs}",
            [
                (torch_geometric.nn.GraphNorm(num_node_features), "x -> x"),
                (
                    gat_layer(
                        num_node_features,
                        num_gat_hidden,
                        heads=num_heads,
                        dropout=dropout,
                        edge_dim=edge_dim,
                    ),
                    f"{gat_inputs} -> x",
                ),
                torch.nn.LeakyReLU(inplace=True),
            ]
            + self.blocks
            + [
                (torch_geometric.nn.GraphNorm(num_hidden), "x -> x"),
                (
                    gat_layer(
                        num_hidden,
                        num_gat_outputs,
                        heads=num_heads,
                        dropout=dropout,
                        edge_dim=edge_dim,
                    ),
                    f"{gat_inputs} -> x",
                ),
            ],
        )

    def forward(self, data):
        if self.edge_embedder is not None:
            edge_attr = self.edge_embedder(data.edge_code)
            model_args = data.x, data.edge_index, data.edge_weight, edge_attr
        else:
            model_args = data.x, data.edge_index, data.edge_weight

        xhat = self.model(*model_args)

        return xhat


class ResGATN(torch.nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_hidden: int,
        num_outputs: int,
        num_heads: int,
        num_blocks: int,
        dropout: float = 0,
        edge_embedder: Optional[torch.nn.Embedding] = None,
        gat_layer: torch.nn.Module = WGATv2Conv,
        norm_layer: torch.nn.Module = torch_geometric.nn.GraphNorm,
        act_layer: torch.nn.Module = torch.nn.LeakyReLU,
    ):
        super().__init__()

        self.num_node_features = num_node_features
        self.num_outputs = num_outputs
        self.edge_embedder = edge_embedder

        num_gat_hidden = num_hidden // num_heads

        gat_inputs = "x, edge_index, edge_weight"
        edge_dim = None
        if self.edge_embedder is not None:
            gat_inputs += ", edge_attr"
            edge_dim = self.edge_embedder.embedding_dim

        self.blocks = []

        if num_node_features != num_hidden:
            self.blocks.append((torch.nn.Linear(num_node_features, num_hidden), "x -> x"))
            self.blocks.append((norm_layer(num_hidden), "x -> x"))

        for _ in range(num_blocks):
            self.blocks.append(
                (
                    torch_geometric.nn.DeepGCNLayer(
                        conv=gat_layer(
                            num_hidden,
                            num_gat_hidden,
                            heads=num_heads,
                            dropout=dropout,
                            edge_dim=edge_dim,
                        ),
                        norm=norm_layer(num_hidden),
                        act=act_layer(),
                        dropout=dropout,
                    ),
                    f"{gat_inputs} -> x",
                )
            )

        if num_hidden != num_outputs:
            self.blocks.append((torch.nn.Linear(num_hidden, num_outputs), "x -> x"))

        self.model = torch_geometric.nn.Sequential(f"{gat_inputs}", self.blocks)

    def forward(self, data):
        if self.edge_embedder is not None:
            edge_attr = self.edge_embedder(data.edge_code)
            model_args = data.x, data.edge_index, data.edge_weight, edge_attr
        else:
            model_args = data.x, data.edge_index, data.edge_weight

        xhat = self.model(*model_args)

        return xhat


class GATIIN(torch.nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_hidden: int,
        num_outputs: int,
        num_heads: int,
        num_blocks: int,
        dropout: float = 0,
        edge_embedder: Optional[torch.nn.Embedding] = None,
        gat_layer: torch.nn.Module = WGATv2Conv,
        norm_layer: torch.nn.Module = torch_geometric.nn.GraphNorm,
        act_layer: torch.nn.Module = torch.nn.LeakyReLU,
        alpha: float = 0.2,
    ):
        super().__init__()

        self.num_node_features = num_node_features
        self.num_outputs = num_outputs
        self.edge_embedder = edge_embedder
        self.alpha = alpha
        self.act = act_layer()

        num_gat_hidden = num_hidden // num_heads

        edge_dim = None
        if self.edge_embedder is not None:
            edge_dim = self.edge_embedder.embedding_dim

        self.pre = None
        if num_node_features != num_hidden:
            self.pre = torch.nn.Linear(num_node_features, num_hidden)
        self.pre_norm = norm_layer(num_hidden)

        self.blocks = []
        self.norms = []
        for _ in range(num_blocks):
            self.blocks.append(
                gat_layer(
                    num_hidden,
                    num_gat_hidden,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                )
            )
            self.norms.append(norm_layer(num_hidden))
        self.blocks = torch.nn.ModuleList(self.blocks)
        self.norms = torch.nn.ModuleList(self.norms)

        self.post = None
        if num_hidden != num_outputs:
            self.post = torch.nn.Linear(num_hidden, num_outputs)
        self.post_norm = norm_layer(num_outputs)

    def forward(self, data):
        assert self.edge_embedder is not None
        edge_attr = self.edge_embedder(data.edge_code)

        x_0 = data.x
        if self.pre is not None:
            x_0 = torch.utils.checkpoint.checkpoint(self.pre, x_0)
        x_0 = self.pre_norm(x_0)

        x = x_0
        for block, norm in zip(self.blocks, self.norms):
            x = norm(x)
            x_hat = torch.utils.checkpoint.checkpoint(
                block, x, data.edge_index, data.edge_weight, edge_attr, preserve_rng_state=True
            )
            x_hat = block(x, data.edge_index, data.edge_weight, edge_attr)
            x_hat = self.act(x_hat)
            x = self.alpha * x_0 + (1 - self.alpha) * x_hat

        if self.post is not None:
            x = torch.utils.checkpoint.checkpoint(self.post, x)

        return x


class SequentialGAT(torch.nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_hidden: int,
        num_outputs: int,
        num_heads: int,
        num_blocks: int,
        dropout: float = 0,
        edge_embedder: Optional[torch.nn.Embedding] = None,
        gat_layer: torch.nn.Module = WGATv2Conv,
        norm_layer: torch.nn.Module = torch_geometric.nn.GraphNorm,
        act_layer: torch.nn.Module = torch.nn.LeakyReLU,
        alpha: float = 0.2,
    ):
        super().__init__()

        self.num_node_features = num_node_features
        self.num_outputs = num_outputs
        self.edge_embedder = edge_embedder
        self.alpha = alpha
        self.act = act_layer()

        num_gat_hidden = num_hidden // num_heads

        edge_dim = None
        if self.edge_embedder is not None:
            edge_dim = self.edge_embedder.embedding_dim

        self.pre = None
        if num_node_features != num_hidden:
            self.pre = torch.nn.Linear(num_node_features, num_hidden)

        self.blocks = []
        for _ in range(num_blocks):
            self.blocks.append(
                gat_layer(
                    num_hidden,
                    num_gat_hidden,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                )
            )
        self.blocks = torch.nn.ModuleList(self.blocks)

        self.post = None
        if num_hidden != num_outputs:
            self.post = torch.nn.Linear(num_hidden, num_outputs)

    def forward(self, data):
        assert self.edge_embedder is not None
        edge_attr = torch.utils.checkpoint.checkpoint(self.edge_embedder, data.edge_code)

        x = data.x
        if self.pre is not None:
            x = torch.utils.checkpoint.checkpoint(self.pre, x)

        for block in self.blocks:
            x = torch.utils.checkpoint.checkpoint(
                block, x, data.edge_index, data.edge_weight, edge_attr, preserve_rng_state=True
            )

        if self.post is not None:
            x = torch.utils.checkpoint.checkpoint(self.post, x)

        return x


class Identity(torch.nn.Module):
    def __init__(self, num_inputs: int):
        super().__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_inputs
        self.identity = torch.nn.Identity()

    def forward(self, input):
        return self.identity(input)


class SparseGCN(torch.nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_hidden: int,
        num_outputs: int,
        num_blocks: int,
        act_layer: torch.nn.Module = torch.nn.LeakyReLU,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.num_node_features = num_node_features
        self.num_outputs = num_outputs
        self.act = act_layer()

        self.pre = None
        if num_node_features != num_hidden:
            self.pre = torch.nn.Linear(num_node_features, num_hidden)

        self.blocks = []
        for _ in range(num_blocks):
            self.blocks.append(torch_geometric.nn.GCNConv(num_hidden, num_hidden, improved=True))
        self.blocks = torch.nn.ModuleList(self.blocks)

        self.post = None
        if num_hidden != num_outputs:
            self.post = torch.nn.Linear(num_hidden, num_outputs)

        self.transform = T.Compose([T.AddSelfLoops(), T.ToSparseTensor()])

    def execute_block(self, block, x, adj_t):
        x = block(x, adj_t)
        x = self.act(x)
        return x

    def forward(self, data):
        data = self.transform(data)

        x = data.x
        if self.pre is not None:
            x = torch.utils.checkpoint.checkpoint(self.pre, x)

        for block in self.blocks:
            x = torch.utils.checkpoint.checkpoint(self.execute_block, block, x, data.adj_t)

        if self.post is not None:
            x = torch.utils.checkpoint.checkpoint(self.post, x)

        return x


class HeteroGNN(torch.nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_hidden: int,
        num_outputs: int,
        num_blocks: int,
        metadata,
        act_layer: torch.nn.Module = torch.nn.LeakyReLU,
        norm_layer: torch.nn.Module = torch_geometric.nn.GraphNorm,
        gradient_checkpointing: bool = True,
        weight_norm: bool = True,
        out_norm: bool = False,
        dropout: float = 0.0,
        gcnii_alpha: float = 0.1,
        gcnii_theta: float = 0.5,
    ):
        super().__init__()

        self.act = act_layer()
        self.num_outputs = num_outputs
        self.out_norm = out_norm

        # if num_node_features != num_hidden:
        linear = torch.nn.Linear(num_node_features, num_hidden)
        if weight_norm:
            linear = torch.nn.utils.weight_norm(linear)
        self.pre = linear

        self.convs = torch.nn.ModuleList()

        def maybe_add_weight_norm(module, name):
            if weight_norm:
                return torch.nn.utils.weight_norm(module, name=name)
            else:
                return module

        for i in range(num_blocks):
            gcn_convs = {
                edge_type: maybe_add_weight_norm(
                    torch_geometric.nn.GCN2Conv(
                        num_hidden, alpha=gcnii_alpha, theta=gcnii_theta, layer=i + 1
                    ),
                    name="weight1",
                )
                for edge_type in metadata[1]
            }

            conv = HeteroConv(
                gcn_convs,
                aggr="sum",
                gradient_checkpointing=False,
            )
            self.convs.append(conv)

        self.post = None
        if num_hidden != num_outputs:
            linear = torch.nn.Linear(num_hidden, num_outputs)
            if weight_norm:
                linear = torch.nn.utils.weight_norm(linear)
            self.post = linear

        self.gradient_checkpointing = gradient_checkpointing
        self.weight_norm = weight_norm
        self.dropout = dropout

    def gradient_checkpoint_wrapper(self, module, x_dict, *args):
        if self.gradient_checkpointing:

            def forward_wrapper(*inputs):
                dummy = inputs[-1]
                inputs = inputs[:-1]
                inputs = module(*inputs)

                return inputs, dummy

            dummy = torch.ones(1, requires_grad=True)
            x_0 = x_dict["0"]
            x_0, dummy = torch.utils.checkpoint.checkpoint(forward_wrapper, x_0, *args, dummy)
            x_dict["0"] = x_0
        else:
            x_dict["0"] = module(x_dict["0"])

        return x_dict

    def execute_block(self, conv, x_vals, x_0, adj_t_dict):
        x0 = x_vals[0]
        x0 = torch.nn.functional.normalize(x0)
        x_vals = conv(torch.stack([x0]), x_0, adj_t_dict)
        x0 = x_vals[0]
        x0 = F.leaky_relu(x0)

        if self.dropout:
            x0 = F.dropout(x0, self.dropout, self.training)

        return torch.stack([x0])

    def forward(self, x_dict, adj_t_dict):
        if self.pre is not None:
            x_dict = self.gradient_checkpoint_wrapper(self.pre, x_dict)

        if self.dropout:
            x_dict = self.gradient_checkpoint_wrapper(
                F.dropout, x_dict, self.dropout, self.training
            )

        x_0 = x_dict["0"]

        for conv in self.convs:
            if self.gradient_checkpointing:

                def forward_wrapper(*inputs):
                    dummy = inputs[-1]
                    inputs = inputs[:-1]
                    inputs = self.execute_block(*inputs)

                    return inputs, dummy

                dummy = torch.ones(1, requires_grad=True)
                x_vals = torch.stack(list(x_dict.values()))
                x_vals, dummy = torch.utils.checkpoint.checkpoint(
                    forward_wrapper, conv, x_vals, x_0, adj_t_dict, dummy
                )
                x_dict["0"] = x_vals[0]
            else:
                x_dict = self.execute_block(conv, x_dict, x_0, adj_t_dict)

        # TODO: apply only to record nodes
        if self.post is not None:
            x_dict = self.gradient_checkpoint_wrapper(self.post, x_dict)

        if self.out_norm is not None:
            x_dict = self.gradient_checkpoint_wrapper(torch.nn.functional.normalize, x_dict)

        return x_dict["0"]
