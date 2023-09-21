import numpy as np
import torch
from torch import nn

GEGLU = None  # recover activations should this be needed!


class IdentityHead(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.identity = torch.nn.Identity()

    def forward(self, x):
        x = self.identity(x)

        return dict(pre_logits=None, logits=x)


class LinearHead(torch.nn.Module):
    def __init__(
        self, num_features, num_endpoints, incidence=None, dropout=0.2, gradient_checkpointing=False
    ):
        super().__init__()

        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(num_features, num_endpoints)

        if incidence is not None:
            self.linear.bias.data = torch.logit(torch.from_numpy(incidence.astype(np.float32)))

    def forward(self, x):
        x = self.linear(self.dropout(x))

        return dict(pre_logits=None, logits=x)


def init_weights(module, initializer_range=0.02):
    """
    Initializes the weights of the given module with a normal distribution
    with mean 0 and standard deviation `initializer_range`.
    """
    if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
            module.bias.data.zero_()

            # Normalize output weights for Linear layers
            if module.bias.dim() > 0:
                num_output_features = module.bias.size(0)
                module.weight.data /= torch.sqrt(
                    torch.tensor(num_output_features, dtype=torch.float)
                )
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class MLPHead(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_endpoints,
        num_hidden,
        num_layers,
        incidence=None,
        dropout=0.2,
        nonlin=torch.nn.LeakyReLU,
        norm=torch.nn.LayerNorm,
        detach_clf=False,
        gradient_checkpointing=True,
        initial_dropout=0.0,
    ):
        super().__init__()

        assert num_layers >= 0

        layers = []
        if initial_dropout > 0:
            layers.append(torch.nn.Dropout(initial_dropout))
        if norm is not None:
            layers.append(norm(num_features))
        layers.append(torch.nn.Linear(num_features, num_hidden))
        layers.append(nonlin())
        if norm is not None:
            layers.append(norm(num_hidden))

        for _ in range(num_layers):
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Linear(num_hidden, num_hidden))
            layers.append(nonlin())
            if norm is not None:
                layers.append(norm(num_hidden))

        self.out = torch.nn.Linear(num_hidden, num_endpoints, bias=False)

        if incidence is not None:
            self.out.bias.data = torch.logit(torch.from_numpy(incidence.astype(np.float32)))

        self.layers = torch.nn.Sequential(*layers)
        self.detach_clf = detach_clf

        self.gradient_checkpointing = gradient_checkpointing

        self.apply(init_weights)

    def forward(self, x):
        if self.gradient_checkpointing:
            rep = torch.utils.checkpoint.checkpoint(self.layers, x)
        else:
            rep = self.layers(x)

        clf_input = rep
        if self.detach_clf:
            clf_input = clf_input.detach()

        logits = self.out(clf_input)

        return dict(pre_logits=rep, logits=logits)


class ResBlock(torch.nn.Module):
    def __init__(self, num_features, nonlin=GEGLU, norm=torch.nn.LayerNorm, dropout=0.2):
        super().__init__()

        self.norm1 = norm(num_features)
        self.linear1 = torch.nn.utils.weight_norm(torch.nn.Linear(num_features, num_features * 2))
        self.act1 = nonlin()

        self.norm2 = norm(num_features)
        self.linear2 = torch.nn.utils.weight_norm(torch.nn.Linear(num_features, num_features * 2))
        self.act2 = nonlin()

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        identity = x

        x = self.norm1(x)
        x = self.linear1(x)
        x = self.act1(x)

        x = self.norm2(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.act2(x)

        return x + identity


class ResMLPHead(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_endpoints,
        num_hidden,
        num_blocks,
        incidence,
        dropout=0.2,
        nonlin=GEGLU,
        norm=torch.nn.LayerNorm,
        detach_clf=False,
        checkpoint=True,
    ):
        super().__init__()

        layers = []
        layers.append(torch.nn.Dropout(dropout))
        if num_features != num_hidden:
            layers.append(
                torch.nn.utils.weight_norm(torch.nn.Linear(num_features, num_hidden, bias=False))
            )

        layers += [
            ResBlock(num_hidden, nonlin=nonlin, norm=norm, dropout=dropout)
            for _ in range(num_blocks)
        ]
        self.layers = torch.nn.Sequential(*layers)

        self.out = torch.nn.Linear(num_hidden, num_endpoints)
        self.out.bias.data = torch.logit(torch.from_numpy(incidence.astype(np.float32)))

        self.detach_clf = detach_clf
        self.checkpoint = checkpoint

    def forward(self, x):
        if self.checkpoint:
            rep = torch.utils.checkpoint.checkpoint(self.layers, x)
        else:
            rep = self.layers(x)

        clf_input = rep
        if self.detach_clf:
            clf_input = clf_input.detach()

        logits = self.out(clf_input)

        return dict(pre_logits=rep, logits=logits)


class IndependentMLPHeads(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_endpoints,
        num_hidden,
        num_layers,
        incidence,
        dropout=0.2,
        nonlin=torch.nn.LeakyReLU,
        norm=torch.nn.LayerNorm,
        detach_clf=False,
    ):
        super().__init__()

        self.linear = torch.nn.Linear(num_features, 259, bias=False)
        self.ortho = torch.randn(259, num_features, requires_grad=False)
        torch.nn.init.orthogonal_(self.ortho, gain=1)

        self.heads = torch.nn.ModuleList(
            [
                MLPHead(
                    259,
                    1,
                    num_hidden,
                    num_layers,
                    incidence[:, i : i + 1],
                    dropout,
                    nonlin,
                    norm,
                    detach_clf,
                )
                for i in range(num_endpoints)
            ]
        )

    def forward(self, x):
        pre_logits = []
        logits = []

        with torch.no_grad():
            self.linear.weight.copy_(self.ortho)
        x = self.linear(x)

        for head in self.heads:
            x = head(x)

        pre_logits = None
        logits = torch.cat(logits, dim=1)

        return dict(pre_logits=pre_logits, logits=logits)


class AlphaHead(torch.nn.Module):
    """Wrapper head designed to successively shift weight from one head to another by means of a weight alpha."""

    def __init__(self, head1: torch.nn.Module, head2: torch.nn.Module, alpha: float = 1.0):
        super().__init__()
        self.head1 = head1
        self.head2 = head2
        self.alpha = torch.tensor([alpha], requires_grad=False)

    def update_alpha(self, alpha: float):
        self.alpha = alpha

    def forward(self, x):
        outputs = {}
        x_head1 = self.head1(x)
        x_head2 = self.head2(x)

        for key in x_head1.keys():
            if x_head1[key] is not None and x_head2[key] is not None:
                outputs[key] = self.alpha * x_head1[key] + (1.0 - self.alpha) * x_head2[key]

        return outputs
