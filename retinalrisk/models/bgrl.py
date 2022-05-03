import copy
from typing import Dict, List, Optional, Tuple

import GCL
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
import torchmetrics
import wandb
from pytorch_lightning import LightningModule

from retinalrisk.models.supervised import AbstractEHRGraphTraining
from retinalrisk.modules.gnn import Identity


class BGRLModelWrapper(torch.nn.Module):
    def __init__(self, graph_encoder, hidden_dim, dropout=0.2):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.batch_norm = torch.nn.BatchNorm1d(hidden_dim)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout),
        )

    def forward(self, data):
        z = self.graph_encoder(data)
        z = self.batch_norm(z)

        return z, self.projection_head(z)


class BGRLEncoder(torch.nn.Module):
    def __init__(self, encoder: BGRLModelWrapper, hidden_dim, dropout=0.2):
        super().__init__()
        self.online_encoder = encoder
        self.target_encoder = None
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout),
        )

        A = GCL.augmentors
        self.aug1 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])
        self.aug2 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])

    def get_target_encoder(self):
        if self.target_encoder is None:
            self.target_encoder = copy.deepcopy(self.online_encoder)

            for p in self.target_encoder.parameters():
                p.requires_grad = False
        return self.target_encoder

    def update_target_encoder(self, momentum: float):
        for p, new_p in zip(
            self.get_target_encoder().parameters(), self.online_encoder.parameters()
        ):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p

    @staticmethod
    def augment(augmenter, data):
        edge_index = data.edge_index
        edge_code = data.edge_code
        edge_weight = data.edge_weight
        edge_attr_combined = torch.cat((edge_code.unsqueeze(-1), edge_weight.unsqueeze(-1)), dim=1)
        x = data.x

        x_aug, edge_index_aug, edge_attr_aug = augmenter(data.x, edge_index, edge_attr_combined)
        edge_code_aug = edge_attr_aug[:, 0].long()
        edge_weight_aug = edge_attr_aug[:, 1]

        data_aug = torch_geometric.data.Data(
            x=x_aug, edge_index=edge_index_aug, edge_code=edge_code_aug, edge_weight=edge_weight_aug
        )
        return data_aug

    def forward(self, data):
        data1 = self.augment(self.aug1, data)
        data2 = self.augment(self.aug2, data)

        h1, h1_online = self.online_encoder(data1)
        h2, h2_online = self.online_encoder(data2)

        h1_pred = self.predictor(h1_online)
        h2_pred = self.predictor(h2_online)

        with torch.no_grad():
            _, h1_target = self.get_target_encoder()(data1)
            _, h2_target = self.get_target_encoder()(data2)

        return h1, h2, h1_pred, h2_pred, h1_target, h2_target


class BGRLSemiSupervisedTraining(AbstractEHRGraphTraining):
    def __init__(
        self,
        graph_encoder: torch.nn.Module,
        head: torch.nn.Module,
        label_mapping: Dict,
        num_outputs: int,
        optimizer: Optional[torch.optim.Optimizer] = torch.optim.AdamW,
        optimizer_kwargs: Optional[Dict] = {"weight_decay": 5e-4},
        metrics_list: Optional[List[torchmetrics.Metric]] = [torchmetrics.AUROC],
        metrics_kwargs: Optional[List[Dict]] = None,
        exclusions_on: Optional[Dict] = {"loss": True, "metrics": True},
        loss: torch.nn.modules.loss._Loss = torch.nn.modules.loss.BCEWithLogitsLoss,
        loss_kwargs: Dict = {},
    ):
        gnn_wrapper = BGRLModelWrapper(graph_encoder, num_outputs)
        graph_encoder = BGRLEncoder(gnn_wrapper, num_outputs)

        super().__init__(
            graph_encoder,
            head,
            label_mapping,
            optimizer,
            optimizer_kwargs,
            metrics_list,
            metrics_kwargs,
            exclusions_on,
        )

        self.gnn_wrapper = gnn_wrapper
        self.contrast_model = GCL.models.BootstrapContrast(
            loss=GCL.losses.BootstrapLatent(), mode="L2L"
        )

        self.loss_fn = loss(**loss_kwargs)

        self.automatic_optimization = False

    def unpack_batch(self, batch: Tuple):
        data, (records, covariates, exclusions), (events, _) = batch
        return (data, records, covariates), [events], exclusions

    def maybe_concat_covariates(self, features: torch.Tensor, covariates: Optional[torch.Tensor]):
        if covariates is not None:
            features = torch.cat((features, covariates), axis=1)

        return features

    def forward(
        self,
        data: torch_geometric.data.Data,
        records: torch.Tensor,
        covariates: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        full_node_embeddings, _, h1_pred, h2_pred, h1_target, h2_target = self.graph_encoder(data)

        sampled_record_indices = np.arange(records.shape[1])
        record_node_embeddings = full_node_embeddings[sampled_record_indices]

        fts = records @ record_node_embeddings
        fts = self.maybe_concat_covariates(fts, covariates)

        logits = self.head(fts)

        return logits, h1_pred, h2_pred, h1_target, h2_target

    def bgrl_loss(self, h1_pred, h2_pred, h1_target, h2_target):
        loss = self.contrast_model(
            h1_pred=h1_pred,
            h2_pred=h2_pred,
            h1_target=h1_target.detach(),
            h2_target=h2_target.detach(),
        )

        return loss

    def records_loss(self, predictions, labels, exclusion_mask) -> Dict:
        loss = self.loss_fn(predictions, labels)

        if self.exclusions_on["loss"]:
            # TODO: hack
            exclusion_mask = exclusion_mask.bool().float()
            inclusion_mask = 1 - exclusion_mask
            loss = (loss * inclusion_mask).sum(dim=0) / inclusion_mask.sum(dim=0)

        loss = loss[torch.isfinite(loss)].mean()

        return loss

    def shared_step(self, batch: Tuple) -> Dict:
        data_tuple, labels_tuple, exclusion_mask = self.unpack_batch(batch)

        outputs = self(*data_tuple)
        records_loss = self.records_loss(outputs[0], labels_tuple[0], exclusion_mask)
        bgrl_loss = self.bgrl_loss(*outputs[1:])

        loss_dict = dict(loss=records_loss, bgrl_loss=bgrl_loss)

        return loss_dict, outputs[0], labels_tuple, exclusion_mask

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        loss_dict, predictions, labels_tuple, exclusion_mask = self.shared_step(batch)

        # TODO: fix manual loss scaling
        loss = loss_dict["loss"] + loss_dict["bgrl_loss"] / 1000
        self.manual_backward(loss)

        opt.step()

        self.graph_encoder.update_target_encoder(0.99)

        self.update_metrics(self.train_metrics, predictions, labels_tuple, exclusion_mask)

        for k, v in loss_dict.items():
            self.log(
                f"train/{k}",
                v,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                batch_size=labels_tuple[0].shape[0],
                sync_dist=True,
            )
