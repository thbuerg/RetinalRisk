from typing import Dict

import torch
import torch_geometric

from retinalrisk.data.collate import Batch
from retinalrisk.layers.augmentations import HeteroEdgeRemoving, HeteroFeatureMasking
from retinalrisk.loss.vicreg import vicreg_loss_func
from retinalrisk.models.supervised import LossWrapper, RecordNodeFeaturesMixin, RecordsTraining


class VICRegLoss(LossWrapper):
    def __init__(self, scale: float = 1.0, cov_batch_size: int = 2048):
        super().__init__("VICReg", scale)

        self.loss_fn = vicreg_loss_func
        self.cov_batch_size = cov_batch_size

    def compute(self, batch: Batch, outputs: Dict) -> Dict:
        z1 = outputs["full_node_embeddings"]
        z2 = outputs["full_node_embeddings_hat"]

        with torch.autocast(dtype=torch.float32, device_type="cuda"):
            loss = self.loss_fn(z1, z2, cov_batch_size=self.cov_batch_size)

        return loss * self.scale, loss


class VICRegRecordsTraining(RecordsTraining):
    def __init__(self, vicreg_loss_scale, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.losses.append(VICRegLoss(scale=vicreg_loss_scale))

        self.augmentations = [HeteroEdgeRemoving(pe=0.5), HeteroFeatureMasking(pf=0.1)]

    @staticmethod
    def augment(augmenters, data: torch_geometric.data.HeteroData):
        for aug in augmenters:
            data = aug.augment(data)
        return data

    def forward(self, batch: Batch) -> Dict:
        graph_aug1 = self.augment(self.augmentations, batch.graph)
        graph_aug2 = self.augment(self.augmentations, batch.graph)

        record_node_embeddings, full_node_embeddings = self.get_node_embeddings(batch, graph_aug1)

        with torch.no_grad():
            _, full_node_embeddings_hat = self.get_node_embeddings(batch, graph_aug2)

        individual_features, future_features = self.get_individual_features(
            batch, record_node_embeddings
        )
        individual_features = self.maybe_concat_covariates(batch, individual_features)

        head_outputs = self.head(individual_features)
        head_projection = self.get_projection(head_outputs)

        return dict(
            record_node_embeddings=record_node_embeddings,
            individual_features=individual_features,
            future_features=future_features,
            head_outputs=head_outputs,
            head_projection=head_projection,
            full_node_embeddings=full_node_embeddings,
            full_node_embeddings_hat=full_node_embeddings_hat,
        )


class AugmentedGraphEncoderMixin:
    def get_node_embeddings(self, batch: Batch, augmented_graph: torch_geometric.data.HeteroData):
        full_node_embeddings = self.graph_encoder(
            augmented_graph.x_dict, augmented_graph.adj_t_dict
        )
        record_node_embeddings = full_node_embeddings[batch.record_indices]

        if self.normalize_node_embeddings:
            record_node_embeddings = torch.nn.functional.normalize(record_node_embeddings)

        return record_node_embeddings, full_node_embeddings


class VICRegRecordsGraphTraining(
    AugmentedGraphEncoderMixin, RecordNodeFeaturesMixin, VICRegRecordsTraining
):
    pass
