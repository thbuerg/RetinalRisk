import torch.nn as nn

from retinalrisk.data.collate import Batch


class ShapWrapper(nn.Module):
    def __init__(self, head_model, record_node_embeddings, n_records, device, lightning_module):
        """
        Wrapps a Module of instance RecordsTraining and fixes the embedding to be able to attribute to the records.
        """
        super().__init__()
        self.head_model = head_model
        self.head_model.gradient_checkpointing = False
        self.record_node_embeddings = nn.Parameter(record_node_embeddings)
        self.record_node_embeddings.requires_grad = False
        self.n_records = n_records
        self.device = device
        self.data = None
        self.lightning_module = lightning_module

    def forward(self, batch_data):
        covariates=None
        records = batch_data
        if batch_data.shape[1] > self.n_records:
            # then we have covariates:
            records = batch_data[:, : self.n_records]
            covariates = batch_data[:, self.n_records :]

        batch = Batch(
            graph=None,
            record_indices=None,
            records=records,
            covariates=covariates,
            exclusions=None,
            events=None,
            times=None,
            censorings=None,
            eids=None
        )

        individual_features, future_features = self.lightning_module.get_individual_features(
            batch, self.record_node_embeddings
        )
        individual_features = self.lightning_module.maybe_concat_covariates(
            batch, individual_features
        )

        head_outputs = self.head_model(individual_features)
        logits = head_outputs["logits"]

        return logits
