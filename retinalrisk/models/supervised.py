import copy
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torchmetrics
import wandb
from deepspeed.ops.adam import FusedAdam
from pytorch_lightning import LightningModule

from retinalrisk.data.collate import Batch
from retinalrisk.models.loss_wrapper import LossWrapper
from retinalrisk.modules.head import AlphaHead
from retinalrisk.schedulers.alpha_schedulers import AlphaScheduler


class SupervisedTraining(LightningModule):
    def __init__(
        self,
        encoder: torch.nn.Module,
        head: torch.nn.Module,
        losses: Iterable[LossWrapper],
        label_mapping: Dict,
        incidence_mapping: Dict,
        # projector: Optional[torch.nn.Module],
        optimizer: Optional[torch.optim.Optimizer] = FusedAdam,
        optimizer_kwargs: Optional[Dict] = {"weight_decay": 5e-4},
        metrics_list: Optional[List[torchmetrics.Metric]] = [torchmetrics.AUROC],
        metrics_kwargs: Optional[List[Dict]] = None,
        exclusions_on_metrics: Optional[bool] = True,
        alpha_scheduler: Optional[AlphaScheduler] = None,
        node_dropout: Optional[float] = None,
        binarize_records: bool = False,
        gradient_checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["encoder", "head", "projector"])

        self.encoder = encoder
        self.head = head
        # self.projector = projector

        self.losses = losses

        self.label_mapping = label_mapping
        self.incidence_mapping = incidence_mapping

        if isinstance(self.head, AlphaHead):
            alpha = alpha_scheduler.get_alpha()
            self.head.update_alpha(alpha)

        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs

        self.valid_metrics = self.initialize_metrics(label_mapping, metrics_list, metrics_kwargs)
        self.train_metrics = copy.deepcopy(self.valid_metrics)
        self.max_mean_metrics = defaultdict(float)
        self.exclusions_on_metrics = exclusions_on_metrics

        self.executor = ThreadPoolExecutor(max_workers=32)
        self.alpha_scheduler = alpha_scheduler
        self.gradient_checkpointing = gradient_checkpointing

        self.node_dropout = node_dropout
        self.binarize_records = binarize_records

    def initialize_metrics(
        self,
        label_mapping: Dict,
        metrics_list: List[torchmetrics.Metric],
        metrics_kwargs: List[Dict],
    ):
        """
        Soft-wrapper for metrics instatiation. When loading from cpt these are already instatiated,
        throwing a typeerror when called -> soft wrap it!
        :return:
        """
        if metrics_kwargs is None:
            metrics_kwargs = [{} for m in metrics_list]

        metrics = torch.nn.ModuleDict()
        for l in label_mapping.keys():
            metrics[l] = torch.nn.ModuleList(
                [m(**kwargs) for m, kwargs in zip(metrics_list, metrics_kwargs)]
            )
        return metrics

    def update_metrics(self, metrics: Iterable, batch: Batch, outputs: Dict, loss_dict: Dict):
        """
        Calculate the validation metrics. Stepwise!
        :return:
        """
        times = batch.times.clone()
        no_event_idxs = times == 0
        times[no_event_idxs] = batch.censorings.repeat(1, times.shape[1])[no_event_idxs]

        for idx, (_, m_list) in enumerate(metrics.items()):
            p_i = outputs["head_outputs"]["logits"][:, idx].detach().float()
            l_i = batch.events[:, idx]
            t_i = times[:, idx]

            if self.exclusions_on_metrics:
                mask = batch.exclusions[:, idx] == 0
                p_i = p_i[mask]
                l_i = l_i[mask]
                t_i = t_i[mask]

            for m in m_list:
                if m.__class__.__name__ == "CIndex":
                    m.update(p_i, l_i.long(), t_i)
                else:
                    m.update(p_i, l_i.long())

    def compute_and_log_metrics(self, metrics: Dict, kind: Optional[str] = "train"):
        if kind != "train":
            averages = defaultdict(list)
            incidence_averages = defaultdict(list)

            def compute(args):
                l, m_list = args
                rs = []

                for m in m_list:
                    r = m.compute()
                    rs.append(r)

                return rs

            results = list(self.executor.map(compute, metrics.items()))

            for (l, m_list), rs in zip(metrics.items(), results):
                for m, r in zip(m_list, rs):
                    self.log(f"{kind}/{self.label_mapping[l]}_{m.__class__.__name__}", r)
                    averages[m.__class__.__name__].append(r)
                    incidence_averages[self.incidence_mapping[l]].append(r)
                    m.reset()

            for m, v in averages.items():
                v = torch.stack(v)
                # TODO: filter out nans, e.g. endpoints with no positive samples
                # probably should be done in a better way, otherwise CV results might not
                # be valid.
                metric_name = f"{kind}/mean_{m}"
                value = torch.nanmean(v)
                self.log(metric_name, value)

                self.logger.experiment.log(
                    {
                        f"{kind}/auroc_hist": wandb.Histogram(
                            sequence=v[torch.isfinite(v)].cpu(), num_bins=100
                        )
                    }
                )

                if value > self.max_mean_metrics[metric_name]:
                    self.max_mean_metrics[metric_name] = value

                self.log(f"{metric_name}_max", self.max_mean_metrics[metric_name])

            for m, v in incidence_averages.items():
                v = torch.stack(v)
                metric_name = f"{kind}/mean_{m}"
                value = torch.nanmean(v)
                self.log(metric_name, value)

        else:
            for l, m_list in metrics.items():
                for m in m_list:
                    m.reset()

    def shared_step(self, batch: Batch) -> Dict:
        outputs = self(batch)
        loss_dict = self.loss(batch, outputs)

        return outputs, loss_dict

    def training_step(self, batch: Batch, batch_idx: int):
        outputs, loss_dict = self.shared_step(batch)
        self.update_metrics(self.train_metrics, batch, outputs, loss_dict)
        if self.alpha_scheduler:
            self.alpha_scheduler.step()
        if isinstance(self.head, AlphaHead):
            alpha = self.alpha_scheduler.get_alpha()
            self.log("head_alpha", alpha, on_step=True)
            self.head.update_alpha(alpha)

        for k, v in loss_dict.items():
            self.log(
                f"train/{k}", v, prog_bar=True, batch_size=batch.events.shape[0], sync_dist=True
            )
        return loss_dict

    def training_epoch_end(self, outputs) -> None:
        self.compute_and_log_metrics(self.train_metrics, kind="train")

    def validation_step(self, batch: Tuple, batch_idx: int):
        outputs, loss_dict = self.shared_step(batch)
        self.update_metrics(self.valid_metrics, batch, outputs, loss_dict)

        for k, v in loss_dict.items():
            self.log(
                f"valid/{k}", v, prog_bar=True, batch_size=batch.events.shape[0], sync_dist=True
            )
        return {f"val_{k}": v for k, v in loss_dict.items()}

    def validation_epoch_end(self, outputs) -> None:
        self.compute_and_log_metrics(self.valid_metrics, kind="valid")

    def maybe_concat_covariates(self, batch: Batch, features: torch.Tensor):
        if features is None:
            assert batch.covariates is not None
            return batch.covariates

        if batch.covariates is not None:
            features = torch.cat((features, batch.covariates), axis=1)

        return features

    def forward(self, batch: Batch) -> Dict:
        embeddings = self.get_data_embeddings(batch)

        individual_features = self.maybe_concat_covariates(batch, embeddings)

        head_outputs = self.head(individual_features)

        return dict(
            latents=embeddings,
            individual_features=individual_features,
            head_outputs=head_outputs,
        )

    def loss(self, batch: Batch, outputs: Dict) -> Dict:
        loss_dict = {}

        losses = []
        for loss_wrapper in self.losses:
            loss, loss_unscaled = loss_wrapper.compute(batch, outputs)
            losses.append(loss)

            loss_dict[loss_wrapper.name] = loss_unscaled.detach()
            loss_dict[f"{loss_wrapper.name}_scaled"] = loss.detach()

        loss_dict["loss"] = torch.sum(torch.stack(losses))

        return loss_dict

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = self.optimizer(
            filter(lambda p: p.requires_grad, self.parameters()),
            **self.optimizer_kwargs)
        return optimizer


class ImageEncoderMixin:
    def get_data_embeddings(self, batch: Batch):

        if self.gradient_checkpointing:
            if self.encoder.__class__.__name__ in ['ConvNeXt'] :
                # THIS ONLY WORKS FOR CONVNEXT!!
                x = torch.utils.checkpoint.checkpoint_sequential(self.encoder.features, 5, batch.data)
                x = self.encoder.avgpool(x)
                embeddings = self.encoder.classifier(x)
            elif self.encoder.__class__.__name__ in ['EfficientNet'] :
                # EFFICIENTNET has flatten operation between avgpool and classifier
                x = torch.utils.checkpoint.checkpoint_sequential(self.encoder.features, 9, batch.data)
                x = self.encoder.avgpool(x)
                x = torch.flatten(x, 1)
                embeddings = self.encoder.classifier(x)
        else:
            embeddings = self.encoder(batch.data)

        return embeddings

class ImagePerceiverMixin:
    def get_data_embeddings(self, batch: Batch):
        permuted_data = torch.permute(batch.data, (0, 2, 3, 1)) # (B, C, W, H) -> (B, W, H, C)

        embeddings = self.encoder(permuted_data)

        return embeddings


class ImageTraining(ImageEncoderMixin, SupervisedTraining):
    pass

class ImagePerceiverTraining(ImagePerceiverMixin, SupervisedTraining):
    pass

class CovariatesOnlyMixin:
    def get_data_embeddings(self, batch: Batch):
        return None


class CovariatesOnlyTraining(CovariatesOnlyMixin, SupervisedTraining):
    pass
