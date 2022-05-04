import math
import warnings
from socket import gethostname

import hydra
import numpy as np
import torch
import torchmetrics
from omegaconf import DictConfig

from retinalrisk.data.data import WandBBaselineData, get_or_load_wandbdataobj
from retinalrisk.data.datamodules import RetinaDataModule
from retinalrisk.loss.focal import FocalBCEWithLogitsLoss
from retinalrisk.loss.tte import CIndex, CoxPHLoss
from retinalrisk.models.loss_wrapper import (
    EndpointClassificationLoss,
    EndpointContrastiveLoss,
    EndpointTTELoss,
)
from retinalrisk.models.supervised import (
    CovariatesOnlyTraining,
    RecordsGraphTraining,
    RecordsIdentityTraining,
    RecordsLearnedEmbeddingsTraining,
    RecordsPretrainedEmbeddingsTraining,
    RecordsShuffledGraphTraining,
)
from retinalrisk.models.vicreg import VICRegRecordsGraphTraining
from retinalrisk.modules.gnn import HeteroGNN
from retinalrisk.modules.head import AlphaHead, IndependentMLPHeads, LinearHead, MLPHead, ResMLPHead


def setup_training(args: DictConfig):
    tags = list(args.setup.tags)

    host = gethostname()
    cluster = "charite-hpc" if host.startswith("s-sc") else "eils-hpc"
    data_root = f"{args.setup.root[cluster]}/{args.setup.data_path}"

    if args.setup.use_data_artifact_if_available:
        data = get_or_load_wandbdataobj(
            data_root,
            identifier=args.setup.data_identifier,
            entity=args.setup.entity,
            project=args.setup.project,
            **args.setup.data,
        )
    else:
        data = WandBBaselineData(
                    data_root=data_root,
                    wandb_entity="cardiors",
                    wandb_project="Retina",
                **args.setup.data,
            )

    datamodule = RetinaDataModule(
        data,
        **args.datamodule,
    )
    datamodule.prepare_data()

    def get_head(head_config, num_head_features):
        if args.datamodule.task == "binary":
            incidence = datamodule.train_dataset.labels_events.mean(
                axis=0
            )  # if this is not respecting potential exclusions, could it crash in the loss/metrics for rare events?
        else:
            incidence = None

        if head_config.model_type == "Linear":
            cls = LinearHead
        elif head_config.model_type == "MLP":
            cls = MLPHead
        elif head_config.model_type == "ResMLP":
            cls = ResMLPHead
        elif args.head.model_type == "MLP_independent":
            cls = IndependentMLPHeads
        elif head_config.model_type == "AlphaHead":
            head1 = get_head(head_config.head1, num_head_features)
            head2 = get_head(head_config.head2, num_head_features)
            return AlphaHead(head1, head2, alpha=head_config.alpha)
        else:
            assert False

        num_endpoints = len(datamodule.labels)
        if args.training.use_endpoint_embeddings:
            num_endpoints = datamodule.graph.num_features

        return cls(
            num_head_features,
            num_endpoints,
            incidence=incidence,
            dropout=head_config.dropout,
            gradient_checkpointing=args.training.gradient_checkpointing,
            **head_config.kwargs,
        )

    if "alpha_scheduler" in args.training and args.training:
        steps_per_epoch = math.ceil(len(datamodule.train_dataset) / args.datamodule.batch_size)
        max_epochs = args.trainer.max_epochs
        try:
            alpha_scheduler = hydra.utils.instantiate(
                args.training.alpha_scheduler,
                steps_per_epoch=steps_per_epoch,
                max_epochs=max_epochs,
            )
        except Exception:
            warnings.warn("Failed to instantiate AlphaScheduler, proceeding without it.")
            alpha_scheduler = None
    else:
        alpha_scheduler = None

    # TODO: fix
    if len(args.datamodule.covariates):
        num_covariates = len(args.datamodule.covariates) + 1
    else:
        num_covariates = 0

    incidence = datamodule.train_dataset.labels_events.mean(axis=0)
    incidence_mapping = {}
    fix_str = lambda s: s.replace(".", "-").replace("/", "-")
    for l, i in zip(datamodule.labels, np.array(incidence)[0]):
        l = fix_str(l)
        if i > 0.1:
            incidence_mapping[l] = ">1:10"
        elif (i <= 0.1) and (i > 0.01):
            incidence_mapping[l] = ">1:100"
        elif (i <= 0.01) and (i > 0.001):
            incidence_mapping[l] = ">1:1000"
        elif i <= 0.001:
            incidence_mapping[l] = "<1:1000"

    loss_weights = None
    if args.datamodule.use_loss_weights:
        loss_weights = 1 / np.log1p(1 / (incidence + 1e-8))

    losses = []
    metrics = []
    if args.datamodule.task == "binary":
        losses.append(
            EndpointClassificationLoss(
                FocalBCEWithLogitsLoss,
                {},
                scale=args.training.endpoint_loss_factor,
                use_exclusion_mask=args.training.exclusions_on_losses,
            )
        )
        metrics.append(torchmetrics.AUROC)
    elif args.datamodule.task == "tte":
        losses.append(
            EndpointTTELoss(
                CoxPHLoss,
                {},
                scale=args.training.endpoint_loss_factor,
                use_exclusion_mask=args.training.exclusions_on_losses,
                loss_weights=loss_weights,
            )
        )
        metrics.append(CIndex)

    projector = None
    if args.training.contrastive_loss_factor > 0:
        losses.append(EndpointContrastiveLoss(scale=args.training.contrastive_loss_factor))
        projector = torch.nn.Linear(args.head.kwargs["num_hidden"], args.model.num_outputs)

    training_kwargs = dict(
        label_mapping=datamodule.label_mapping,
        incidence_mapping=incidence_mapping,
        exclusions_on_metrics=args.training.exclusions_on_metrics,
        losses=losses,
        metrics_list=metrics,
        projector=projector,
        alpha_scheduler=alpha_scheduler,
        node_dropout=args.training.node_dropout,
        normalize_node_embeddings=args.training.normalize_node_embeddings,
        optimizer_kwargs=args.training.optimizer_kwargs,
        binarize_records=args.training.binarize_records,
        use_endpoint_embeddings=args.training.use_endpoint_embeddings,
    )

    # TODO  restrict choices of model types here!
    if args.model.model_type == "GNN":
        tags.append("gnn")

        gnn = HeteroGNN(
            datamodule.graph.num_features,
            args.model.num_hidden,
            args.model.num_outputs,
            args.model.num_blocks,
            metadata=datamodule.graph.metadata(),
            gradient_checkpointing=args.training.gradient_checkpointing,
            weight_norm=args.model.weight_norm,
            dropout=args.model.dropout,
        )

        num_head_features = gnn.num_outputs + num_covariates
        head = get_head(args.head, num_head_features)

        cls = RecordsShuffledGraphTraining if args.model.shuffled else RecordsGraphTraining

        model = cls(
            graph_encoder=gnn,
            head=head,
            **training_kwargs,
        )
    elif args.model.model_type == "VICReg":
        tags.append("gnn")
        tags.append("vicreg")

        gnn = HeteroGNN(
            datamodule.graph.num_features,
            args.model.num_hidden,
            args.model.num_outputs,
            args.model.num_blocks,
            metadata=datamodule.graph.metadata(),
            gradient_checkpointing=args.training.gradient_checkpointing,
            weight_norm=args.model.weight_norm,
            dropout=args.model.dropout,
        )

        num_head_features = gnn.num_outputs + num_covariates
        head = get_head(args.head, num_head_features)

        model = VICRegRecordsGraphTraining(
            vicreg_loss_scale=args.training.vicreg_loss_factor,
            graph_encoder=gnn,
            head=head,
            **training_kwargs,
        )

    # TODO: expand on the identity training
    elif args.model.model_type == "raw_image":
        tags.append("raw_image")


        latent_size = 10

        head = get_head(args.head, latent_size)

        model = RecordsIdentityTraining(graph_encoder=None, head=head, **training_kwargs)
    elif args.model.model_type == "Covariates":
        # TODO: keep the covariates only training
        tags.append("covariates_baseline")

        num_head_features = num_covariates
        head = get_head(args.head, num_head_features)

        model = CovariatesOnlyTraining(graph_encoder=None, head=head, **training_kwargs)
    else:
        assert False

    return datamodule, model, tags
