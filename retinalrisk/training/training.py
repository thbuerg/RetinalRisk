import math
import warnings
from socket import gethostname

import hydra
import numpy as np
import torch
import torchvision as tv
import torchmetrics
from omegaconf import DictConfig
from vit_pytorch.efficient import ViT
from vit_pytorch import SimpleViT
from vit_pytorch.cct import CCT
from perceiver_pytorch import Perceiver
from nystrom_attention import Nystromformer

from retinalrisk.models.retfound import vit_large_patch16, interpolate_pos_embed
from timm.models.layers import trunc_normal_
from retinalrisk.data.data import WandBBaselineData, get_or_load_wandbdataobj
from retinalrisk.data.datamodules import RetinaDataModule
from retinalrisk.loss.focal import FocalBCEWithLogitsLoss
from retinalrisk.loss.tte import CIndex, CoxPHLoss
from retinalrisk.models.loss_wrapper import (
    EndpointClassificationLoss,
    EndpointTTELoss,
)
from retinalrisk.models.supervised import (
    CovariatesOnlyTraining,
    ImageTraining,
    ImagePerceiverTraining,
)
from retinalrisk.modules.head import (
    AlphaHead,
    IndependentMLPHeads,
    LinearHead,
    MLPHead,
    ResMLPHead,
    IdentityHead,
)

from retinalrisk.models.retfound import get_retfound_transforms


def setup_training(args: DictConfig):
    torch.set_float32_matmul_precision("medium")

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
        elif head_config.model_type == "Identity":
            return IdentityHead()
        else:
            assert False

        num_endpoints = len(datamodule.labels)

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

    if len(args.datamodule.covariates):
        num_covariates = len(args.datamodule.covariates) + 1
    else:
        num_covariates = 0

    incidence = datamodule.train_dataset.labels_events.mean(axis=0)
    incidence_mapping = {}
    fix_str = lambda s: s.replace(".", "-").replace("/", "-")

    for l, i in zip(datamodule.labels, incidence.values):
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

    training_kwargs = dict(
        label_mapping=datamodule.label_mapping,
        incidence_mapping=incidence_mapping,
        exclusions_on_metrics=args.training.exclusions_on_metrics,
        losses=losses,
        metrics_list=metrics,
        alpha_scheduler=alpha_scheduler,
        optimizer_kwargs=args.training.optimizer_kwargs,
        binarize_records=args.training.binarize_records,
        gradient_checkpointing=args.training.gradient_checkpointing,
        # sync_dist=args.training.sync_dist
    )

    if args.model.model_type == "image":
        tags.append("image")

        if "convnext" in args.model.encoder:
            try:
                # encoder = tv.models.__dict__[args.model.encoder](pretrained=args.model.pretrained)
                weights = "DEFAULT" if args.model.pretrained else None
                encoder = tv.models.__dict__[args.model.encoder](weights=weights)

                outshape = (
                    768
                    if any(["small" in args.model.encoder, "tiny" in args.model.encoder])
                    else 1024
                )

                setattr(encoder.classifier, "2", torch.nn.Identity())
            except KeyError:
                print(f"No model named `{args.model.encoder}`.")
                raise KeyError("Please check available torchvision models.")

        elif "efficientnet" in args.model.encoder:
            # encoder = tv.models.__dict__[args.model.encoder](pretrained=args.model.pretrained)
            weights = "DEFAULT" if args.model.pretrained else None
            encoder = tv.models.__dict__[args.model.encoder](weights=weights)

            outshape = encoder.classifier[1].weight.shape[1]
            encoder.classifier = torch.nn.Identity()

        elif "retfound" in args.model.encoder:
            checkpoint_path = args.model.checkpoint_path

            image_size = args.datamodule.augmentation.train.CenterCrop.size

            model = vit_large_patch16(
                num_classes=2,
                drop_path_rate=args.model.drop_path_rate,
                global_pool=True,
                img_size=image_size,
            )

            if args.model.pretrained:
                # load RETFound weights
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                checkpoint_model = checkpoint["model"]
                state_dict = model.state_dict()
                for k in ["head.weight", "head.bias"]:
                    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint_model[k]

                # interpolate position embedding
                interpolate_pos_embed(model, checkpoint_model)

                # load pre-trained model
                msg = model.load_state_dict(checkpoint_model, strict=False)

                assert set(msg.missing_keys) == {
                    "head.weight",
                    "head.bias",
                    "fc_norm.weight",
                    "fc_norm.bias",
                }

            # manually initialize fc layer
            trunc_normal_(model.head.weight, std=2e-5)

            encoder = model
            encoder.head = torch.nn.Identity()

            outshape = 1024

            if args.model.retfound_augment:
                transform, valid_transform = get_retfound_transforms()
                datamodule.train_dataset.transforms = transform
                datamodule.valid_dataset.transforms = valid_transform

                print(datamodule.train_dataset.transforms)
                print(datamodule.valid_dataset.transforms)

        elif "resnet" in args.model.encoder:
            try:
                encoder = tv.models.__dict__[args.model.encoder](pretrained=args.model.pretrained)
                outshape = encoder.fc.weight.shape[1]
                encoder.fc = torch.nn.Identity()
            except KeyError:
                print(f"No model named `{args.model.encoder}`.")
                raise KeyError("Please check available torchvision models.")

        elif "perceiver" in args.model.encoder:
            print("using perceiver with following kwargs:\n", args.model.perceiver)
            encoder = Perceiver(num_classes=len(datamodule.labels), **args.model.perceiver)
            outshape = len(datamodule.labels)

        elif "simple_vit" in args.model.encoder:
            tags.append("simple_ViT")
            encoder = SimpleViT(
                image_size=args.model.encoder_image_size,
                patch_size=args.model.encoder_patch_size,
                num_classes=args.model.encoder_num_classes,
                dim=1024,
                depth=6,
                heads=16,
                mlp_dim=2048,
            )
            outshape = 1024
            setattr(encoder.linear_head, "1", torch.nn.Identity())

        elif "efficient_vit" in args.model.encoder:
            tags.append("efficient_ViT")
            efficient_transformer = Nystromformer(dim=512, depth=12, heads=8, num_landmarks=256)
            encoder = ViT(
                dim=512,
                transformer=efficient_transformer,
                image_size=args.model.encoder_image_size,
                patch_size=args.model.encoder_patch_size,
                num_classes=args.model.encoder_num_classes,
            )

            outshape = 512
            setattr(encoder.mlp_head, "1", torch.nn.Identity())

        elif "cct_vit" in args.model.encoder:
            tags.append("CCT_ViT")
            encoder = CCT(
                img_size=(args.model.encoder_image_size, args.model.encoder_image_size),
                embedding_dim=512,
                n_conv_layers=2,
                kernel_size=7,
                stride=2,
                padding=3,
                pooling_kernel_size=3,
                pooling_stride=2,
                pooling_padding=1,
                num_layers=14,
                num_heads=6,
                mlp_radio=3.0,
                num_classes=args.model.encoder_num_classes,
                positional_embedding="learnable",  # ['sine', 'learnable', 'none']
            )
            outshape = encoder.classifier.fc.weight.shape[1]
            encoder.classifier.fc = torch.nn.Identity()

        else:
            print("args.model.encoder", args.model.encoder)
            raise NotImplementedError()

        if args.model.freeze_encoder:
            assert isinstance(args.model.freeze_encoder, bool)
            print(args.model.freeze_encoder)
            for name, param in encoder.named_parameters():
                if param.requires_grad:
                    param.requires_grad = False

        # todo: make this more pretty
        if args.datamodule.covariates == ["age_at_recruitment_f21022_0_0", "sex_f31_0_0"]:
            head_outdim = outshape + len(args.datamodule.covariates) + 1
        else:
            head_outdim = outshape + len(args.datamodule.covariates)
        head = get_head(args.head, head_outdim)

        if "perceiver" in args.model.encoder:
            tags.append("perceiver")
            model = ImagePerceiverTraining(encoder=encoder, head=head, **training_kwargs)
        else:
            model = ImageTraining(encoder=encoder, head=head, **training_kwargs)

    elif args.model.model_type == "covariates":
        tags.append("covariates_baseline")

        num_head_features = num_covariates
        head = get_head(args.head, num_head_features)

        model = CovariatesOnlyTraining(encoder=None, head=head, **training_kwargs)

    elif args.model.model_type == "transformer":
        tags.append("transformer")

        raise NotImplementedError()

    else:
        assert False

    return datamodule, model, tags
