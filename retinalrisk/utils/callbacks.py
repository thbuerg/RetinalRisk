import os
import pickle
from pathlib import Path

import pandas as pd
import ray
import torch
import PIL
import zstandard
from torchvision import transforms
from timm.data.transforms import RandomResizedCropAndInterpolation
import torchvision.transforms.functional as TF
import PIL

# from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import BaseFinetuning

# from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from tqdm import tqdm

from retinalrisk.transforms.transforms import AdaptiveRandomCropTransform


def annotate_df(df, datamodule, module, split=None):
    df["partition"] = datamodule.partition
    if split:
        df["split"] = split
    df["module"] = type(module).__name__
    if module.encoder:
        df["encoder"] = module.encoder._get_name()
    else:
        df["encoder"] = "None"
    df["head"] = module.head._get_name()
    df["covariate_cols"] = str(datamodule.covariate_cols)
    # df["record_cols"] = str(datamodule.record_cols)

    for col in ["partition", "module", "encoder", "head", "covariate_cols"]:
        df[col] = df[col].astype("category")

    if split:
        df["split"] = df["split"].astype("category")

    return df


class WritePredictionsDataFrame(Callback):
    """
    Write Predictions.
    """

    def __init__(self, **kwargs):
        super().__init__()

    # def on_exception(self, trainer, module):
    #    self.on_fit_end(trainer, module)

    def manual(self, args, datamodule, module, testtime_crop_ratios=None):
        print("Write predictions and patient embeddings")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        ckpt = torch.load(args.model.restore_from_ckpt, map_location=device)
        # ckpt = torch.load(args.model.restore_from_ckpt)
        module.load_state_dict(ckpt["state_dict"])
        module.eval()

        module.to(device)

        # write the predictions, could be extended for all sets, just works if not shuffled
        endpoints = list(datamodule.label_mapping.keys())

        predictions_dfs = []

        if testtime_crop_ratios is None:
            testtime_crop_ratios = [None]

        testtime_crop_ratios = [None]

        # add the functionality of running multiple predictions w/ different TTA settings!
        for crop_ratio in testtime_crop_ratios:
            for split in tqdm(["train", "valid", "test"]):
                """
                # overwrite transforms in  train/test/valid according to the TTA settings!
                if crop_ratio is not None:
                    t = transforms.Compose(
                        [
                            AdaptiveRandomCropTransform(
                                crop_ratio=crop_ratio,
                                out_size=datamodule.img_size_to_gpu,
                                interpolation=PIL.Image.BICUBIC,
                            ),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                            ),
                        ]
                    )
                """

                t = transforms.Compose(
                    [
                        RandomResizedCropAndInterpolation(
                            size=(datamodule.img_size_to_gpu, datamodule.img_size_to_gpu),
                            scale=(0.08, 1.0),
                            ratio=(1, 1),
                            interpolation="bicubic",
                        ),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
                )

                print(t)

                if split == "train":
                    # if crop_ratio is not None:
                    datamodule.train_dataset.transforms = t
                    dataloader = datamodule.train_dataloader(
                        shuffle=False, drop_last=False, testtime=True
                    )
                if split == "valid":
                    # if crop_ratio is not None:
                    datamodule.valid_dataset.transforms = t
                    dataloader = datamodule.val_dataloader(testtime=True)
                if split == "test":
                    datamodule.test_dataset = datamodule.get_retina_dataset(set="test")
                    # if crop_ratio is not None:
                    datamodule.test_dataset.transforms = t
                    dataloader = datamodule.test_dataloader(testtime=True)

                outputs = self.predict_dataloader(module, dataloader, device)

                index = dataloader.dataset.retina_map["eid"].values
                if args.datamodule.img_n_testtime_views > 1:
                    # prepare tta index:
                    index = []
                    for i in dataloader.dataset.retina_map["eid"].values:
                        index.extend([i] * args.datamodule.img_n_testtime_views)

                predictions_df = pd.DataFrame(
                    data=outputs["preds"], index=index, columns=endpoints
                ).reset_index(drop=False)
                predictions_df = annotate_df(predictions_df, datamodule, module).assign(split=split)
                predictions_dfs.append(predictions_df)

            predictions_dfs_cc = pd.concat(predictions_dfs, axis=0).reset_index(drop=True)

            # write to disk
            outdir = os.path.join(Path(args.model.restore_from_ckpt).parent, "predictions")
            if not os.path.exists(outdir):
                os.mkdir(outdir)

            tag = f"_cropratio{str(crop_ratio)}" if crop_ratio is not None else ""
            predictions_dfs_cc.to_feather(os.path.join(outdir, f"predictions{tag}.feather"))
            print(f"Predictions saved {os.path.join(outdir, f'predictions{tag}.feather')}")

            del predictions_df
            del dataloader

    def on_fit_end(self, trainer, module):
        print("Write predictions and patient embeddings")

        device = torch.device("cuda")
        module.to(device)

        ckpt = torch.load(trainer.checkpoint_callback.best_model_path)
        module.load_state_dict(ckpt["state_dict"])
        module.eval()
        module.to(device)

        # write the predictions, could be extended for all sets, just works if not shuffled
        endpoints = list(trainer.datamodule.label_mapping.keys())

        predictions_dfs = []

        for split in tqdm(["train", "valid", "test"]):
            if split == "train":
                dataloader = trainer.datamodule.train_dataloader(
                    shuffle=False, drop_last=False, testtime=True
                )
            if split == "valid":
                dataloader = trainer.datamodule.val_dataloader(testtime=True)
            if split == "test":
                dataloader = trainer.datamodule.test_dataloader(testtime=True)

            outputs = self.predict_dataloader(module, dataloader, device)

            index = dataloader.dataset.retina_map["eid"].values
            if trainer.datamodule.img_n_testtime_views > 1:
                # prepare tta index:
                index = []
                for i in dataloader.dataset.retina_map["eid"].values:
                    index.extend([i] * trainer.datamodule.img_n_testtime_views)

            predictions_df = pd.DataFrame(
                data=outputs["preds"], index=index, columns=endpoints
            ).reset_index(drop=False)
            predictions_df = annotate_df(predictions_df, trainer.datamodule, module).assign(
                split=split
            )
            predictions_dfs.append(predictions_df)

        predictions_dfs_cc = pd.concat(predictions_dfs, axis=0).reset_index(drop=True)
        self.write_and_log_preds(trainer, predictions_dfs_cc)

    def predict_dataloader(self, model, dataloader, device):
        preds_list = []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch.data = batch.data.to(device)
                batch.covariates = batch.covariates.to(device)

                head_outputs = model(batch)["head_outputs"]
                del batch

                preds = head_outputs["logits"].detach().cpu()
                del head_outputs
                preds_list.append(preds)

        return {"preds": torch.cat(preds_list, axis=0).numpy()}

    def write_and_log_embs(self, trainer, embs_df):
        # write the predictions.csv
        outdir = os.path.join(Path(trainer.checkpoint_callback.dirpath).parent, "embeddings")
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        embs_df.to_feather(os.path.join(outdir, "patient_embeddings.feather"))
        print(f"Patient embeddings saved {os.path.join(outdir, 'patient_embeddings.feather')}")

    def write_and_log_preds(self, trainer, predictions_df, split=None):
        # write the predictions.csv
        outdir = os.path.join(Path(trainer.checkpoint_callback.dirpath).parent, "predictions")
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        predictions_df.to_feather(
            os.path.join(
                outdir,
                (
                    "predictions_augmented.feather"
                    if split is None
                    else f"{split}_predictions_augmented.feather"
                ),
            )
        )
        print(f"Predictions saved {os.path.join(outdir, 'predictions_augmented.feather')}")


@ray.remote
def save_tensor(fp, tensor):
    array = tensor.numpy()
    with open(fp, "wb") as fh:
        cctx = zstandard.ZstdCompressor()
        with cctx.stream_writer(fh) as compressor:
            compressor.write(pickle.dumps(array, protocol=pickle.HIGHEST_PROTOCOL))


class WriteFeatureAttributions(Callback):
    """
    Calculate Attributions to the binary record mask.
    """

    def __init__(self, batch_size, baseline_mode="zeros", **kwargs):
        self.batch_size = batch_size
        self.baseline_mode = baseline_mode
        super().__init__()

    # def on_exception(self, trainer, module):
    #   self.on_fit_end(trainer, module)

    def on_fit_end(self, trainer, module):
        # prepare ray for async pickling
        ray.init(num_cpus=10)

        # instantiate devices
        gpu = torch.device("cuda")
        cpu = torch.device("cpu")

        # prepare model with best checkpoint
        ckpt = torch.load(trainer.checkpoint_callback.best_model_path)
        module.load_state_dict(ckpt["state_dict"])
        module.eval()
        module.to(cpu)

        # prepare data
        _ = trainer.datamodule.train_dataloader()
        test_dataloader = trainer.datamodule.test_dataloader()
        record_node_embeddings = self.predict_batch(module, test_dataloader, cpu)
        test_data, baseline_data = self.prepare_data(trainer.datamodule, cpu)

        # prepare endpoints and metadata
        endpoints = sorted(list(trainer.datamodule.label_mapping.values()))

        records = trainer.datamodule.record_cols
        features = records + self.get_covariate_names(trainer.datamodule)

        eids = trainer.datamodule.eids["test"]
        partition = trainer.datamodule.partition

        # save eids and features for later
        attrib_path = os.path.join(Path(trainer.checkpoint_callback.dirpath).parent, "attributions")
        Path(attrib_path).mkdir(parents=True, exist_ok=True)

        self.write_list_to_txt(f"{attrib_path}/eids.txt", eids)
        self.write_list_to_txt(f"{attrib_path}/features.txt", features)

        # instantiate classes
        task = ShapWrapper(module.head, record_node_embeddings, len(records), gpu, module)

        task = task.to(gpu)
        raise NotImplementedError()
        explainer = DeepLiftShap(task)

        # attribute
        test_splits = torch.split(test_data, self.batch_size)

        for endpoint_idx in tqdm(range(len(endpoints))):
            endpoint_label = endpoints[endpoint_idx]

            temp_shap = torch.cat(
                [
                    explainer.attribute(
                        test_batch.to(gpu), baseline_data.to(gpu), target=endpoint_idx
                    )
                    .detach()
                    .cpu()
                    for test_batch in test_splits
                ]
            )

            fp = f"{attrib_path}/shap_{endpoint_label}_{partition}.p"
            save_tensor.remote(fp, temp_shap)

    def write_list_to_txt(self, path, l):
        with open(path, "w") as filehandle:
            for e in l:
                filehandle.write(f"{e}\n")

    def get_covariate_names(self, datamodule):
        feature_names = []
        if len(datamodule.covariate_cols) == 0:
            return feature_names

        for name, trans, column, _ in datamodule.covariate_preprocessor._iter(fitted=True):
            try:
                feature_names += list(trans.get_feature_names_out())
            except AttributeError:
                # should be numerical, i.e. feature_name_in == feature_name_out
                feature_names += list(trans.feature_names_in_)
        return feature_names

    def predict_batch(self, model, dataloader, device):
        batch = next(iter(dataloader))

        batch.graph = batch.graph.to(device)
        batch.records = batch.records.to(device)
        batch.covariates = batch.covariates.to(device)

        predictions = model(batch)
        del batch

        record_node_embeddings = predictions["record_node_embeddings"]
        if record_node_embeddings is not None:
            record_node_embeddings = record_node_embeddings.detach()
            record_node_embeddings.requires_grad = False

        return record_node_embeddings

    def prepare_data(self, datamodule, device, n_rows=None):
        # and returns an Nx10 tensor of class probabilities.
        # collect inputs
        test_data = torch.cat(
            (
                torch.Tensor(datamodule.test_dataset.records.todense()),
                torch.Tensor(datamodule.test_dataset.covariates),
            ),
            axis=1,
        )

        if n_rows is not None:
            test_data = test_data[:n_rows, :].to(device)  # TODO remov
        else:
            test_data = test_data[:, :].to(device)

        if self.baseline_mode == "zeros":
            baseline_data = torch.zeros(2, test_data.shape[1]).to(device)
        elif self.baseline_mode == "mean":
            baseline_data = test_data.mean(axis=0)[None, :].repeat(2, 1)
        else:
            assert False

        baseline_data.requires_grad = True

        return test_data, baseline_data


class EncoderFreezeUnfreeze(BaseFinetuning):
    def __init__(self, warmup_period=10):
        super().__init__()
        self._warmup_period = warmup_period

    def freeze_before_training(self, pl_module):
        self.freeze(pl_module.encoder)

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
        if current_epoch == self._warmup_period:
            self.unfreeze_and_add_param_group(
                modules=pl_module.encoder,
                optimizer=optimizer,
                train_bn=True,
            )
