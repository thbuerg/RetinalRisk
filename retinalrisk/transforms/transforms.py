from typing import Union

import PIL
import torch.nn as nn
import torchvision as tv
import torchvision.transforms.functional as TF
from torchvision import transforms
from random import choice
from omegaconf import ListConfig


class AdaptiveRandomCropTransform(nn.Module):
    def __init__(
        self,
        crop_ratio: Union[list, float],
        out_size: int,
        interpolation=TF.InterpolationMode.BILINEAR,
    ):
        super().__init__()
        self.crop_ratio = crop_ratio
        self.out_size = out_size
        self.interpolation = interpolation

    def forward(self, sample):
        input_size = min(sample.size)
        if isinstance(self.crop_ratio, (list, ListConfig)):
            crop_ratio = choice(self.crop_ratio)
        else:
            crop_ratio = self.crop_ratio

        crop_size = int(crop_ratio * input_size)
        if crop_size < self.out_size:
            crop_size = tv.transforms.transforms._setup_size(
                self.out_size, error_msg="Please provide only two dimensions (h, w) for size."
            )
            i, j, h, w = transforms.RandomCrop.get_params(sample, crop_size)
            return TF.crop(sample, i, j, h, w)
        else:
            crop_size = tv.transforms.transforms._setup_size(
                crop_size, error_msg="Please provide only two dimensions (h, w) for size."
            )
            i, j, h, w = transforms.RandomCrop.get_params(sample, crop_size)
            cropped = TF.crop(sample, i, j, h, w)
        out = TF.resize(cropped, self.out_size, self.interpolation)
        return out
