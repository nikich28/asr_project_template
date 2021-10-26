from torchaudio import transforms
from torch import randn, Tensor
import random

from hw_asr.augmentations.base import AugmentationBase


class FrequencyMasking(AugmentationBase):
    def __init__(self, frequency=15, *args, **kwargs):
        self.augm = transforms.FrequencyMasking(frequency)
        self.p = 1

    def __call__(self, data: Tensor):
        if random.random() < self.p:
            x = data.unsqueeze(1)
            return self.augm(x).squeeze(1)
        return data


class TimeMasking(AugmentationBase):
    def __init__(self, frequency=35, *args, **kwargs):
        self.augm = transforms.TimeMasking(frequency)
        self.p = 1

    def __call__(self, data: Tensor):
        if random.random() < self.p:
            x = data.unsqueeze(1)
            return self.augm(x).squeeze(1)
        return data


class Normalize(AugmentationBase):
    def __init__(self, mean=0.0, std=1.0, eps=1e-8, *args, **kwargs):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, data: Tensor):
        data = (data - self.mean) / (self.std + self.eps)
        return data


class Noise(AugmentationBase):
    def __init__(self, scale=0.05, *args, **kwargs):
        self.scale = scale

    def __call__(self, data: Tensor):
        return data + self.scale * randn(data.shape)