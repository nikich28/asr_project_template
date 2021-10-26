import torch_audiomentations
import random
from torch import Tensor, rand, randint
from torchaudio import transforms

from hw_asr.augmentations.base import AugmentationBase


class Vol(AugmentationBase):
    def __init__(self, *args, **kwargs):
        gain = random.random()
        self.aug = transforms.Vol(gain)

    def __call__(self, wave):
        return self.aug(wave)


class Fade(AugmentationBase):
    def __init__(self, *args, **kwargs):
        f = 20
        fade = randint(0, f, size=(1, )).item()
        self.aug = transforms.Fade(fade, f - fade)

    def __call__(self, wave):
        return self.aug(wave)


class PitchShift(AugmentationBase):
    def __init__(self, sample_rate=16000, p=0.1, *args, **kwargs):
        self._aug = torch_audiomentations.PitchShift(sample_rate=sample_rate, p=p, mode='per_example',
                                                     p_mode="per_example", *args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)


class Gain(AugmentationBase):
    def __init__(self, sample_rate=16000, p=0.1, *args, **kwargs):
        self._aug = torch_audiomentations.PitchShift(sample_rate=sample_rate, p=p, mode='per_example',
                                                     p_mode="per_example", *args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)