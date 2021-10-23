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