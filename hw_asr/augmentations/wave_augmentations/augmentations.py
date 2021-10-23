import torch_audiomentations
from torch import Tensor, rand, randint
from torchaudio.transforms import Fade, Vol

from hw_asr.augmentations.base import AugmentationBase


class Vol(AugmentationBase):
    def __init__(self, *args, **kwargs):
        gain = rand((1,)).item()
        self.aug = Vol(gain)

    def __call__(self, wave):
        return self.aug(wave)


class Fade(AugmentationBase):
    def __init__(self, *args, **kwargs):
        f = 20
        fade = randint(0, f, size=(1, )).item()
        self.aug = Fade(fade, f - fade)

    def __call__(self, wave):
        return self.aug(wave)