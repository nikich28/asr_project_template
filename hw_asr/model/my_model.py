from torch import nn
from torch.nn import Sequential
from hw_asr.base import BaseModel


class SimpleConvBlock(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch, stride=1, dilation=1, padding=True):
        super(SimpleConvBlock, self).__init__()
        self.block = Sequential(nn.Conv1d(in_ch, out_ch, kernel_size, stride,
                                             kernel_size // 2 if padding else 0, dilation),
                                   nn.BatchNorm1d(out_ch),
                                   nn.ReLU(inplace=True)
                                   )

    def forward(self, x):
        return self.block(x)


class TCSConvBlock(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch, activation=False, stride=1, dilation=1, padding=True):
        super(TCSConvBlock, self).__init__()
        self.block = Sequential(nn.Conv1d(in_ch, in_ch, kernel_size, stride=stride,
                                          padding=kernel_size // 2, dilation=dilation, groups=in_ch),
                                nn.Conv1d(in_ch, out_ch, kernel_size=1),
                                nn.BatchNorm1d(out_ch)
                                )
        self.activation = nn.ReLU(inplace=True) if activation else None

    def forward(self, x):
        if self.activation is not None:
            return self.activation(self.block(x))
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch, R, ind):
        super(ResidualBlock, self).__init__()
        self.blocks = R
        if ind != 0:
            in_ch = out_ch
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        self.BN = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        layers = nn.ModuleList([TCSConvBlock(kernel_size, in_ch, out_ch, activation=True)])
        for _ in range(1, self.blocks - 1):
            layers.append(TCSConvBlock(kernel_size, out_ch, out_ch, True))
        layers.append(TCSConvBlock(kernel_size, out_ch, out_ch))
        self.seq = Sequential(*layers)

    def forward(self, x):
        id = self.BN(self.pointwise(x))
        x = self.seq(x)
        return self.relu(x + id)


class QuartzNet(BaseModel):
    def __init__(self, n_feats, n_class, size=1, *args, **kwargs):
        #nums - number of spectrogram
        #size could be 1, 2, 3 for 3 different sizes of QN (5, 10 or 15)x5
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.net = Sequential(
            SimpleConvBlock(33, n_feats, 256),
            *[ResidualBlock(33, 256, 256, 5, i) for i in range(size)],
            *[ResidualBlock(39, 256, 256, 5, i) for i in range(size)],
            *[ResidualBlock(51, 256, 512, 5, i) for i in range(size)],
            *[ResidualBlock(63, 512, 512, 5, i) for i in range(size)],
            *[ResidualBlock(75, 512, 512, 5, i) for i in range(size)],
            SimpleConvBlock(87, 512, 512),
            SimpleConvBlock(1, 512, 1024),
            SimpleConvBlock(1, 1024, n_class)
        )

    def forward(self, spectrogram, *args, **kwargs):
        spectrogram = spectrogram.transpose(1, 2)
        res = self.net(spectrogram)
        return {"logits": res.transpose(1, 2)}

    def transform_input_lengths(self, input_lengths):
        return input_lengths // 2 # we don't reduce time dimension here
