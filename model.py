import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Variable

from math import sqrt

import random

def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)

def init_linear(linear):
    init.xavier_normal(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.kaiming_normal(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size,
                 padding,
                 kernel_size2=None, padding2=None,
                 pixel_norm=True, spectral_norm=False):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        self.conv = nn.Sequential(EqualConv2d(in_channel, out_channel,
                                            kernel1, padding=pad1),
                                nn.LeakyReLU(0.2),
                                EqualConv2d(out_channel, out_channel,
                                            kernel2, padding=pad2),
                                nn.LeakyReLU(0.2))

    def forward(self, input):
        out = self.conv(input)

        return out

class PGConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, initial=False, norm='bnorm'):
        super().__init__()

        self.conv1 = EqualConv2d(in_channel, out_channel, kernel_size, padding=padding)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size, padding=padding)
        self.lrelu2 = nn.LeakyReLU(0.2)

        if not norm is None:
            if norm == "bnorm":
                self.norm1 = nn.BatchNorm2d(num_features=out_channel)
                self.norm2 = nn.BatchNorm2d(num_features=out_channel)
            elif norm == "inorm":
                self.norm1 = nn.InstanceNorm2d(num_features=out_channel)
                self.norm2 = nn.InstanceNorm2d(num_features=out_channel)
            elif norm == "pnorm":
                self.norm1 = pixel_norm(num_features=out_channel)
                self.norm2 = pixel_norm(num_features=out_channel)

    def forward(self, input):
        out = self.conv1(input)
        out = self.norm1(out)     # 기존과 다른점 추가
        out = self.lrelu1(out)

        out = self.conv2(out)
        out = self.norm2(out)     # 기존과 다른점 추가
        out = self.lrelu2(out)

        return out

class CBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, norm="bnorm", relu=0.0):
        super().__init__()

        layers = []
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                             bias=bias)]

        if not norm is None:
            if norm == "bnorm":
                layers += [nn.BatchNorm2d(num_features=out_channels)]
            elif norm == "inorm":
                layers += [nn.InstanceNorm2d(num_features=out_channels)]
            elif norm == "pnorm":
                layers += [pixel_norm(num_features=out_channels)]

        if not relu is None and relu >= 0.0:
            layers += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)

class Encoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=512, nker=64, norm="bnorm"):
        super(Encoder, self).__init__()

        self.enc1 = CBR2d(1 * in_channels, 1 * nker, kernel_size=4, stride=2,
                          padding=1, norm=norm, relu=0.2, bias=False)

        self.enc2 = CBR2d(1 * nker, 2 * nker, kernel_size=4, stride=2,
                          padding=1, norm=norm, relu=0.2, bias=False)

        self.enc3 = CBR2d(2 * nker, 4 * nker, kernel_size=4, stride=2,
                          padding=1, norm=norm, relu=0.2, bias=False)

        self.enc4 = CBR2d(4 * nker, 8 * nker, kernel_size=4, stride=2,
                          padding=1, norm=norm, relu=0.2, bias=False)

        self.enc5 = CBR2d(8 * nker, 8 * nker, kernel_size=4, stride=2,
                          padding=1, norm=norm, relu=0.2, bias=False)

        self.enc6 = CBR2d(8 * nker, out_channels, kernel_size=3, stride=1,
                          padding=1, norm=norm, relu=0.2, bias=False)

        # 4 x 4 x 512
    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        x = self.enc6(x)

        x = torch.tanh(x)

        return x

class Encoder_PEPSI(nn.Module):
    def __init__(self, in_channels=3, out_channels=512, nker=64, norm="bnorm"):
        super(Encoder_PEPSI, self).__init__()

        self.enc1 = CBR2d(1 * in_channels, 1 * nker, kernel_size=5, stride=1,
                          padding=1, norm=norm, relu=0.2, bias=False)
        self.enc2 = CBR2d(1 * nker, 2 * nker, kernel_size=3, stride=2,
                          padding=1, norm=norm, relu=0.2, bias=False)
        self.enc3 = CBR2d(2 * nker, 2 * nker, kernel_size=3, stride=1,
                          padding=1, norm=norm, relu=0.2, bias=False)
        self.enc4 = CBR2d(2 * nker, 4 * nker, kernel_size=3, stride=2,
                          padding=1, norm=norm, relu=0.2, bias=False)
        self.enc5 = CBR2d(4 * nker, 4 * nker, kernel_size=3, stride=1,
                          padding=1, norm=norm, relu=0.2, bias=False)
        self.enc6 = CBR2d(4 * nker, out_channels, kernel_size=3, stride=2,
                          padding=1, norm=norm, relu=0.2, bias=False)

        self.enc7 = CBR2d(out_channels, out_channels, kernel_size=3, stride=1,
                          padding=1, dilation=2, norm=norm, relu=0.2, bias=False)
        self.enc8 = CBR2d(out_channels, out_channels, kernel_size=3, stride=1,
                          padding=1, dilation=3, norm=norm, relu=0.2, bias=False)
        self.enc9 = CBR2d(out_channels, out_channels, kernel_size=3, stride=1,
                          padding=1, dilation=4, norm=norm, relu=0.2, bias=False)

        # 4 x 4 x 512
    def forward(self, x):
        # print(x.shape)
        x = self.enc1(x)
        #print(x.shape)
        x = self.enc2(x)
        #print(x.shape)
        x = self.enc3(x)
        #print(x.shape)
        x = self.enc4(x)
        #print(x.shape)
        x = self.enc5(x)
        #print(x.shape)
        x = self.enc6(x)
        #print(x.shape)

        x = self.enc7(x)
        #print(x.shape)
        x = self.enc8(x)
        #print(x.shape)
        x = self.enc9(x)
        #print(x.shape)
        #x = self.enc10(x)
        #print(x.shape)

        x = torch.tanh(x)

        return x

class Generator(nn.Module):
    def __init__(self, code_dim):
        super().__init__()
        self.Pro = Encoder(3, 512, 64)
        self.progression = nn.ModuleList([PGConvBlock(512, 512, 3, 1),
                                          PGConvBlock(512, 512, 3, 1),
                                          PGConvBlock(512, 512, 3, 1),
                                          PGConvBlock(512, 512, 3, 1),
                                          PGConvBlock(512, 256, 3, 1),
                                          PGConvBlock(256, 128, 3, 1)])

        self.to_rgb = nn.ModuleList([EqualConv2d(512, 3, 1),
                                     EqualConv2d(512, 3, 1),
                                     EqualConv2d(512, 3, 1),
                                     EqualConv2d(512, 3, 1),
                                     EqualConv2d(256, 3, 1),
                                     EqualConv2d(128, 3, 1)])

    def forward(self, input, step=0):
        input = self.Pro(input)

        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):

            if i > 0 and step > 0:
                upsample = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)

                out = conv(upsample)

            else:
                out = conv(input)

            if i == step:
                out = to_rgb(out)

                break

        return out

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.progression = nn.ModuleList([ConvBlock(128, 256, 3, 1),
                                          ConvBlock(256, 512, 3, 1),
                                          ConvBlock(512, 512, 3, 1),
                                          ConvBlock(512, 512, 3, 1),
                                          ConvBlock(512, 512, 3, 1),
                                          ConvBlock(513, 512, 3, 1, 4, 0)])

        self.from_rgb = nn.ModuleList([EqualConv2d(3, 128, 1),
                                       EqualConv2d(3, 256, 1),
                                       EqualConv2d(3, 512, 1),
                                       EqualConv2d(3, 512, 1),
                                       EqualConv2d(3, 512, 1),
                                       EqualConv2d(3, 512, 1)])

        # self.blur = Blur()

        self.n_layer = len(self.progression)

        self.linear = EqualLinear(512, 1)

    def forward(self, input, step=0):
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)

            if i == 0:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0:
                out = F.interpolate(out, scale_factor=0.5, mode='bilinear', align_corners=False)

        out = out.squeeze(2).squeeze(2)
        out = self.linear(out)
        out = torch.sigmoid(out)

        return out
