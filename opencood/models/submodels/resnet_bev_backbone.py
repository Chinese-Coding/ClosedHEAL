# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import numpy as np
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock

from opencood.models.submodels.pyramid_resnet import PyramidResnet

DEBUG = False


class ResnetBEVBackbone(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        if "layer_nums" in model_cfg:
            layer_nums, layer_strides, num_filters = model_cfg["layer_nums"], model_cfg["layer_strides"], model_cfg["num_filters"] # fmt: skip
            assert len(layer_nums) == len(layer_strides) == len(num_filters)
        else:
            layer_nums = layer_strides = num_filters = []

        if "upsample_strides" in model_cfg:
            num_upsample_filters, upsample_strides = model_cfg["num_upsample_filter"], model_cfg["upsample_strides"]
            assert len(num_upsample_filters) == len(upsample_strides)
        else:
            upsample_strides = num_upsample_filters = []

        self.resnet = PyramidResnet(BasicBlock, layer_nums, layer_strides, num_filters, inplanes=model_cfg.get("inplanes", 64))

        self.num_levels = len(layer_nums)
        self.deblocks = nn.ModuleList()
        if len(upsample_strides) > 0:
            self._MakeDeblocks(upsample_strides, num_filters, num_upsample_filters)

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > self.num_levels:
            self.deblocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                    nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                )
            )

        self.num_bev_features = c_in
        self.deblocks_len = len(self.deblocks)

    def _MakeDeblocks(self, upsample_strides, num_filters, num_upsample_filters):
        for i in range(self.num_levels):
            stride = upsample_strides[i]
            if stride >= 1:
                self.deblocks.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[i],
                            num_upsample_filters[i],
                            upsample_strides[i],
                            stride=upsample_strides[i],
                            bias=False,
                        ),
                        nn.BatchNorm2d(num_upsample_filters[i], eps=1e-3, momentum=0.01),
                        nn.ReLU(),
                    )
                )
            else:
                stride = np.round(1 / stride).astype(np.int)
                self.deblocks.append(
                    nn.Sequential(
                        nn.Conv2d(num_filters[i], num_upsample_filters[i], stride, stride=stride, bias=False),
                        nn.BatchNorm2d(num_upsample_filters[i], eps=1e-3, momentum=0.01),
                        nn.ReLU(),
                    )
                )

    def forward(self, spatial_features):
        x = self.resnet(spatial_features)  # tuple of features
        ups = (
            [self.deblocks[i](x[i]) for i in range(self.num_levels)]
            if self.deblocks_len > 0
            else [x[i] for i in range(self.num_levels)]
        )
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if self.deblocks_len > self.num_levels:
            x = self.deblocks[-1](x)

        return x

    # these two functions are seperated for multiscale intermediate fusion
    def get_multiscale_feature(self, spatial_features):
        """before multiscale intermediate fusion"""
        x = self.resnet(spatial_features)  # tuple of features
        return x

    def decode_multiscale_feature(self, x):
        """after multiscale interemediate fusion"""
        ups = []
        for i in range(self.num_levels):
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x[i]))
            else:
                ups.append(x[i])
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > self.num_levels:
            x = self.deblocks[-1](x)
        return x

    def get_layer_i_feature(self, spatial_features, layer_i):
        """before multiscale intermediate fusion"""
        return eval(f"self.resnet.layer{layer_i}")(spatial_features)  # tuple of features
