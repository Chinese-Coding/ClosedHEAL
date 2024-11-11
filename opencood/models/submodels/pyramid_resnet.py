from typing import Type, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import conv1x1, BasicBlock, Bottleneck

Block = Type[Union[BasicBlock, Bottleneck]]


class PyramidResnet(nn.Module):
    # fmt: off
    def __init__(
        self, block: Block,
        layers: List[int], layer_strides: List[int], num_filters: List[int],
        groups= 1, width_per_group= 64, norm_layer: Optional[Callable[..., nn.Module]] = None, inplanes=64
    ):
    # fmt: on
        """
        # number of block in one layer
        # stride after one layer
        # feature dim
        """
        super().__init__()
        self._norm_layer = nn.BatchNorm2d if norm_layer is None else norm_layer
        self.inplanes, self.dilation = inplanes, 1
        self.groups, self.base_width = groups, width_per_group

        self.layers = nn.ModuleList(
            [self._make_layer(block, num_filters[i], layers[i], layer_strides[i]) for i, _ in enumerate(num_filters)]
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Block, planes: int, blocks: int, stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        def _MakeBlock(inplanes, planes, stride=1, downsample=None, dilation=1):
            return block(inplanes, planes, stride, downsample, self.groups, self.base_width, dilation, norm_layer)

        # if stride != 1, the first block will downsample the feature map plane is the feature dim
        # if Bottleneck, then the output dim is planes * block.expansion(4)
        layers = [_MakeBlock(self.inplanes, planes, stride, downsample, previous_dilation)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(_MakeBlock(self.inplanes, planes, dilation=self.dilation))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor, return_inter: bool = True):
        inter_features = []
        for layer in self.layers:
            x = layer(x)
            inter_features.append(x)
        return inter_features if return_inter else inter_features[-1]


if __name__ == "__main__":
    Bottleneck.expansion = 1
    model = PyramidResnet(
        Bottleneck, layers=[3, 4, 5], layer_strides=[1, 2, 2], num_filters=[64, 128, 256], groups=32, width_per_group=4
    )
    input = torch.randn(4, 64, 200, 704)
    print(model)
    output = model(input)
    from icecream import ic

    for out in output:
        ic(out.shape)
