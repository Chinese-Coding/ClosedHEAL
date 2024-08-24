from typing import List, Type

from torch import nn, Tensor
from torchvision.models.resnet import BasicBlock, Bottleneck

"""从 torchvision.models.ResNet 文件中拿来的一个函数"""


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


Block = Type[BasicBlock | Bottleneck]


class Resnet(nn.Module):
    """
    自己 resnet 的实现, 源项目中的实现参考了 torchvision.models.ResNet 中的实现. 而官方库的实现为了通用性有许多参数.
    然而, 复写该项目时则不需要考虑这么多, 因此去除许多参数, 重新编排一下. 所以该类只包含了用到的参数, 没用到的参数全部默认处理了
    一开始我本打算让这个类继承官方实现的 ResNet, 然而, 官方实现的 ResNet 中只有三层, 并且不支持 strides, 所以放弃.
    """

    def __init__(self, block: Block, layers: List[int], layer_strides: List[int], num_filters: List[int],
                 groups=1, width_per_group=64, inplanes=64):  # fmt:skip
        super().__init__()

        """一些未使用到的参数, 对应的属性全部按照默认值初始化"""
        self.norm_layer = nn.BatchNorm2d
        self.dilation = 1

        self.groups, self.base_width, self.inplanes = groups, width_per_group, inplanes

        self.layers = [
            self._make_layer(block, filters, layer, stride)
            # `layers`, `layer_strides` 和 `num_filters` 长度是相等的, 这一点在调用的时候就保证了
            for filters, layer, stride in zip(num_filters, layers, layer_strides)
        ]
        self._init_weights()

    def _zero_init_residual(self):
        """
        该函数默认没有调用, 但是看起来性能有提升, 所以先写在这里. 下面是源项目中的注释:
        Zero-initialize the last BN in each residual branch, so that the residual branch starts with zeros, and each residual block behaves like an identity.
        This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        """
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Block, planes: int, blocks: int, stride: int = 1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                self.norm_layer(planes * block.expansion),
            )
        layers = [
            block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, self.norm_layer)
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            # 这里传参的时候 `stride` 和 `downsample` 传递的都是默认值, 只是为了方便后续传参的时候不必显示的写出参数的名字
            layers.append(block(self.inplanes, planes, 1, None, self.groups, self.base_width, self.dilation, self.norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> List[Tensor]:
        interm_features = []
        for layer in self.layers:
            x = layer(x)
            interm_features.append(x)
        return interm_features


if __name__ == "__main__":
    from icecream import ic
    import torch

    Bottleneck.expansion = 1
    model = Resnet(Bottleneck, [3, 4, 5], [1, 2, 2], [64, 128, 256], 32, 4)
    input = torch.randn(4, 64, 200, 704)
    output = model(input)

    for out in output:
        ic(out.shape)
    ic(model)
