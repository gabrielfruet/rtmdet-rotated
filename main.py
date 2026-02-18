from typing import Sequence

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        momentum: float = 0.03,
        eps: float = 0.001,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=momentum, eps=eps)

    def forward(
        self, x: Float[Tensor, "batch in_channels height width"]
    ) -> Float[Tensor, "batch out_channels height width"]:
        x = self.conv(x)
        x = self.bn(x)
        x = F.silu(x)
        return x


class DepthwiseSeparableConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.depthwise_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise_conv = ConvModule(in_channels, out_channels, kernel_size=1)

    def forward(
        self, x: Float[Tensor, "batch in_channels height width"]
    ) -> Float[Tensor, "batch out_channels height width"]:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class SPPBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple[int, ...] = (5, 9, 13),
    ):
        super().__init__()

        mid_channels = in_channels // 2
        self.conv1 = ConvModule(
            in_channels,
            mid_channels,
            1,
            stride=1,
        )
        self.poolings = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = mid_channels * (len(kernel_sizes) + 1)
        self.conv2 = ConvModule(
            conv2_channels,
            out_channels,
            1,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [pooling(x) for pooling in self.poolings], dim=1)
        x = self.conv2(x)
        return x


class CSPNeXtBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        add_identity: bool = True,
        use_depthwise: bool = False,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.conv1 = conv(
            in_channels,
            hidden_channels,
            3,
            stride=1,
            padding=1,
        )
        self.conv2 = DepthwiseSeparableConvModule(
            hidden_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out


class ChannelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(
        self, x: Float[Tensor, "batch channels height width"]
    ) -> Float[Tensor, "batch channels height width"]:
        """Forward function for ChannelAttention."""
        out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        return x * out


class CSPLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 0.5,
        num_blocks: int = 1,
        add_identity: bool = True,
        use_depthwise: bool = False,
        channel_attention: bool = False,
    ) -> None:
        super().__init__()
        mid_channels = int(out_channels * expand_ratio)
        self.channel_attention = channel_attention
        self.main_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
        )
        self.short_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
        )
        self.final_conv = ConvModule(
            2 * mid_channels,
            out_channels,
            1,
        )

        self.blocks = nn.Sequential(
            *[
                CSPNeXtBlock(
                    mid_channels,
                    mid_channels,
                    1.0,
                    add_identity,
                    use_depthwise,
                )
                for _ in range(num_blocks)
            ]
        )
        if channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        x_short = self.short_conv(x)

        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = torch.cat((x_main, x_short), dim=1)

        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)


class CSPNeXt(nn.Module):
    BASE_CHANNELS = [64, 128, 256, 512, 1024]
    BASE_NUM_BLOCKS = [3, 6, 6, 3]
    BASE_ADD_IDENTITY = [True, True, True, False]
    BASE_USE_SPP = [False, False, False, True]

    IMAGE_INPUT_CHANNELS = 3

    def __init__(
        self,
        arch: str = "P5",
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        out_indices: Sequence[int] = (2, 3, 4),
        frozen_stages: int = -1,
        use_depthwise: bool = False,
        expand_ratio: float = 0.5,
        spp_kernel_sizes: tuple[int, ...] = (5, 9, 13),
        channel_attention: bool = True,
        norm_eval: bool = False,
    ) -> None:
        super().__init__()
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_depthwise = use_depthwise
        self.norm_eval = norm_eval
        self.widen_channels = [int(c * widen_factor) for c in self.BASE_CHANNELS]
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.stem = nn.Sequential(
            ConvModule(
                self.IMAGE_INPUT_CHANNELS,
                int(self.widen_channels[0] // 2),
                3,
                padding=1,
                stride=2,
            ),
            ConvModule(
                int(self.widen_channels[0] // 2),
                int(self.widen_channels[0] // 2),
                3,
                padding=1,
                stride=1,
            ),
            ConvModule(
                int(self.widen_channels[0] // 2),
                int(self.widen_channels[0]),
                3,
                padding=1,
                stride=1,
            ),
        )
        self.layers = ["stem"]

        for i, (
            in_channels,
            out_channels,
            num_blocks,
            add_identity,
            use_spp,
        ) in enumerate(
            zip(
                self.widen_channels,
                self.widen_channels[1:],
                self.BASE_NUM_BLOCKS,
                self.BASE_ADD_IDENTITY,
                self.BASE_USE_SPP,
            )
        ):
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            stage = []
            conv_layer = conv(
                in_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
            )
            stage.append(conv_layer)
            if use_spp:
                spp = SPPBottleneck(
                    out_channels,
                    out_channels,
                    kernel_sizes=spp_kernel_sizes,
                )
                stage.append(spp)
            csp_layer = CSPLayer(
                out_channels,
                out_channels,
                num_blocks=num_blocks,
                add_identity=add_identity,
                use_depthwise=use_depthwise,
                expand_ratio=expand_ratio,
                channel_attention=channel_attention,
            )
            stage.append(csp_layer)
            self.add_module(f"stage{i + 1}", nn.Sequential(*stage))
            self.layers.append(f"stage{i + 1}")

    def forward(
        self, x: Float[Tensor, "batch_size channels height width"]
    ) -> tuple[Tensor, ...]:
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


class RotatedRTMDet(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(
        self, x: Float[Tensor, "batch_size channels height width"]
    ) -> tuple[Tensor, ...]:
        return self.backbone(x)


if __name__ == "__main__":
    model = RotatedRTMDet(
        CSPNeXt(deepen_factor=0.167, widen_factor=0.375, use_depthwise=False)
    )
    ckpt = torch.load(
        "./cspnext-tiny_imagenet_600e.pth",
        map_location="cpu",
    )
    missing_keys, unexpected_keys = model.load_state_dict(
        ckpt["state_dict"], strict=False
    )
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    x = torch.randn(1, 3, 640, 640)
    outs = model(x)
    for out in outs:
        print(out.shape)
