import pytest
import torch
from jaxtyping import Float
from torch import Tensor, nn

from rtmdet.model import (
    ConvModule,
    CSPLayer,
    DepthwiseSeparableConvModule,
)


class CANONICAL_CSPNeXtPAFPN(nn.Module):
    def __init__(
        self,
        in_channels: tuple[int, int, int],
        out_channels: int,
        num_csp_blocks: int = 3,
        use_depthwise: bool = False,
        expand_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in reversed(range(1, len(in_channels))):
            self.reduce_layers.append(
                ConvModule(
                    in_channels=in_channels[idx],
                    out_channels=in_channels[idx - 1],
                    kernel_size=1,
                )
            )
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    expand_ratio=expand_ratio,
                )
            )

        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                conv(
                    in_channels[idx],
                    in_channels[idx],
                    3,
                    stride=2,
                    padding=1,
                )
            )
            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    expand_ratio=expand_ratio,
                )
            )

        self.out_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.out_convs.append(
                conv(
                    in_channels[i],
                    out_channels,
                    3,
                    padding=1,
                )
            )

    def forward(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        assert len(inputs) == len(self.in_channels)

        inner_outs = [inputs[-1]]
        for idx in reversed(range(1, len(self.in_channels))):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_heigh = self.reduce_layers[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = self.upsample(feat_heigh)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1)
            )
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_height], 1)
            )
            outs.append(out)

        # out convs
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])

        return tuple(outs)


def test_cspnext_pafpn():
    from rtmdet.model import CSPNeXtPAFPN

    in_channels = (128, 256, 512)
    out_channels = 256
    num_csp_blocks = 3
    use_depthwise = False
    expand_ratio = 0.5

    neck = CSPNeXtPAFPN(
        in_channels=in_channels,
        out_channels=out_channels,
        num_csp_blocks=num_csp_blocks,
        use_depthwise=use_depthwise,
        expand_ratio=expand_ratio,
    )

    inputs = [
        torch.randn(1, in_channels[0], 56, 56),
        torch.randn(1, in_channels[1], 28, 28),
        torch.randn(1, in_channels[2], 14, 14),
    ]

    outputs = neck(inputs)
    assert len(outputs) == len(in_channels)
    for output in outputs:
        assert output.shape[1] == out_channels
