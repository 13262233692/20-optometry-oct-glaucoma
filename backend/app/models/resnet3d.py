from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3x3(
    in_planes: int,
    out_planes: int,
    stride: Union[int, Tuple[int, int, int]] = 1,
    groups: int = 1,
    dilation: Union[int, Tuple[int, int, int]] = 1
) -> nn.Conv3d:
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )


def conv1x1x1(
    in_planes: int,
    out_planes: int,
    stride: Union[int, Tuple[int, int, int]] = 1
) -> nn.Conv3d:
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False
    )


class BasicBlock3D(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: Union[int, Tuple[int, int, int]] = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        norm_layer: Optional[Type[nn.Module]] = None,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.InstanceNorm3d

        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")

        self.conv1 = conv3x3x3(inplanes, planes, stride, dilation=dilation)
        self.norm1 = norm_layer(planes, affine=True)
        self.relu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.norm2 = norm_layer(planes, affine=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck3D(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: Union[int, Tuple[int, int, int]] = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        norm_layer: Optional[Type[nn.Module]] = None,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.InstanceNorm3d

        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1x1(inplanes, width)
        self.norm1 = norm_layer(width, affine=True)

        self.conv2 = conv3x3x3(width, width, stride, groups, dilation)
        self.norm2 = norm_layer(width, affine=True)

        self.conv3 = conv1x1x1(width, planes * self.expansion)
        self.norm3 = norm_layer(planes * self.expansion, affine=True)

        self.relu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet3DEncoder(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock3D, Bottleneck3D]],
        layers: List[int],
        in_channels: int = 1,
        zero_init_residual: bool = True,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Type[nn.Module]] = None,
        dropout_rate: float = 0.1
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.InstanceNorm3d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv3d(
            in_channels, self.inplanes,
            kernel_size=7, stride=(2, 2, 2), padding=3, bias=False
        )
        self.norm1 = norm_layer(self.inplanes, affine=True)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2,
            dilate=replace_stride_with_dilation[0], dropout_rate=dropout_rate
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2,
            dilate=replace_stride_with_dilation[1], dropout_rate=dropout_rate
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2,
            dilate=replace_stride_with_dilation[2], dropout_rate=dropout_rate
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, (nn.InstanceNorm3d, nn.GroupNorm, nn.BatchNorm3d)):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck3D):
                    if m.norm3.affine:
                        nn.init.constant_(m.norm3.weight, 0)
                elif isinstance(m, BasicBlock3D):
                    if m.norm2.affine:
                        nn.init.constant_(m.norm2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock3D, Bottleneck3D]],
        planes: int,
        blocks: int,
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilate: bool = False,
        dropout_rate: float = 0.1
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, affine=True),
            )

        layers = []
        layers.append(block(
            self.inplanes, planes, stride, downsample, self.groups,
            self.base_width, previous_dilation, norm_layer, dropout_rate
        ))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(
                self.inplanes, planes, groups=self.groups,
                base_width=self.base_width, dilation=self.dilation,
                norm_layer=norm_layer, dropout_rate=dropout_rate
            ))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        skip_connections = []

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        skip_connections.append(x)

        x = self.maxpool(x)
        x = self.layer1(x)
        skip_connections.append(x)

        x = self.layer2(x)
        skip_connections.append(x)

        x = self.layer3(x)
        skip_connections.append(x)

        x = self.layer4(x)
        skip_connections.append(x)

        return skip_connections


class DecoderBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm_layer: Optional[Type[nn.Module]] = None,
        dropout_rate: float = 0.1
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.InstanceNorm3d

        self.upsample = nn.ConvTranspose3d(
            in_channels, out_channels,
            kernel_size=2, stride=2
        )

        total_channels = out_channels + skip_channels

        self.conv1 = conv3x3x3(total_channels, out_channels)
        self.norm1 = norm_layer(out_channels, affine=True)
        self.relu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.conv2 = conv3x3x3(out_channels, out_channels)
        self.norm2 = norm_layer(out_channels, affine=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        if x.shape != skip.shape:
            diff_d = skip.shape[2] - x.shape[2]
            diff_h = skip.shape[3] - x.shape[3]
            diff_w = skip.shape[4] - x.shape[4]
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                          diff_h // 2, diff_h - diff_h // 2,
                          diff_d // 2, diff_d - diff_d // 2])

        x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        return x


class AttentionGate3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.W_g = nn.Sequential(
            conv1x1x1(in_channels, out_channels),
            nn.InstanceNorm3d(out_channels, affine=True)
        )
        self.W_x = nn.Sequential(
            conv1x1x1(skip_channels, out_channels),
            nn.InstanceNorm3d(out_channels, affine=True)
        )
        self.upsample_g = nn.ConvTranspose3d(
            out_channels, out_channels, kernel_size=2, stride=2
        )
        self.psi = nn.Sequential(
            conv1x1x1(out_channels, 1),
            nn.InstanceNorm3d(1, affine=True),
            nn.Sigmoid()
        )
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if g1.shape[2:] != x1.shape[2:]:
            g1 = self.upsample_g(g1)
            if g1.shape[2:] != x1.shape[2:]:
                g1 = F.interpolate(
                    g1, size=x1.shape[2:], mode="trilinear", align_corners=False
                )

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ResNet3DUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        encoder_depth: int = 18,
        use_attention: bool = True,
        dropout_rate: float = 0.1,
        deep_supervision: bool = True
    ):
        super().__init__()

        self.use_attention = use_attention
        self.deep_supervision = deep_supervision

        if encoder_depth == 10:
            block = BasicBlock3D
            layers = [1, 1, 1, 1]
        elif encoder_depth == 18:
            block = BasicBlock3D
            layers = [2, 2, 2, 2]
        elif encoder_depth == 34:
            block = BasicBlock3D
            layers = [3, 4, 6, 3]
        elif encoder_depth == 50:
            block = Bottleneck3D
            layers = [3, 4, 6, 3]
        else:
            raise ValueError(f"Unsupported encoder depth: {encoder_depth}")

        self.encoder = ResNet3DEncoder(
            block=block,
            layers=layers,
            in_channels=in_channels,
            dropout_rate=dropout_rate
        )

        encoder_channels = [64, 64 * block.expansion, 128 * block.expansion,
                           256 * block.expansion, 512 * block.expansion]
        decoder_channels = [256, 128, 64, 32]

        self.bottleneck = nn.Sequential(
            conv3x3x3(encoder_channels[-1], 512),
            nn.InstanceNorm3d(512, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout3d(p=dropout_rate),
            conv3x3x3(512, 512),
            nn.InstanceNorm3d(512, affine=True),
            nn.LeakyReLU(inplace=True)
        )

        self.decoders = nn.ModuleList()
        self.attention_gates = nn.ModuleList() if use_attention else None

        for i, (enc_ch, dec_ch) in enumerate(zip(
            reversed(encoder_channels[1:]),
            decoder_channels
        )):
            if use_attention:
                gate = AttentionGate3D(
                    in_channels=decoder_channels[i - 1] if i > 0 else 512,
                    skip_channels=enc_ch,
                    out_channels=enc_ch // 2
                )
                self.attention_gates.append(gate)

            decoder = DecoderBlock3D(
                in_channels=decoder_channels[i - 1] if i > 0 else 512,
                skip_channels=enc_ch,
                out_channels=dec_ch,
                dropout_rate=dropout_rate
            )
            self.decoders.append(decoder)

        self.final_upsample = nn.ConvTranspose3d(
            decoder_channels[-1], decoder_channels[-1],
            kernel_size=2, stride=2
        )

        self.final_conv = nn.Sequential(
            conv3x3x3(decoder_channels[-1] + encoder_channels[0], decoder_channels[-1]),
            nn.InstanceNorm3d(decoder_channels[-1], affine=True),
            nn.LeakyReLU(inplace=True),
            conv3x3x3(decoder_channels[-1], decoder_channels[-1] // 2),
            nn.InstanceNorm3d(decoder_channels[-1] // 2, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(decoder_channels[-1] // 2, num_classes, kernel_size=1)
        )

        if deep_supervision:
            self.ds_conv1 = nn.Conv3d(decoder_channels[1], num_classes, kernel_size=1)
            self.ds_conv2 = nn.Conv3d(decoder_channels[2], num_classes, kernel_size=1)

        self.activation = nn.Softmax(dim=1) if num_classes > 1 else nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        input_size = x.shape[2:]

        skips = self.encoder(x)
        x = self.bottleneck(skips[-1])

        decoder_outputs = []
        reversed_skips = list(reversed(skips[1:]))

        for i, (decoder, skip) in enumerate(zip(self.decoders, reversed_skips)):
            if self.use_attention and self.attention_gates is not None:
                skip = self.attention_gates[i](x, skip)
            x = decoder(x, skip)
            decoder_outputs.append(x)

        x = self.final_upsample(x)

        if x.shape[2:] != skips[0].shape[2:]:
            x = F.interpolate(x, size=skips[0].shape[2:], mode="trilinear", align_corners=False)

        x = torch.cat([x, skips[0]], dim=1)
        x = self.final_conv(x)

        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="trilinear", align_corners=False)

        if self.training and self.deep_supervision:
            ds1 = self.ds_conv1(decoder_outputs[1])
            ds1 = F.interpolate(ds1, size=input_size, mode="trilinear", align_corners=False)
            ds2 = self.ds_conv2(decoder_outputs[2])
            ds2 = F.interpolate(ds2, size=input_size, mode="trilinear", align_corners=False)
            return [x, ds1, ds2]

        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        if isinstance(logits, list):
            logits = logits[0]
        return self.activation(logits)


class ResNet3DForRNFL(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        model_depth: int = 18,
        **kwargs
    ):
        super().__init__()
        self.model = ResNet3DUNet(
            in_channels=in_channels,
            num_classes=num_classes,
            encoder_depth=model_depth,
            use_attention=True,
            dropout_rate=0.15,
            deep_supervision=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.predict(x)

    def get_parameters_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def create_resnet3d_model(
    in_channels: int = 1,
    num_classes: int = 2,
    model_name: str = "resnet3d_18",
    pretrained: bool = False,
    **kwargs
) -> ResNet3DForRNFL:
    depth_map = {
        "resnet3d_10": 10,
        "resnet3d_18": 18,
        "resnet3d_34": 34,
        "resnet3d_50": 50
    }

    if model_name not in depth_map:
        raise ValueError(f"Unknown model name: {model_name}")

    model = ResNet3DForRNFL(
        in_channels=in_channels,
        num_classes=num_classes,
        model_depth=depth_map[model_name],
        **kwargs
    )

    if pretrained:
        import warnings
        warnings.warn(
            "Medical image pretraining weights not included. "
            "Train with your dataset for best results."
        )

    return model
