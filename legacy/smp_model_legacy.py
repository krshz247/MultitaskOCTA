"""Network classes of the original BSDA-Net code that the current pipeline does not use (moved out of smp_model.py):
single-input classifier encoders, the first-connect decoder and MyUnetModel. Run with PYTHONPATH=.:legacy from the repository root."""
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import CenterBlock, DecoderBlock
from segmentation_models_pytorch.base import SegmentationModel, SegmentationHead
import segmentation_models_pytorch.base.initialization as init
from segmentation_models_pytorch.encoders.resnet import ResNetEncoder
from segmentation_models_pytorch.encoders.vgg import VGGEncoder


from smp_model import MyUnetDecoder


class MyResNetEncoder(ResNetEncoder):

    def __init__(self, out_channels, depth, decoder_channels, **kwargs):
        super().__init__(out_channels, depth, **kwargs)
        self.my_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.my_fc = nn.Linear(512 * kwargs["block"].expansion, kwargs["num_classes"])

        self.cat_convs = nn.Sequential(
            nn.Conv2d(out_channels[1] + decoder_channels[3], out_channels[1], kernel_size=1, stride=1, bias=False),
            nn.Conv2d(out_channels[2] + decoder_channels[2], out_channels[2], kernel_size=1, stride=1, bias=False),
            nn.Conv2d(out_channels[3] + decoder_channels[1], out_channels[3], kernel_size=1, stride=1, bias=False),
            nn.Conv2d(out_channels[4] + decoder_channels[0], out_channels[4], kernel_size=1, stride=1, bias=False)
        )
        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, decoder_features):

        decoder_features = decoder_features[::-1]
        '''for j, fe in enumerate(decoder_features):
            print(j, fe.shape)
        print()'''

        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            # print(stages[i])
            x = stages[i](x)
            # print(i, x.shape)
            if i > 0 and i < 5:
                skip = decoder_features[i]
                # print(skip.shape)
                x = torch.cat([x, skip], dim=1)
                x = self.cat_convs[i - 1](x)
                # print(x.shape)
            # print()
            features.append(x)

        x = self.my_avgpool(x)
        x = torch.flatten(x, 1)
        x = self.my_fc(x)

        return x


class MyVggEncoder(VGGEncoder):

    def __init__(self, out_channels, config, decoder_channels, batch_norm=False, depth=5, **kwargs):
        super().__init__(out_channels, config, batch_norm, depth, **kwargs)
        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3

        self.my_avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.my_classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, kwargs["num_classes"]),
        )

        self.cat_convs = nn.Sequential(
            nn.Conv2d(out_channels[1] + decoder_channels[3], out_channels[1], kernel_size=1, stride=1, bias=False),
            nn.Conv2d(out_channels[2] + decoder_channels[2], out_channels[2], kernel_size=1, stride=1, bias=False),
            nn.Conv2d(out_channels[3] + decoder_channels[1], out_channels[3], kernel_size=1, stride=1, bias=False),
            nn.Conv2d(out_channels[4] + decoder_channels[0], out_channels[4], kernel_size=1, stride=1, bias=False)
        )
        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, decoder1_features):

        decoder1_features = decoder1_features[::-1]
        # decoder2_features = decoder2_features[::-1]
        # decoder3_features = decoder3_features[::-1]
        '''for j, fe in enumerate(decoder_features):
            print(j, fe.shape)
        print()'''

        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            # print(stages[i])
            x = stages[i](x)
            # print(i, x.shape)
            if i > 0 and i < 5:
                skip1 = decoder1_features[i]
                # skip2 = decoder2_features[i]
                # skip3 = decoder3_features[i]
                # print(skip.shape)
                x = torch.cat([x, skip1], dim=1)
                x = self.cat_convs[i - 1](x)
                # print(x.shape)
            # print()
            features.append(x)

        x = self.my_avgpool(x)
        x = torch.flatten(x, 1)
        x = self.my_classifier(x)

        return x


class MyUnetDecoder_withfirstconnect(nn.Module):

    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[0:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[0:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        decoder_features = []

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            '''if i < len(skips):
                print(x.shape, skip.shape)
            else:
                print(x.shape, skip)'''
            x = decoder_block(x, skip)
            # print(decoder_block, x.shape)
            decoder_features.append(x)

        return [x, decoder_features]


class MyUnetModel(SegmentationModel):

    def __init__(
        self,
        encoder_name="resnet34",
        encoder_depth=5,
        encoder_weights="imagenet",
        decoder_use_batchnorm=True,
        decoder_channels=(256, 128, 64, 32, 16),
        decoder_attention_type=None,
        in_channels=3,
        classes=2,
        activation=None,
        aux_params=None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder1 = MyUnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        '''self.decoder2 = MyUnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.decoder3= MyUnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )'''

        self.segmentation_head1 = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        '''self.segmentation_head2 = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        self.segmentation_head3 = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )'''
        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder1)
        init.initialize_head(self.segmentation_head1)
        '''init.initialize_decoder(self.decoder2)
        init.initialize_head(self.segmentation_head2)
        init.initialize_decoder(self.decoder3)
        init.initialize_head(self.segmentation_head3)'''

    def forward(self, x):
        features = self.encoder(x)
        '''for f in features:
            print(f.shape)'''
        decoder1_output, decoder1_features = self.decoder1(*features)
        # decoder2_output = self.decoder2(*features)
        # decoder3_output = self.decoder3(*features)

        mask1 = self.segmentation_head1(decoder1_output)
        # mask2 = self.segmentation_head2(decoder2_output)
        # mask3 = self.segmentation_head3(decoder3_output)

        # return [mask1, mask2, mask3]
        return [mask1, decoder1_features]
