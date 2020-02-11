import torch.nn as nn
import torchvision
#from segmentation_models_pytorch.fpn.decoder import FPNDecoder
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import *
import torch
from torchsummary import summary
from collections import OrderedDict
import pretrainedmodels

class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):

        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3),
                              stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x

class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x):


        x, skip = x
        #print(skip.size())
        #print(x.size())
        x = F.interpolate(x, size=(skip.size(2), skip.size(3)), mode='nearest')
        skip = self.skip_conv(skip)

        x = x + skip

        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [
            Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))
        ]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)

class SegmentationBlockV2(nn.Module):

    def __init__(self, in_channels, out_channels): #n_upsamples=0):
        super().__init__()

        blocks = [
            Conv3x3GNReLU(in_channels, out_channels, upsample=False),
            Conv3x3GNReLU(out_channels, out_channels, upsample=False)
        ]
        #self.n_upsamples = n_upsamples

        #if n_upsamples > 1:
            #for _ in range(1, n_upsamples):
                #blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):

        x = self.block(x)
        #if self.n_upsamples > 0:
            #x = F.interpolate(x, scale_factor=2**self.n_upsamples, mode='bilinear', align_corners=True)

        return x

class FPNDecoder(Model):

    def __init__(
            self,
            encoder_channels,
            pyramid_channels=256,
            segmentation_channels=128,
            final_channels=1,
            dropout=0.2,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=(1, 1))

        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

        self.s5 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=3)
        self.s4 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=2)
        self.s3 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=1)
        self.s2 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=0)

        self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        self.final_conv = nn.Conv2d(segmentation_channels*4, final_channels, kernel_size=1, padding=0)

        self.initialize()

    def forward(self, x):
        c5, c4, c3, c2, _ = x

        p5 = self.conv1(c5)
        p4 = self.p4([p5, c4])
        p3 = self.p3([p4, c3])
        p2 = self.p2([p3, c2])

        s5 = self.s5(p5)
        s4 = self.s4(p4)
        s3 = self.s3(p3)
        s2 = self.s2(p2)

        x = torch.cat([s5,s4,s3,s2], dim=1)
        x = self.dropout(x)
        x = self.final_conv(x)

        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

        return x

class FPNDecoderV2(Model):

    def __init__(
            self,
            encoder_channels,
            pyramid_channels=256,
            segmentation_channels=128,
            final_channels=1,
            dropout=0.2,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=(1, 1))

        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

        self.s5 = SegmentationBlockV2(pyramid_channels, segmentation_channels)#, n_upsamples=3)
        self.s4 = SegmentationBlockV2(pyramid_channels, segmentation_channels)#, n_upsamples=2)
        self.s3 = SegmentationBlockV2(pyramid_channels, segmentation_channels)#, n_upsamples=1)
        self.s2 = SegmentationBlockV2(pyramid_channels, segmentation_channels)#, n_upsamples=0)

        self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        self.final_conv = nn.Conv2d(segmentation_channels*4, final_channels, kernel_size=1, padding=0)

        self.initialize()

    def forward(self, x):
        c5, c4, c3, c2, _ = x

        p5 = self.conv1(c5)
        p4 = self.p4([p5, c4])
        p3 = self.p3([p4, c3])
        p2 = self.p2([p3, c2])

        #print(p5.size())
        s5 = self.s5(p5)
        #print(s5.size())
        s4 = self.s4(p4)
        #print(s4.size())
        s3 = self.s3(p3)
        #print(s3.size())
        s2 = self.s2(p2)
        #print(s2.size())
        h, w = s2.size()[2:]
        #s5 = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)


        x = torch.cat([F.upsample_bilinear(s5, size=(h, w)),
                       F.upsample_bilinear(s4, size=(h, w)),
                       F.upsample_bilinear(s2, size=(h, w)),
                       s2], dim=1)
        x = self.dropout(x)
        x = self.final_conv(x)

        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

        return x

class ResNetEncoder(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        # self.pretrained = pretrained
        self.resnet = torchvision.models.resnet34(pretrained)
        del self.resnet.fc

    def forward(self, x):
        x0 = self.resnet.conv1(x)
        x0 = self.resnet.bn1(x0)
        x0 = self.resnet.relu(x0)

        x1 = self.resnet.maxpool(x0)
        x1 = self.resnet.layer1(x1)

        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        return [x4, x3, x2, x1, x0]


class ResNet_FPN(nn.Module):

    def __init__(self):
        super().__init__()
        self.model_encoder = ResNetEncoder()
        self.model_decoder = FPNDecoder(encoder_channels=(512, 256, 128, 64, 64),
                                        pyramid_channels=256,
                                        segmentation_channels=64,
                                        final_channels=4,
                                        dropout=0.2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512, 1, bias=True)

    def forward(self, x):

        #x = self.model_encoder(x)
        global_features = self.model_encoder(x)

        cls_feature = global_features[0]
        cls_feature = self.avgpool(cls_feature)
        cls_feature = cls_feature.view(cls_feature.size(0), -1)
        cls_feature = self.fc(cls_feature)

        seg_feature = self.model_decoder(global_features)

        return seg_feature, cls_feature

class EfficientNet_5_Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = EfficientNet.from_pretrained('efficientnet-b5')

    def forward(self, inputs):
        x = relu_fn(self.model._bn0(self.model._conv_stem(inputs)))

        global_features = []

        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx in [2, 7, 12, 26]:
                global_features.append(x)
        x = relu_fn(self.model._bn1(self.model._conv_head(x)))
        global_features.append(x)
        global_features.reverse()

        return global_features


class EfficientNet_3_Encoder(nn.Module):

    def __init__(self):
        super(EfficientNet_3_Encoder, self).__init__()

        self.model = EfficientNet.from_pretrained('efficientnet-b3')

    def forward(self, inputs):
        x = relu_fn(self.model._bn0(self.model._conv_stem(inputs)))

        global_features = []

        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx in [1, 4, 7, 17]:
                global_features.append(x)
        x = relu_fn(self.model._bn1(self.model._conv_head(x)))
        global_features.append(x)
        global_features.reverse()

        return global_features


class EfficientNet_b3_FPN(nn.Module):

    def __init__(self):
        super().__init__()
        self.model_encoder = EfficientNet_3_Encoder()
        self.model_decoder = FPNDecoder(encoder_channels=(1536, 136, 48, 32, 24),
                                        pyramid_channels=256,
                                        segmentation_channels=64,
                                        final_channels=4,
                                        dropout=0.2)
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc      = nn.Sequential(nn.Linear(1536, 1536, bias=True),
                                     #nn.Linear(1536, 1, bias=True))
                                     #nn.BatchNorm2d(1536 // 4),
                                     #nn.ReLU(),
                                     #nn.Linear(1536//4, 4))

    def forward(self, x):

        global_features = self.model_encoder(x)
        #cls_feature = global_features[0]
        #cls_feature = self.avgpool(cls_feature)
        #cls_feature = cls_feature.view(cls_feature.size(0), -1)
        seg_feature = self.model_decoder(global_features)

        return seg_feature#, cls_feature

class EfficientNet_b5_FPN(nn.Module):

    def __init__(self):
        super().__init__()
        self.model_encoder = EfficientNet_5_Encoder()
        self.model_decoder = FPNDecoderV2(encoder_channels=(2048, 176, 64, 40, 24),
                                          pyramid_channels=128,
                                          segmentation_channels=64,
                                          final_channels=4,
                                          dropout=0.2)


    def forward(self, x):
        global_features = self.model_encoder(x)
        seg_feature = self.model_decoder(global_features)

        return seg_feature


class SENet_Encoder(nn.Module):
    def __init__(self):
        super(SENet_Encoder, self).__init__()

        self.planes = [256 // 4, 512 // 4, 1024 // 4, 2048 // 4]
        senet = pretrainedmodels.__dict__["se_resnext50_32x4d"](num_classes=1000, pretrained='imagenet')#se_resnext50_32x4d(num_classes=1000, pretrained=None)
        self.encode1 = senet.layer0
        self.encode2 = senet.layer1
        self.encode3 = senet.layer2
        self.encode4 = senet.layer3
        self.encode5 = senet.layer4

        self.down1 = nn.Conv2d(256, self.planes[0], kernel_size=1)
        self.down2 = nn.Conv2d(512, self.planes[1], kernel_size=1)
        self.down3 = nn.Conv2d(1024, self.planes[2], kernel_size=1)
        self.down4 = nn.Conv2d(2048, self.planes[3], kernel_size=1)

    def forward(self, x):
        x1 = self.encode1(x)
        x2 = self.encode2(x1)
        x3 = self.encode3(x2)
        x4 = self.encode4(x3)
        x5 = self.encode5(x4)

        return [x5, x4, x3, x2, x1]

class SeResNet_FPN(nn.Module):

    def __init__(self):
        super().__init__()
        self.model_encoder = SENet_Encoder()
        self.model_decoder = FPNDecoder(encoder_channels=(2048, 1024, 512, 256, 64),
                                        pyramid_channels=256,
                                        segmentation_channels=64,
                                        final_channels=4,
                                        dropout=0.2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(2048, 2048, bias=True),
                                nn.Linear(2048, 1, bias=True))

    def forward(self, x):

        global_features = self.model_encoder(x)
        cls_feature = global_features[0]
        cls_feature = self.avgpool(cls_feature)
        cls_feature = cls_feature.view(cls_feature.size(0), -1)
        cls_feature = self.fc(cls_feature)
        seg_feature = self.model_decoder(global_features)

        return seg_feature, cls_feature


if __name__ == "__main__":
    pass
    #x = torch.ones((2, 3, 256, 400))
    #model = EfficientNet_b5_FPN()
    #output = model(x)
    #print(output.size())
