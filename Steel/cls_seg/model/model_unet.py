import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import torch
import math
from torch.utils import model_zoo
import pretrainedmodels
from torchsummary import summary
#print(pretrainedmodels.model_names)
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import *
import segmentation_models_pytorch as smp
import torchvision
from segmentation_models_pytorch.unet.decoder import *
from segmentation_models_pytorch.common.blocks import *
from segmentation_models_pytorch.base.model import *

class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)


class FPA_V2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FPA_V2, self).__init__()
        """
        Feature Pyramid Attention module can 
        provide enough precise pixel-level prediction and class identification
        """

        self.glob = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False))

        self.down1_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=7, stride=2, padding=3, bias=False),
                                     nn.BatchNorm2d(input_dim),
                                     nn.ReLU(True))
        self.down1_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=7, padding=3, bias=False),
                                     nn.BatchNorm2d(output_dim),
                                     nn.ReLU(True))

        self.down2_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=5, stride=2, padding=2, bias=False),
                                     nn.BatchNorm2d(input_dim),
                                     nn.ReLU(True))
        self.down2_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=5, padding=2, bias=False),
                                     nn.BatchNorm2d(output_dim),
                                     nn.ReLU(True))

        self.down3_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(input_dim),
                                     nn.ReLU(True))
        self.down3_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(output_dim),
                                     nn.ReLU(True))

        self.conv1 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(output_dim),
                                   nn.ReLU(True))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def forward(self, x):

        x_glob = self.glob(x)
        x_glob = F.interpolate(x_glob, scale_factor=(x.size(2), x.size(3)))  # 512, 16, 25

        d1_1 = self.down1_1(x)
        d1_2 = self.down1_2(d1_1)

        d2_1 = self.down2_1(d1_1)
        d2_2 = self.down2_2(d2_1)

        d3_1 = self.down3_1(d2_1)
        d3_2 = self.down3_2(d3_1)

        d3 = F.interpolate(d3_2, size=(d2_2.size(2), d2_2.size(3)), mode='bilinear', align_corners=True)
        d2 = d2_2 + d3

        d2 = F.interpolate(d2, size=(d1_2.size(2), d1_2.size(3)), mode='bilinear', align_corners=True)
        d1 = d1_2 + d2

        d1 = F.interpolate(d1, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        x = self.conv1(x)
        x = x * d1

        x = x + x_glob

        return x


class FPA(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FPA, self).__init__()
        """
        Feature Pyramid Attention module can 
        provide enough precise pixel-level prediction and class identification
        """

        self.glob = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False))

        # self.down1_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=7, stride=2, padding=3, bias=False),
        # nn.BatchNorm2d(input_dim),
        # nn.ELU(True))
        # self.down1_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=7, padding=3, bias=False),
        # nn.BatchNorm2d(output_dim),
        # nn.ELU(True))

        self.down2_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=5, stride=2, padding=2, bias=False),
                                     nn.BatchNorm2d(input_dim),
                                     nn.ReLU(True))
        self.down2_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=5, padding=2, bias=False),
                                     nn.BatchNorm2d(output_dim),
                                     nn.ReLU(True))

        self.down3_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(input_dim),
                                     nn.ReLU(True))
        self.down3_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(output_dim),
                                     nn.ReLU(True))

        self.conv1 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(output_dim),
                                   nn.ReLU(True))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def forward(self, x):

        x_glob = self.glob(x)
        x_glob = F.interpolate(x_glob, scale_factor=(x.size(2), x.size(3)))  # 512, 16, 25

        # d1_1 = self.down1_1(x)
        # d1_2 = self.down1_2(d1_1)

        d2_1 = self.down2_1(x)
        d2_2 = self.down2_2(d2_1)

        d3_1 = self.down3_1(d2_1)
        d3_2 = self.down3_2(d3_1)

        # d3 = F.interpolate(d3_2, size=(d2_2.size(2), d2_2.size(3)), mode='bilinear', align_corners=True)
        # d2 = d2_2 + d3

        d2 = F.interpolate(d3_2, size=(d2_2.size(2), d2_2.size(3)))
        d1 = d2_2 + d2

        d1 = F.interpolate(d1, size=(x.size(2), x.size(3)))

        x = self.conv1(x)
        x = x * d1

        x = x + x_glob

        return x


class UpBlock(nn.Module):
    def __init__(self, in_c, mid_c, out_c):
        super(UpBlock, self).__init__()
        self.conv = nn.Sequential(
            SEModule(in_c, reduction=16),
            nn.Conv2d(in_c, mid_c, kernel_size=1),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(mid_c, mid_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(),
            nn.Conv2d(mid_c, out_c, kernel_size=1),
            nn.ReLU()
        )
        # todo add seblock!

    def forward(self, x):
        return self.conv(x)

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes=512, mid_c=256, dilations=[1, 6, 12, 18]):
        super(ASPP, self).__init__()
        self.aspp1 = _ASPPModule(inplanes, mid_c, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, mid_c, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, mid_c, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, mid_c, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, mid_c, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(mid_c),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(mid_c * 5, mid_c, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):

        # (512, 8, 50)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class PANet(nn.Module):
    def __init__(self, model_name='se_resnet50', num_class=4):
        super(PANet, self).__init__()

        senet = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        self.planes = [256 // 4, 512 // 4, 1024 // 4, 2048 // 4]
        inplanes = 64
        # replace the 7x7 convolutional layer in the original se_resnet by three 3×3
        # convolutional layers like PSP-Net and DUC.
        layer0_modules = [
            ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1, bias=False)),
            ('bn3', nn.BatchNorm2d(inplanes)),
            ('relu3', nn.ReLU(inplace=True)),
        ]
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True)))
        self.encode1 = nn.Sequential(OrderedDict(layer0_modules))
        self.encode2 = senet.layer1
        self.encode3 = senet.layer2
        self.encode4 = senet.layer3
        self.encode5 = senet.layer4

        self.down1 = nn.Conv2d(256, self.planes[0], kernel_size=1)
        self.down2 = nn.Conv2d(512, self.planes[1], kernel_size=1)
        self.down3 = nn.Conv2d(1024, self.planes[2], kernel_size=1)
        self.down4 = nn.Conv2d(2048, self.planes[3], kernel_size=1)

        self.center = FPA(self.planes[3], self.planes[2])
        #self.center = ASPP(self.planes[3], self.planes[2])
        #self.fc_op = nn.Sequential(
            #nn.Conv2d(self.planes[2], 64, kernel_size=1),
            #nn.AdaptiveAvgPool2d(1))

        #self.fc = nn.Linear(64, 1)
        self.UP4 = UpBlock(self.planes[3], 64, 64)
        self.UP3 = UpBlock(self.planes[2] + 64, 64, 64)
        self.UP2 = UpBlock(self.planes[1] + 64, 64, 64)
        self.UP1 = UpBlock(self.planes[0] + 64, 64, 64)
        self.final = nn.Sequential(
            nn.Conv2d(64 * 4, self.planes[0] // 2, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.planes[0] // 2),
            nn.Conv2d(self.planes[0] // 2, self.planes[0] // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.planes[0] // 2),
            nn.UpsamplingBilinear2d(size=(256, 1600)),
            nn.Conv2d(self.planes[0] // 2, num_class, kernel_size=1)
        )

    def forward(self, x):
        # x: (batch_size, 3, 256, 1600)
        # output:(batch_size, 4, 256, 320)

        e1 = self.encode1(x)  # 64, 64, 400
        e2 = self.encode2(e1)  # 256, 64, 400
        e3 = self.encode3(e2)  # 512， 32, 200
        e4 = self.encode4(e3)  # 1024, 16, 100
        e5 = self.encode5(e4)  # 2048, 8, 50

        e2 = self.down1(e2)  # 64, 64, 400
        e3 = self.down2(e3)  # 128, 32, 200
        e4 = self.down3(e4)  # 256, 16, 100
        e5 = self.down4(e5)  # 512, 8, 50

        #e5 = self.center(e5)

        d4 = self.UP4(e5)
        d3 = self.UP3(torch.cat([e4, d4], 1))
        d2 = self.UP2(torch.cat([e3, d3], 1))
        d1 = self.UP1(torch.cat([e2, d2], 1))
        h, w = d1.size()[2:]
        x = torch.cat(
            [
                F.upsample_bilinear(d4, size=(h, w)),
                F.upsample_bilinear(d3, size=(h, w)),
                F.upsample_bilinear(d2, size=(h, w)),
                d1
            ],
            1
        )

        return self.final(x)#, fc


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


class ResNet_unet(nn.Module):

    def __init__(self, num_class=5):
        super(ResNet_unet, self).__init__()
        self.model_encoder = ResNetEncoder(pretrained=True)
        self.model_decoder = UnetDecoder(encoder_channels=(256, 256, 128, 64, 64),
                                         decoder_channels=(256, 128, 64, 32, 16),
                                         final_channels=num_class,
                                         use_batchnorm=True,
                                         center=False,
                                         )
        self.center = FPA(512, 256)

    def forward(self, x):
        global_features = self.model_encoder(x)

        high_feature = global_features[0]
        high_feature = self.center(high_feature)
        global_features[0] = high_feature
        #cls_feature = self.avgpool(cls_feature)
        #cls_feature = cls_feature.view(cls_feature.size(0), -1)

        # cls_feature = self.fea_bn(cls_feature)
        #cls_feature = self.cls_head(cls_feature)
        seg_feature = self.model_decoder(global_features)

        return seg_feature#, cls_feature


class UnetDecoder_V2(Model):

    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=1,
            use_batchnorm=True,
            center=False,
            attention_type=None
        ):
        super().__init__()

        if center:
            channels = encoder_channels[0]
            self.center = FPA(channels, channels)#CenterBlock(channels, channels, use_batchnorm=use_batchnorm)
        else:
            self.center = None

        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        self.layer1 = DecoderBlock(in_channels[0], out_channels[0], use_batchnorm=use_batchnorm) #attention_type=attention_type)
        self.layer2 = DecoderBlock(in_channels[1], out_channels[1], use_batchnorm=use_batchnorm) #attention_type=attention_type)
        self.layer3 = DecoderBlock(in_channels[2], out_channels[2], use_batchnorm=use_batchnorm) #attention_type=attention_type)
        self.layer4 = DecoderBlock(in_channels[3], out_channels[3], use_batchnorm=use_batchnorm) #attention_type=attention_type)
        self.layer5 = DecoderBlock(in_channels[4], out_channels[4], use_batchnorm=use_batchnorm) #attention_type=attention_type)
        self.final_conv = nn.Sequential(
                          nn.Conv2d(out_channels[4]*4, out_channels[4] // 2, kernel_size=1),
                          nn.ReLU(),
                          nn.BatchNorm2d(out_channels[4] // 2),
                          nn.Conv2d(out_channels[4] // 2, out_channels[4] // 2, kernel_size=3, padding=1),
                          nn.ReLU(),
                          nn.BatchNorm2d(out_channels[4] // 2),
                          nn.UpsamplingBilinear2d(size=(256, 1600)),
                          nn.Conv2d(out_channels[4] // 2, final_channels, kernel_size=1)
        )

        self.initialize()

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
                encoder_channels[0] + encoder_channels[1],
                encoder_channels[2] + decoder_channels[0],
                encoder_channels[3] + decoder_channels[1],
                encoder_channels[4] + decoder_channels[2],
                0 + decoder_channels[3],
            ]
        return channels

    def forward(self, x):
        encoder_head = x[0]
        skips = x[1:]

        if self.center:
            encoder_head = self.center(encoder_head)

        x1 = self.layer1([encoder_head, skips[0]])
        x2 = self.layer2([x1, skips[1]])
        x3 = self.layer3([x2, skips[2]])
        x4 = self.layer4([x3, skips[3]])
        h, w = x4.size()[2:]
        x = torch.cat(
            [
                F.upsample_bilinear(x1, size=(h, w)),
                F.upsample_bilinear(x2, size=(h, w)),
                F.upsample_bilinear(x3, size=(h, w)),
                x4
                ],
                1
            )

        x = self.final_conv(x)

        return x

class ResNet_unet_v2(nn.Module):

    def __init__(self, num_class=5):
        super(ResNet_unet_v2, self).__init__()
        self.model_encoder = ResNetEncoder(pretrained=True)
        self.model_decoder = UnetDecoder_V2(encoder_channels=(512, 256, 128, 64, 64),
                                            decoder_channels=(64, 64, 64, 64, 64),
                                            final_channels=num_class,
                                            use_batchnorm=True,
                                            center=True)

    def forward(self, x):
        global_features = self.model_encoder(x)
        seg_feature = self.model_decoder(global_features)

        return seg_feature

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

class EfficientNet_b3_Unet(nn.Module):

    def __init__(self):
        super().__init__()
        self.model_encoder = EfficientNet_3_Encoder()
        self.model_decoder = UnetDecoder(encoder_channels=(1536, 136, 48, 32, 24),
                                         decoder_channels=(256, 128, 64, 32, 16),
                                         final_channels=4,
                                         use_batchnorm=True)
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

        return seg_feature

if __name__ == "__main__":
    pass
    #if 4 % 4:
        #print(1)

    #x = torch.ones((2, 3, 256, 1600))
    #model = ASPP()
    #model  = ResNet_unet()#PANet()#ResNet_unet_v2(num_class=5)
    #output = model(x)
    #print(output.size())
    #model = ResNet_unet_v2().cuda()
    #model = EfficientNet_3_unet().cuda()
    #seg = model(x)
    #print(seg.size())
    #summary(model=PANet().cuda(), input_size=(3, 256, 1600))
