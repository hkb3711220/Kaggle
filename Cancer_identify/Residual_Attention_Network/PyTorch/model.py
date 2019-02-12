import torch
import torch.nn as nn
import torch.functional as F

class ResUnit(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1):

        """
        Residual Unit
        """
        super(ResidualUnit, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv2 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inplanes)
        self.conv3 = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.stride = stride
        self.make_downsample = nn.Sequential(nn.Conv2d(self.inplanes, self.outplanes, kernel_size=1,
                                             stride=stride, bias=False),
                                             nn.BatchNorm2d(self.outplanes),)

    def forward(self, x):

        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if (self.inplanes != self.outplanes) or (self.stride !=1 ):
            residual = self.make_downsample(residual)

        out += residual


        return out

class MaskBranch(nn.Module):

    def __init__(self, inplanes, outplanes, kernel_size):
        """
        max_pooling layers are used in mask branch size with input
        """
        super(MaskBranch, self).__init__()
        self.Resunit1 = ResUnit(inplanes, outplanes)
        #48 * 48
        self.maxpool1 = nn.MaxPool2d(kernel_size=kernel_size, stride=2, padding=1)
        self.Resunit2 = ResUnit(inplanes, outplanes)
        self.Resunit3 = ResUnit(inplanes, outplanes)
        self.maxpool2 = nn.MaxPool2d(kernel_size=kernel_size, stride=2, padding=1)
        self.Resblock3 = ResUnit(inplanes, outplanes)
        self.skip2 = ResUnit(inplanes, outplanes)
        self.maxpool3 = nn.MaxPool2d(kernel_size=kernel_size, stride=2, padding=1)
        self.Resblock4 = ResUnit(inplanes, outplanes)
        self.skip3 = ResUnit(inplanes, outplanes)
        self.maxpool4 = nn.MaxPool2d(kernel_size=kernel_size, stride=2, padding=1)
        self.Resblock5 = ResUnit(inplanes, outplanes)
        self.skip4 = ResUnit(inplanes, outplanes)

        #self.upsample = nn.UpsamplingBilinear2d(scale_factor=)


class trunkBranch(nn.Module):

    def __init__(self, )



class Residual_Attention_Net(nn.Module):

    def __init__(self):
        super(ResidualAttentionNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2,
                               padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)


    def forward(self, x):

        x = self.conv1(x)
        x = self.maxpool1(x)
