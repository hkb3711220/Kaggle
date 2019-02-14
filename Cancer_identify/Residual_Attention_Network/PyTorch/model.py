import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable
from torchviz import make_dot

import sys
sys.path.append('./')

from ResUnit import ResUnit
from attentionlayer import AttentionModule_stage1, AttentionModule_stage2, AttentionModule_stage3


class ResidualAttentionNetwork(nn.Module):

    def __init__(self):
        super(ResidualAttentionNetwork, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64,
                                             kernel_size=3, stride=1, padding=1,bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.Resblock1 = ResUnit(64, 256)
        self.attention_module1 = AttentionModule_stage1(256, 256, size1=(48,48), size2=(24,24), size3=(12,12)) #(48,48)
        self.Resblock2 = ResUnit(256, 512, 2) #(24, 24)
        self.attention_module2 = AttentionModule_stage2(512, 512, size1=(24,24), size2=(12,12))
        self.Resblock3 = ResUnit(512, 1024, 2)
        self.attention_module3 = AttentionModule_stage3(1024, 1024, size1=(12, 12))
        self.Resblock4 = nn.Sequential(ResUnit(1024, 2048, 2),
                                       ResUnit(2048, 2048),
                                       ResUnit(2048, 2048))
        self.Avergepool = nn.Sequential(
                nn.BatchNorm2d(2048),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=6, stride=1)
            )

        self.fc = nn.Linear(2048, 1)
        self.pred = nn.Sigmoid()

    def forward(self, x):

        x = self.conv1(x) # (46,46)
        x = self.maxpool1(x) #
        x = self.Resblock1(x)
        x = self.attention_module1(x)
        x = self.Resblock2(x)
        x = self.attention_module2(x)
        x = self.Resblock3(x)
        x = self.attention_module3(x)
        x = self.Resblock4(x)
        x = self.Avergepool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        #x = self.pred(x)

        return x

model = ResidualAttentionNetwork()

x = Variable(torch.randn(30, 3, 96, 96))
y = model(x)
g = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
g.view()
