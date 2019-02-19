import torch.nn as nn

class ResUnit(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1):

        """
        Residual Unit
        """
        super(ResUnit, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, int(outplanes/4), kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv2 = nn.Conv2d(int(outplanes/4), int(outplanes/4), kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(outplanes/4))
        self.conv3 = nn.Conv2d(int(outplanes/4), outplanes, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(int(outplanes/4))
        self.relu = nn.ReLU(inplace=True)
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.stride = stride
        self.make_downsample = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=1, stride=stride, bias=False)


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
