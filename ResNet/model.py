import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, bias=False, dilation=dilation
        )

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,
        downsample=None, base_width=64, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,
        downsample=None, base_width=64, dilation=1):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.))
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride, dilation)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, pseudo=False, width_per_group=64):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        self.pseudo = pseudo
        self.base_width = width_per_group

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.conv2 = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(self.inplanes)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=False)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=False)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if self.pseudo:
            self.conv3 = nn.Conv2d(512*block.expansion, 16*block.expansion, kernel_size=1)
            self.bn3 = nn.BatchNorm1d(1296*block.expansion)
            self.fc = nn.Linear(1296*block.expansion, 512)
        else:
            self.fc = nn.Linear(512*block.expansion, 512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, base_width=self.base_width,
                dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.pseudo:
            x1 = self.avgpool(x)
            x2 = self.conv3(x)
            x2 = x2.reshape(x2.size(0), -1, 1, 1)

            x = torch.cat((x1, x2), 1)
            x = x.squeeze()
            x = self.bn3(x)
            x = self.fc(x)

        else:
            x = self.avgpool(x)
            x = x.reshape(x.size(0), -1)
            x = self.fc(x)

        return x

def ResNet18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def ResNet34(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def ResNet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def ResNet101(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model
