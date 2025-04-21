import torch.nn as nn

class TruncatedResNet(nn.Module):
    def __init__(self, whole_resnet):
        super(TruncatedResNet, self).__init__()
        self.conv1 = whole_resnet.conv1
        self.bn1 = whole_resnet.bn1
        self.relu = whole_resnet.relu
        self.maxpool = whole_resnet.maxpool

        self.layer1 = whole_resnet.layer1
        self.layer2 = whole_resnet.layer2
        self.layer3 = whole_resnet.layer3

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.flatten(2).transpose(1, 2)
        return x