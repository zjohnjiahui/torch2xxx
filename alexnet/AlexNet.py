import torch
import torch.nn as nn
from collections import OrderedDict

torch.manual_seed(0)


class AlexNet(nn.Module):
    """
    input: 3x224x224
    """
    def __init__(self, num_classes=10, init_weights=True):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2)),  # [48,55,55]
            ('bn', nn.BatchNorm2d(48)),
            ('relu', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2))  # [48,27,27]
        ]))

        self.conv2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(48, 128, kernel_size=5, padding=2)),  # [128,27,27]
            ('bn', nn.BatchNorm2d(128)),
            ('relu', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2))  # [128,13,13]
        ]))

        self.conv3 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(128, 192, kernel_size=5, padding=2)),  # [192,13,13]
            ('bn', nn.BatchNorm2d(192)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.conv4 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(192, 192, kernel_size=5, padding=2)),  # [192,13,13]
            ('bn', nn.BatchNorm2d(192)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.conv5 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(192, 128, kernel_size=5, padding=2)),  # [128,13,13]
            ('bn', nn.BatchNorm2d(128)),
            ('relu', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2))  # [128,6,6]
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('drop1', nn.Dropout(p=0.5)),
            ('fc1', nn.Linear(128*6*6, 2048)),
            ('relu1', nn.ReLU(inplace=True)),
            ('drop2', nn.Dropout(p=0.5)),
            ('fc2', nn.Linear(2048, 2048)),
            ('relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(2048, num_classes)),
            ('softmax', nn.Softmax(dim=-1))
        ]))

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


def test():
    net = AlexNet()
    x = torch.randn((1, 3, 224, 224))
    y = net(x)
    print(y.shape)
    print("test done")


if __name__ == "__main__":
    test()
