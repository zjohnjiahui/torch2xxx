import torch
import torch.nn as nn
from collections import OrderedDict

torch.manual_seed(0)


class LeNet(nn.Module):
    """
    input: bx1x32x32
    """
    def __init__(self):
        super(LeNet, self).__init__()
        self.stage1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(1, 6, kernel_size=(5, 5))),  # (32-5)/1+1 = 28  bx6x28x28
            ('bn', nn.BatchNorm2d(6)),
            ('relu', nn.ReLU()),
            ('pool', nn.MaxPool2d(kernel_size=(2, 2), stride=2))  # 14 bx6x14x14
        ]))
        self.stage2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(6, 16, kernel_size=(5, 5))),  # 10 bx16x10x10
            ('bn', nn.BatchNorm2d(16)),
            ('relu', nn.ReLU()),
            ('pool', nn.MaxPool2d(kernel_size=(2, 2), stride=2))  # 5 bx16x5x5
        ]))
        self.stage3 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(16, 120, kernel_size=(5, 5))),  # 1 bx120x1x1
            ('relu', nn.ReLU())
        ]))
        self.fc1 = nn.Sequential(
            nn.Linear(120, 84),  # input bx120 output bx84
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(84, 10),   # bx10
            nn.Softmax(-1)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = x.squeeze()  # bx120

        x = self.fc1(x)
        x = self.fc2(x)
        return x


def test():
    net = LeNet()
    x = torch.randn([64, 1, 32, 32])
    print(x.size())
    y = net(x)
    print(y.size())


if __name__ == "__main__":
    test()

