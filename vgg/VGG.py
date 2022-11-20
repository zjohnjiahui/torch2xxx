import torch
import torch.nn as nn

torch.manual_seed(0)

# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}


class VGG(nn.Module):
    def __init__(self, name='vgg11', num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        # 224->112->56->28->14->7 7x7x512
        self.cfgs = {
            'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
                      512, 512, 512, 'M', 512, 512, 512, 'M'],
            'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
                      512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        self.features = self.make_feature(model_name=name)
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=-1)
        )
        if init_weights:
            self._initialize_weights()

    def make_feature(self, model_name):
        assert model_name in self.cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
        layers = []
        in_channels = 3
        cfg = self.cfgs[model_name]
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                # bn = nn.BatchNorm2d(v)
                layers += [conv2d, nn.ReLU(True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def test():
    net = VGG(name='vgg16', num_classes=5, init_weights=True)
    print(net)
    x = torch.randn((1, 3, 224, 224))
    y = net(x)
    print(y.shape)
    print("test done")


if __name__ == "__main__":
    test()
