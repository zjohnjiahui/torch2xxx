import numpy as np
import torch
import torch.nn as nn
import torch.autograd
import torch.optim
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
import trainer

torch.manual_seed(0)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


def data_tf(x):
    """
    Data Preprocess
    :param x: input data
    :return:
    """
    x = np.array(x, dtype=np.float32) / 255  # norm [0,1]
    x = (x - 0.5) / 0.5  # center [-1,1]
    x = x.reshape((-1,))  # flatten [784,] 1-dim, it would be 2-dim in batch
    x = torch.from_numpy(x)
    return x


def get_data():
    train_set = mnist.MNIST('./data', train=True, transform=data_tf, download=True)
    test_set = mnist.MNIST('./data', train=False, transform=data_tf, download=True)
    train_data = DataLoader(train_set, batch_size=64, shuffle=True)  # datasize [64, 784]
    test_data = DataLoader(test_set, batch_size=128, shuffle=False)  # datasize [128, 784]
    return train_data, test_data


def main():
    model = MLP()
    print(model)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    train_data, eval_data = get_data()
    trainer.train(model, train_data, eval_data, criterion, optim, 20)

    torch.save(model, "mlp.pth")


if __name__ == "__main__":
    main()
    a = torch.empty(5, dtype=torch.long).random_(5)
