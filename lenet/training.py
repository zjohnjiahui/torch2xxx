from LeNet import LeNet
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
import trainer

torch.manual_seed(0)


def get_data():
    train_set = mnist.MNIST('../dataset', train=True, transform=transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor()]
    ), download=True)
    test_set = mnist.MNIST('../dataset', train=False, transform=transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor()]
    ), download=True)
    train_data = DataLoader(train_set, batch_size=64, shuffle=True)  # datasize [64, 1, 32, 32]
    test_data = DataLoader(test_set, batch_size=128, shuffle=False)  # datasize [128, 1, 32, 32]

    a, a_label = next(iter(train_data))
    print(a.size())
    b, b_label = next(iter(test_data))
    print(b.size())

    return train_data, test_data


def main():
    model = LeNet()
    print(model)
    criterion = nn.NLLLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    train_data, eval_data = get_data()

    trainer.train(model, train_data, eval_data, criterion, optim, 20)

    torch.save(model, "lenet.pth")


if __name__ == "__main__":
    main()
