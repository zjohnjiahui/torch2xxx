from VGG import VGG
from load_pretrained import load_pretrained
import torch
import torch.nn as nn
from dataset import get_data
import trainer

torch.manual_seed(0)


def test():
    model = torch.load("models/model10.pth")
    criterion = nn.NLLLoss()
    _, eval_data = get_data(path='../dataset/flower', batch_size=16)
    trainer.evaluation(model, eval_data, criterion)


if __name__ == "__main__":
    test()
