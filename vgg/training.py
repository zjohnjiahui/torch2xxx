from VGG import VGG
from load_pretrained import load_pretrained
import torch
import torch.nn as nn
from dataset import get_data
import trainer

torch.manual_seed(0)


def main():
    model = load_pretrained()

    print(model)
    criterion = nn.NLLLoss()
    optim = torch.optim.AdamW(model.classifier.parameters(), lr=1e-5)
    train_data, eval_data = get_data(path='../dataset/flower', batch_size=16)
    trainer.train(model, train_data, eval_data, criterion, optim, 100, True)


if __name__ == "__main__":
    main()
