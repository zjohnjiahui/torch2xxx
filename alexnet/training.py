from AlexNet import AlexNet
import torch
import torch.nn as nn
from dataset import get_data
import trainer

torch.manual_seed(0)


def main():
    model = AlexNet(num_classes=5, init_weights=True)
    print(model)
    # model.load_state_dict(torch.load("alexnet_checkpoint.pth"))
    criterion = nn.NLLLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    train_data, eval_data = get_data()

    trainer.train(model, train_data, eval_data, criterion, optim, 100, True)

    torch.save(model, "alexnet.pth")
    # torch.save(model.state_dict(), "alexnet_checkpoint_200.pth")


if __name__ == "__main__":
    main()
