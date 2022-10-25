import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(2022)


class LinearRegression(torch.nn.Module):
    """
    Linear Regression Model
    """
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        y = self.linear(x)
        return y


def get_dataset():
    print("[INFO]: Building dataset...")
    m0 = 2
    m1 = 5
    # y = 2x+5
    x_tensor = torch.tensor(np.arange(0, 10.0, 0.05), dtype=torch.float32).unsqueeze(1)  # [num, 1]
    y_tensor = m0 * x_tensor + m1 + torch.randn_like(x_tensor)

    return x_tensor, y_tensor


def train(model=None, x=None, y=None, epoch=None,
          criterion=None, optimizer=None):
    """
    Train the given model
    :param model: a pytorch model
    :param x: dataset
    :param y: labels
    :param epoch: train epoch
    :param criterion: loss function
    :param optimizer: optimization technique
    :return:
    """
    print("[INFO]: Starting training...\n")
    for epoch in range(epoch):
        # Forward and get loss
        pred_y = model(x)
        loss = criterion(pred_y, y)

        # Zero gradients, backward and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss every 50 epochs
        if epoch % 50 == 0:
            print('[INFO]: Epoch {}, Loss {}'.format(epoch, loss.item()))


def predict(model, x):
    """
    Check inference on the given model, currently with a static input
    :param model:
    :param x:
    :return:
    """
    var = torch.tensor([[x]], dtype=torch.float32)
    print("\n[INFO]: Predicted (after training) \n\tinput: {}\n\toutput:{}".format(var, model(var).item()))


def main():
    # initialize the model
    model = LinearRegression()

    # create the dataset
    x, y = get_dataset()

    # prepare configs
    criterion = torch.nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=0.01)

    # train the model
    train(model=model,
          x=x, y=y, epoch=5000,
          criterion=criterion, optimizer=optim)

    # check inference
    predict(model, 4)

    # save the model
    torch.save(model, "linear_regression.pth")


if __name__ == "__main__":
    main()
