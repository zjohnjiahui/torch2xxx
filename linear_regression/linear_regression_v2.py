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
    # x y train data
    x_np = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                     [9.779], [6.182], [7.59], [2.167], [7.042],
                     [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
    y_np = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                     [3.366], [2.596], [2.53], [1.221], [2.827],
                     [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
    # train data convert to Tensor
    x_tensor = torch.tensor(x_np)
    y_tensor = torch.tensor(y_np)
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
    optim = torch.optim.AdamW(model.parameters(), lr=0.001)

    # train the model
    train(model=model,
          x=x, y=y, epoch=500,
          criterion=criterion, optimizer=optim)

    # check inference
    predict(model, 4)

    # save the model
    torch.save(model, "linear_regression.pth")


if __name__ == "__main__":
    main()
