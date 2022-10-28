import numpy as np
import torch
import torch.nn as nn
import torch.autograd
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)
scale = 8


class LogicRegression(nn.Module):
    """
    Logic Regression Model
    """
    def __init__(self):
        super(LogicRegression, self).__init__()
        self.linear = nn.Linear(2, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        return x


def get_dataset_file():
    """
    Get dataset from txt file
    :return: [100,2] FloatTensor x
    :return: [100,1] FloatTensor y_gt
    """
    print('[INFO]: Building dataset...')
    with open('logic_regression_data.txt', 'r') as f:
        # read lines returns list of str
        # split('\n')[0] discards '\n' of every line
        # split(',') get [x0,x1,y]
        data_list = [i.split('\n')[0].split(',') for i in f.readlines()]  # str list

        # convert to float
        data = [(float(i[0]), float(i[1]), float(i[2])) for i in data_list]  # float list

        # normalization
        x0_max = max(i[0] for i in data)
        x1_max = max(i[1] for i in data)
        data = [(i[0] / x0_max, i[1] / x1_max, i[2]) for i in data]  # normalized float list

        np_data = np.array(data, dtype=np.float32)
        x_data = torch.from_numpy(np_data[:, 0:2])  # [100,2] x
        y_data = torch.from_numpy(np_data[:, -1]).unsqueeze(1)  # [100,1] gt
        print('[INFO]: Building dataset done...')

        return x_data, y_data


def get_dataset_gen():
    """
    : Get dataset using generated data
    :return: [n,3] float data
    """
    print('[INFO]: Building dataset...')
    x0_data = torch.ones(50, 2) * 3 + torch.randn(50, 2)  # (3,3)
    y0_data = torch.zeros(50, 1)  # label = 0

    x1_data = torch.ones(50, 2) * 6 + torch.randn(50, 2)  # (6,6)
    y1_data = torch.ones(50, 1)  # label = 1

    # plot
    plot_x0 = x0_data[:, 0]
    plot_y0 = x0_data[:, 1]
    plot_x1 = x1_data[:, 0]
    plot_y1 = x1_data[:, 1]
    plt.plot(plot_x0, plot_y0, 'ro', label='data_0')
    plt.plot(plot_x1, plot_y1, 'bo', label='data_1')
    plt.show()

    # scale
    x_data = torch.cat([x0_data, x1_data], 0)
    x_data = x_data / scale
    y_data = torch.cat([y0_data, y1_data], 0)

    # shuffle
    index = [i for i in range(len(y_data))]
    np.random.shuffle(index)
    x_data = x_data[index]
    y_data = y_data[index]

    print('[INFO]: Building dataset done...')
    return x_data, y_data


def plot_line(model=None, x_data=None, y_data=None):
    """
    :param model: Logic Regression Model
    :param x_data: [n,2] Tensor pos
    :param y_data: [n,1] Tensor label
    :return:
    """
    data = torch.cat([x_data, y_data], 1)

    # filter data
    data_label0 = list(filter(lambda x: x[-1] == 0.0, data))  # select data with label == 0
    data_label1 = list(filter(lambda x: x[-1] == 1.0, data))  # select data with label == 1

    # plot point
    plot_x0 = [i[0] for i in data_label0]
    plot_y0 = [i[1] for i in data_label0]
    plot_x1 = [i[0] for i in data_label1]
    plot_y1 = [i[1] for i in data_label1]
    plt.plot(plot_x0, plot_y0, 'ro', label='data_0')
    plt.plot(plot_x1, plot_y1, 'bo', label='data_1')

    # plot line
    w0 = model.state_dict()['linear.weight'][0].data[0]
    w1 = model.state_dict()['linear.weight'][0].data[1]
    b0 = model.state_dict()['linear.bias'][0].data
    plot_x = np.arange(0.2, 1, 0.01)
    plot_y = (-w0 * plot_x - b0) / w1
    plt.plot(plot_x, plot_y, 'g', label='cutting line')
    plt.legend(loc='best')
    plt.show()


def train(model=None, x=None, gt=None,
          epoch=None, criterion=None, optimizer=None):
    print('[INFO]: Start training...')
    for e in range(epoch):
        y_pred = model(x)
        loss = criterion(y_pred, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mask = y_pred.ge(0.5).float()
        acc = (mask == gt).sum() / gt.shape[0]
        if e % 100 == 0:
            print('\tepoch: {}, Loss: {:.5f}, Acc: {:.5f}'.format(e, loss, acc))

    print('[INFO]: Complete training...')


def main():
    model = LogicRegression()
    x_data, y_data = get_dataset_gen()
    criterion = torch.nn.BCELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=0.01)
    train(model, x_data, y_data, 1000, criterion, optim)
    plot_line(model, x_data, y_data)
    # save the model
    torch.save(model, "logic_regression.pth")


if __name__ == "__main__":
    main()
