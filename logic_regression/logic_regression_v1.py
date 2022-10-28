import numpy as np
import torch
import torch.nn as nn
import torch.autograd
import matplotlib.pyplot as plt


torch.manual_seed(0)
# weights
# data is [100,2], weight is [2,1], data * weight = [1,1]
w = nn.Parameter(torch.randn(2, 1))
b = nn.Parameter(torch.randn(1, 1))


def logic_regression(x):
    # torch.mm 2-dimension matrix mul
    # torch.bmm batch mm
    # torch.matmul dot mul
    return torch.sigmoid(torch.mm(x, w) + b)


def binary_loss(y_pred, y):
    # use clamp to prevent infinity of log
    loss = -(y * y_pred.clamp(1e-12).log() + (1 - y) * (1 - y_pred).clamp(1e-12).log()).mean()
    return loss


def get_dataset():
    with open('logic_regression_data.txt', 'r') as f:
        # read lines returns list of str
        # split('\n')[0] discards '\n' of every line
        # split(',') get [x0,x1,y]
        data_list = [i.split('\n')[0].split(',') for i in f.readlines()]

        # convert to float
        data = [(float(i[0]), float(i[1]), float(i[2])) for i in data_list]

        # normalization
        x0_max = max(i[0] for i in data)
        x1_max = max(i[1] for i in data)
        data = [(i[0] / x0_max, i[1] / x1_max, i[2]) for i in data]

        return data


def plot_dataset(data):
    # filter data
    data_label0 = list(filter(lambda x: x[-1] == 0.0, data))  # select data with label == 0
    data_label1 = list(filter(lambda x: x[-1] == 1.0, data))  # select data with label == 1

    plot_x0 = [i[0] for i in data_label0]
    plot_y0 = [i[1] for i in data_label0]
    plot_x1 = [i[0] for i in data_label1]
    plot_y1 = [i[1] for i in data_label1]
    plt.plot(plot_x0, plot_y0, 'ro', label='data_0')
    plt.plot(plot_x1, plot_y1, 'bo', label='data_1')

    w0 = w[0].data[0]  # 使用data属性会去掉requires_grad
    w1 = w[1].data[0]
    b0 = b.data[0]
    plot_x = np.arange(0.2, 1, 0.01)
    plot_y = (-w0 * plot_x - b0) / w1
    plt.plot(plot_x, plot_y, 'g', label='cutting line')

    plt.legend(loc='best')
    plt.title('dataset')
    plt.show()


def plot_sigmoid():
    plot_x = np.arange(-10, 10.01, 0.01)
    plot_x = torch.tensor(plot_x, dtype=torch.float32)
    plot_y = torch.sigmoid(plot_x)
    plt.plot(plot_x.numpy(), plot_y.numpy(), 'r')
    plt.title('sigmoid')
    plt.show()


def main():
    data = get_dataset()
    plot_dataset(data)
    plot_sigmoid()

    np_data = np.array(data, dtype=np.float32)
    x_data = torch.from_numpy(np_data[:, 0:2])  # [100,2]
    y_data = torch.from_numpy(np_data[:, -1]).unsqueeze(1)  # [100,1]

    optim = torch.optim.AdamW([w, b], lr=0.01)
    for i in range(1000):
        y_pred = logic_regression(x_data)
        loss = binary_loss(y_pred, y_data)

        optim.zero_grad()
        loss.backward()
        optim.step()

        mask = y_pred.ge(0.5).float()
        acc = (mask == y_data).sum() / y_data.shape[0]
        if (i + 1) % 100 == 0:
            print('epoch: {}, Loss: {:.5f}, Acc: {:.5f}'.format(i + 1, loss, acc))

    plot_dataset(data)


if __name__ == "__main__":
    main()
