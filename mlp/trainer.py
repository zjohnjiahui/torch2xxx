import time
import torch
import torch.autograd
import torch.optim
import matplotlib.pyplot as plt
import numpy as np


def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(dim=1)  # 获取每行 max和 index
    num_correct = (pred_label == label).float().sum().data
    return num_correct / total


def train(net=None, train_data=None, eval_data=None,
          criterion=None, optimizer=None, num_epoch=None):
    print('[INFO]:Start training...')
    loss_list = []  # stores average loss of epochs
    acc_list = []  # stores average acc of epochs
    eval_loss_list = []
    eval_acc_list = []

    # try cuda
    if torch.cuda.is_available():
        net = net.cuda()

    for epoch in range(num_epoch):
        # train
        net.train()
        train_loss = 0  # loss in an epoch
        train_acc = 0  # acc in an epoch
        start = time.time()
        for img, label in train_data:
            if torch.cuda.is_available():
                img = img.cuda()  # (batch, 784)
                label = label.cuda()  # (batch, 10)
            # forward
            y_pred = net(img)
            y_pred = torch.log(y_pred)
            loss = criterion(y_pred, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # accumulate loss and acc
            train_loss += loss.data
            train_acc += get_acc(y_pred, label)
        # record average loss and acc
        loss_list.append(train_loss.cpu().numpy() / len(train_data))
        acc_list.append(train_acc.cpu().numpy() / len(train_data))
        print(f'{epoch} train time:{time.time() - start} s')

        # eval
        if eval_data is not None:
            eval_loss = 0
            eval_acc = 0
            net = net.eval()
            with torch.no_grad():
                for im, label in eval_data:
                    if torch.cuda.is_available():
                        im = im.cuda()
                        label = label.cuda()
                    output = net(im)
                    output = torch.log(output)
                    loss = criterion(output, label)
                    eval_loss += loss.data
                    eval_acc += get_acc(output, label)
            eval_loss_list.append(eval_loss.cpu().numpy() / len(eval_data))
            eval_acc_list.append(eval_acc.cpu().numpy() / len(eval_data))
            epoch_str = (
                    "Epoch %d. Train Loss: %f, Train Acc: %f, eval Loss: %f, eval Acc: %f, "
                    % (epoch, loss_list[epoch], acc_list[epoch],
                       eval_loss_list[epoch], eval_acc_list[epoch]))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, loss_list[epoch], acc_list[epoch]))
        print(epoch_str)

    print('[INFO]:Complete training...')
    x = range(num_epoch)
    plt.title('train loss line')
    plt.plot(x, loss_list)
    plt.show()

    plt.title('train accuracy line')
    plt.plot(x, acc_list)
    plt.show()

    plt.title('evaluation loss line')
    plt.plot(x, eval_loss_list)
    plt.show()

    plt.title('evaluation accuracy line')
    plt.plot(x, eval_acc_list)
    plt.show()

