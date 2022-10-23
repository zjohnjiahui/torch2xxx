import numpy as np
import torch
import torch.autograd
import matplotlib.pyplot as plt

torch.manual_seed(2022)

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

# weight
w = torch.randn(1, 1, requires_grad=True)
b = torch.randn(1, 1, requires_grad=True)


def linear_model(x):
    return x * w + b


def l2_loss(y, gt):
    return torch.mean((y - gt) ** 2)


y_ = linear_model(x_tensor)
plt.plot(x_tensor.data.numpy(), y_tensor.data.numpy(), 'bo', label='real')
plt.plot(x_tensor.data.numpy(), y_.data.numpy(), 'ro', label='estimated')
plt.title("before train")
plt.legend()
plt.show()


for i in range(10):
    y_ = linear_model(x_tensor)
    loss = l2_loss(y_, y_tensor)

    if i != 0:
        w.grad.zero_()
        b.grad.zero_()

    loss.backward()
    w.data = w.data - 1e-2 * w.grad.data  # w
    b.data = b.data - 1e-2 * b.grad.data  # b
    print('epoch: {}, loss: {}'.format(i, loss.data))


y_ = linear_model(x_tensor)
plt.plot(x_tensor.data.numpy(), y_tensor.data.numpy(), 'bo', label='real')
plt.plot(x_tensor.data.numpy(), y_.data.numpy(), 'ro', label='estimated')
plt.legend()
plt.title("after train")
plt.show()
