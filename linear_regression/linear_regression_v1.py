import numpy as np
import torch
import torch.autograd
import matplotlib.pyplot as plt

torch.manual_seed(0)

m0 = 2
m1 = 5

x_tensor = torch.from_numpy(np.arange(0, 10.0, 0.05)).unsqueeze(1)  # [num, 1]
y_tensor = m0 * x_tensor + m1 + torch.randn_like(x_tensor)

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


for i in range(1000):
    y_ = linear_model(x_tensor)
    loss = l2_loss(y_, y_tensor)

    if i != 0:
        w.grad.zero_()
        b.grad.zero_()

    loss.backward()
    w.data = w.data - 1e-2 * w.grad.data  # w
    b.data = b.data - 1e-2 * b.grad.data  # b

    if (i + 1) % 100 == 0:
        print('epoch: {}, loss: {}'.format(i, loss))

print('w: ', end='')
print(w)
print('b: ', end='')
print(b)

y_ = linear_model(x_tensor)
plt.plot(x_tensor.data.numpy(), y_tensor.data.numpy(), 'bo', label='real')
plt.plot(x_tensor.data.numpy(), y_.data.numpy(), 'ro', label='estimated')
plt.legend()
plt.title("after train")
plt.show()
