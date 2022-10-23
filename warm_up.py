import torch
import numpy as np
import matplotlib.pyplot as plt

# %% Tensor的转换和属性

np_array = np.random.randn(3, 5).astype(np.float32)  # ndarray更改类型
print(np_array)

# ndarray 转 tensor
tensor1 = torch.from_numpy(np_array)  # 默认与ndarray类型相同
tensor2 = torch.tensor(np_array)  # Tensor默认为FloatTensor
print(tensor1.type())

# 将Tensor放到gpu上
device = torch.device('cuda:0')
tensor2 = tensor2.cuda(device)
print(tensor2.type())

# cpu tensor 转 ndarray
np_ndarray1 = tensor1.numpy()
print(type(np_ndarray1))

# gpu tensor 转 ndarray 需要先将Tensor放到cpu上
np_ndarray2 = tensor2.cpu().numpy()

# Tensor 大小
print("Tensor size(): ", end='')  # 不换行
print(tensor1.size())
print("Tensor shape: ", end='')
print(tensor1.shape)

# Tensor维度, 3x5大小的Tensor维度为2
print("Tensor dim(): ", end='')
print(tensor1.dim())

# %% Tensor的创建
x = torch.ones(3, 5, dtype=torch.float32)  # 创建全1矩阵,创建时可指定数据类型
x = torch.zeros(4, 4, dtype=torch.int32)  # 创建全0矩阵
x = torch.randn(3, 5)  # 创建正态分布 N(0,1)
x = torch.rand(3, 5)  # 创建均匀分布 U(0,1)
x = torch.eye(3, 5)  # 创建单位阵，可以不为方阵，其它位置补零

x = torch.ones_like(x)  # 创建与输入参数形状相同的矩阵
x = torch.zeros_like(x)
x = torch.randn_like(x)
x = torch.rand_like(x)

# %% Tensor的操作
print('')
print("最大值 和 求和")
max_value, max_idx = torch.max(x, dim=1)  # dim=0求列的最大值 dim=1求行的最大值
print(x)
print(max_value)
print(max_idx)

sum_x = torch.sum(x, dim=1)  # dim=0按列求和 dim=1按行求和
print(sum_x)

# 增加维度,增加和减少的维度都只能是1维，本质上不改变数据，只改变size属性
print('')
print("增加减少维度")
print(x.shape)
x = x.unsqueeze(0)  # 在第0维增加
print(x.shape)
x = x.unsqueeze(1)  # 在第1维增加
print(x.shape)
x = x.squeeze(1)  # 减少第1维
print(x.shape)
x = x.squeeze(0)  # 减少所有1维
print(x.shape)

# %% 维度交换
print('')
print("维度交换")
x = torch.randn(3, 4, 5)
print(x.shape)
x = x.permute(2, 1, 0)  # 重新排列维度
print(x.shape)
x = x.transpose(0, 2)  # 交换维度
print(x.shape)

# 后面带有下划线_ 的函数一般为inplace操作
print('')
print("原地操作")
x.unsqueeze_(0)
print(x.shape)
x.squeeze_(0)
print(x.shape)

x = torch.ones(4, 4)
x[2:4, 0:2] = 2  # 切片访问时先行后列
print(x)

# %% 求导
print('')
print("求导")
x = torch.randn(3, 5, requires_grad=True)
y = torch.randn(3, 5, requires_grad=True)
z = torch.sum(x + y)  # 两个tensor相加后求和
print(z)  # grad_fn显示由相加得到
z.backward()
print(x.grad)
print(y.grad)

x = np.arange(-3, 3.01, 0.1)
y = x ** 2
plt.plot(x, y)
plt.plot(2, 4, 'ro')
plt.show()

x = torch.tensor([2], dtype=torch.float32, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)
