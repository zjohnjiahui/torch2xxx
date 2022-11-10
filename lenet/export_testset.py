import torch
import os
import numpy as np
import torchvision
from skimage import io
import torchvision.transforms as transforms

# 获取数据集，变换为32x32大小
mnist_test = torchvision.datasets.MNIST(
    '../dataset', train=False, transform=transforms.Resize((32, 32)), download=True)
print('mnist_test length:', len(mnist_test))

# 创建输出目录
if not os.path.exists('./testset'):
    os.mkdir('./testset')
f = open("./testset/label.txt", 'w')

# 将前10张图片保存为jpg
for i, (img, label) in enumerate(mnist_test):
    if i == 10:
        break
    img_path = './testset/' + str(i) + '.jpg'
    img = np.array(img, dtype=np.uint8)
    io.imsave(img_path, img)
    f.write(str(label) + '\n')
f.close()
