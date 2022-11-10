import torch
import os
import numpy as np
import torchvision
from skimage import io

mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
print('mnist_test length:', len(mnist_test))

if not os.path.exists('./testset'):
    os.mkdir('./testset')
f = open("./testset/label.txt", 'w')

for i, (img, label) in enumerate(mnist_test):
    if i == 10:
        break
    img_path = './testset/' + str(i) + '.jpg'
    img = np.array(img, dtype=np.uint8)
    io.imsave(img_path, img)
    f.write(str(label) + '\n')
f.close()
