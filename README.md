# torch2xxx

trainng model on toy dataset using pytorch, export weight, deploy on inference framework like TensorRT etc.

pytorch -> weights -> inference framework



|     algorithm     | dataset |  description   | pre |
| :---------------: | :-----: | :------: |:------: |
| linear regression | points  | regression,direct train|[0,1]|
| logic regression  | points  | 2 class, direct train |[0,1]|
|        MLP        |  MNIST  | 10 class, direct train |[-1,1]|
|       LeNet       |  MNIST  | 10 class, direct train |[0,1]|
|      AlexNet      |  Flower | 5 class, direct train ||
|      VGG      |  Flower  | 5 class, transfer learning ||
