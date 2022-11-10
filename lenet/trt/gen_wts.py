import torch
from torchvision.datasets import mnist
import torchvision.transforms as transforms
from LeNet import LeNet
import struct

model_name = 'lenet'


def model_load(model_path: str):
    print(f'[INFO]: Loading saved model...')
    net = torch.load(model_path)
    net = net.to('cuda:0')
    net.eval()
    return net


def model_gen_wts(model):
    """
    Convert weights to .wts format for TensorRT Engine
    Weights are written in the following format:
        <total-weights-count>
        weight.name <weight-count> <weight-val1> <weight-val2> ...
        -- total-weights-count: is an integer
        -- weight.name:         is used as key in TensorRT engine
        -- weight-count:        no. of weights for current layer
        -- weight-valxx:        float to c-bytes to hexadecimal
    :param model: pre-trained model
    :return:
    """
    print(f'[INFO]: Writing weights to .wts ...')
    with open(model_name + '.wts', 'w') as f:
        f.write(f'{len(model.state_dict().keys())}\n')
        for k, v in model.state_dict().items():
            vr = v.reshape(-1).cpu().numpy()
            f.write(f'{k} {len(vr)}')
            for vv in vr:
                f.write(" ")
                # convert weights to c-structs
                # Big-Endian (byte values) to Hex
                f.write(struct.pack('>f', float(vv)).hex())
            f.write('\n')

    with open(model_name + '.txt', 'w') as f:
        f.write(f'{len(model.state_dict().keys())}\n')
        for k, v in model.state_dict().items():
            vr = v.reshape(-1).cpu().numpy()
            f.write(f'{k} {len(vr)}')
            print(f'{k} {len(vr)}')
            f.write('\n')

    print('[INFO]: Successfully, converted weights to WTS ')


def model_test(model):
    test_set = mnist.MNIST('../../dataset', train=False, transform=transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor()]
    ), download=True)

    x, x_label = test_set[1]
    x = x.unsqueeze(0).cuda(0)  # 1x1x32x32
    b = x[0][0][15]
    print("[INFO]: input data label: ", x_label)
    y = model(x)
    _, y_label = y.max(-1)
    y_label = y_label.cpu().numpy()
    print("[INFO]: output data label: ", y_label)


def main():
    model = model_load('../' + model_name + '.pth')
    model_test(model)
    model_gen_wts(model)


if __name__ == "__main__":
    main()
