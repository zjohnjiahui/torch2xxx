import torch
from linear_regression_v2 import LinearRegression
import struct

model_name = 'linear_regression'  # change to other to export other model


def model_load(model_path=''):
    """
    Load saved model from file
    :param model_path: .pth file path
    :return net: loaded model
    """
    print(f'[INFO]: Loading saved model...')
    net = torch.load(model_path)
    net = net.to('cuda:0')
    net.eval()
    return net


def model_test(model, x):
    var = torch.tensor([[x]], dtype=torch.float32).to('cuda:0')
    print("\n[INFO]: test_model  \n\tinput: {}\n\toutput:{}".format(var, model(var).item()))


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
    print('[INFO]: Successfully, converted weights to WTS ')


def main():
    model = model_load(model_name + '.pth')
    model_test(model, 4)
    model_gen_wts(model)


if __name__ == '__main__':
    main()
