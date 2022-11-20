import os
import torch
import torch.nn as nn
from VGG import VGG


def load_pretrained():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load pretrain weights
    # download url: https://download.pytorch.org/models/vgg16-397923af.pth
    model_weight_path = "./vgg16-397923af.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

    model = VGG(name='vgg16', num_classes=5, init_weights=True)
    pre_weights = torch.load(model_weight_path, map_location=device)
    del_key = []
    for key, _ in pre_weights.items():
        if "classifier.6" in key:
            del_key.append(key)

    for key in del_key:
        del pre_weights[key]
    missing_keys, unexpected_keys = model.load_state_dict(pre_weights, strict=False)
    print("[missing_keys]:", *missing_keys, sep="\n")
    print("[unexpected_keys]:", *unexpected_keys, sep="\n")

    return model


if __name__ == '__main__':
    net = load_pretrained()
    print(net)
