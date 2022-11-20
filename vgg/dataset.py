import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader


def get_data(path='../dataset/flower', batch_size=32):
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                     ]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                   ])}

    flower_dataset_root = os.path.abspath(os.path.join(os.getcwd(), path))
    flower_train_path = os.path.join(flower_dataset_root, "train")
    flower_val_path = os.path.join(flower_dataset_root, "val")
    assert os.path.exists(flower_train_path), "{} path does not exist.".format(flower_train_path)
    assert os.path.exists(flower_val_path), "{} path does not exist.".format(flower_val_path)
    print("train dataset path: " + flower_train_path)
    print("val dataset path: " + flower_val_path)

    train_dataset = datasets.ImageFolder(flower_train_path, transform=data_transform["train"])
    val_dataset = datasets.ImageFolder(flower_val_path, transform=data_transform["val"])
    train_len = len(train_dataset)
    val_len = len(val_dataset)
    print("using {} images for training, {} images for validation.".format(train_len, val_len))

    worker = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(worker))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=worker)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=4, shuffle=False,
                                             num_workers=worker)
    return train_loader, val_loader


if __name__ == "__main__":
    get_data()
