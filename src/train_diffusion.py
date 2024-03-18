import torch
from torchvision.models import resnet18


def make_dataset_from_checkpoints():
    filename = 'param_checkpoints/test.pt'
    d = torch.load(filename)


if __name__ == '__main__':
    make_dataset_from_checkpoints()
