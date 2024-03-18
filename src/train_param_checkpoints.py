import torch
from torchvision.models import resnet18


def save_params_example(filename):
    model = resnet18()
    dict_to_save = {}
    for name, param in model.named_parameters():
        if 'bn' in name:
            dict_to_save[name] = param
    torch.save(dict_to_save, filename)


if __name__ == '__main__':
    save_params_example('param_checkpoints/test.pt')
