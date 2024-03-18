import torch
import json
from torchvision.models import resnet18


def save_params_example(filename):
    model = resnet18()
    list_of_params = []
    dict_scheme = {}
    position = 0
    for name, param in model.named_parameters():
        if 'bn' in name:
            # print(name, param.shape)
            dict_scheme[name] = (position, position + len(param))
            position += len(param)
            assert len(param.shape) == 1
            list_of_params.append(param)
    vec_to_save = torch.cat(list_of_params, dim=0)
    torch.save(vec_to_save, filename)
    return dict_scheme


if __name__ == '__main__':    
    scheme = save_params_example('param_checkpoints/test1.pt')
    scheme = save_params_example('param_checkpoints/test2.pt')
    with open('param_checkpoints/scheme.json', 'w') as f:
        json.dump(scheme, f)
