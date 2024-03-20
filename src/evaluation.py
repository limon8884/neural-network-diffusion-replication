import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from models.diffusion import DiffusionModel
from models.diffusion_encoder import DDPMEncoder
from models.autoencoder import AutoEncoder
from pathlib import Path
import numpy as np

from train_param_checkpoints import test_resnet
from train_autoencoder import LAYERS


def create_new_params_for_resnet(layer_name, num_samples, device='cuda'):
    autoencoder_path = f'autoencoder_{layer_name}.pt'
    autoencoder = AutoEncoder(2048, 0, 0).to(device)
    autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=device))

    ddpm_enocoder_path = f'ddpm_encoder_{layer_name}.pt'
    ddpm_encoder = DDPMEncoder(12, 1).to(device)
    ddpm_encoder.load_state_dict(torch.load(ddpm_enocoder_path, map_location=device))

    diffusion = DiffusionModel(eps_model=ddpm_encoder, betas=(1e-4, 2e-2), num_timesteps=1000).to(device)
    z = diffusion.sample(num_samples, (4, 3), device)
    x = autoencoder.decode(z)

    new_params_list = []
    param_size = 256 if layer_name == '10-14' else 512
    for params in x:
        new_params = {}
        for j, ln in enumerate(LAYERS[layer_name]):
            new_params[ln] = params[param_size * j: param_size * (j + 1)]
        new_params_list.append(new_params)

    return new_params_list


def get_resnet_from_params(
    params_dict,
    device='cuda',
):
    resnet = resnet18().to(device)
    resnet.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True).to(device)
    resnet.load_state_dict(torch.load('result_model.pt', map_location=device))

    for name, param in resnet.named_parameters():
        if name in params_dict:
            param.data.copy_(params_dict[name])

    return resnet


def eval_origin_models(layer_name, num_samples, device):
    if layer_name == '10-14':
        pathfile = f'param_checkpoints_layer3'
    elif layer_name == '14-16':
        pathfile = f'param_checkpoints_layer4.0'
    elif layer_name == '16-18':
        pathfile = f'param_checkpoints_layer4.1'
    pathlist = Path(pathfile).rglob('*.pt')
    accs = []
    for i, path in enumerate(pathlist):
        params_dict = torch.load(path, map_location=device)
        resnet = get_resnet_from_params(params_dict, device)
        acc = test_resnet(resnet)
        accs.append(acc)
        if i >= num_samples:
            break

    return np.mean(accs), np.std(accs)


if __name__ == '__main__':
    # device = 'cuda'
    # resnet = resnet18().to(device)
    # resnet.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True).to(device)
    # resnet.load_state_dict(torch.load('result_model.pt', map_location=device))
    # # model = sample_resnet()
    # print('accuracy before:', test_resnet(resnet))
    # new_resnet = sample_resnet(resnet, device=device)
    # print('accuracy after:', test_resnet(new_resnet))