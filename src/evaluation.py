import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import json
from torchvision.models import resnet18
from models.diffusion import DiffusionModel
from models.diffusion_encoder import DDPMEncoder
from models.autoencoder import AutoEncoder


def sample_resnet(device):
    resnet = resnet18().to(device)
    resnet.load_state_dict(torch.load('resnet.pt', map_location=device))

    autoencoder = AutoEncoder(2048, 0, 0).to(device)
    autoencoder.load_state_dict(torch.load('autoencoder.pt', map_location=device))
    ddpm_encoder = DDPMEncoder(12, 1).to(device)
    ddpm_encoder.load_state_dict(torch.load('ddpm_encoder.pt', map_location=device))

    diffusion = DiffusionModel(eps_model=ddpm_encoder, betas=(1e-4, 2e-2), num_timesteps=1000)
    z = diffusion.sample(1, (4, 3), device)
    x = autoencoder.decode(z).detach()
    new_params = {
        'layer4.1.bn1.weight': x[0:512],
        'layer4.1.bn1.bias': x[512:1024],
        'layer4.1.bn2.bias': x[1024:1536],
        'layer4.1.bn2.weight': x[1356:2048],
    }
    for name, param in resnet.named_parameters():
        if name in new_params:
            param.copy_(new_params[name])

    return resnet
