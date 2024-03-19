import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import json
from torchvision.models import resnet18
from models.diffusion import DiffusionModel
from models.diffusion_encoder import DDPMEncoder
from models.autoencoder import AutoEncoder


def sample_resnet(
    result_model_path='result_model.pt',
    autoencoder_path='autoencoder.pt',
    ddpm_enocoder_path='ddpm_encoder.pt',
    device='cuda',
):
    resnet = resnet18().to(device)
    resnet.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True).to(device)
    resnet.load_state_dict(torch.load(result_model_path, map_location=device))

    autoencoder = AutoEncoder(2048, 0, 0).to(device)
    autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=device))
    ddpm_encoder = DDPMEncoder(12, 1).to(device)
    ddpm_encoder.load_state_dict(torch.load(ddpm_enocoder_path, map_location=device))

    diffusion = DiffusionModel(eps_model=ddpm_encoder, betas=(1e-4, 2e-2), num_timesteps=1000)
    z = diffusion.sample(1, (4, 3), device)
    x = autoencoder.decode(z)[0]
    new_params = {
        'layer4.1.bn1.weight': x[0:512].detach(),
        'layer4.1.bn1.bias': x[512:1024].detach(),
        'layer4.1.bn2.bias': x[1024:1536].detach(),
        'layer4.1.bn2.weight': x[1536:2048].detach(),
    }
    for name, param in resnet.named_parameters():
        if name in new_params:
            param.data.copy_(new_params[name])

    return resnet


if __name__ == '__main__':
    model = sample_resnet()
