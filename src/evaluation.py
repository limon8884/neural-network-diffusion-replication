import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from models.diffusion import DiffusionModel
from models.diffusion_encoder import DDPMEncoder
from models.autoencoder import AutoEncoder
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from train_param_checkpoints import test_resnet
from train_autoencoder import LAYERS, ParamDataset
from sklearn.manifold import TSNE


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
        pathfile = 'param_checkpoints_layer3'
    elif layer_name == '14-16':
        pathfile = 'param_checkpoints_layer4.0'
    elif layer_name == '16-18':
        pathfile = 'param_checkpoints_layer4.1'
    pathlist = Path(pathfile).rglob('*.pt')
    accs = []
    for i, path in enumerate(pathlist):
        params_dict = torch.load(path, map_location=device)
        resnet = get_resnet_from_params(params_dict, device)
        acc = test_resnet(resnet)
        accs.append(acc)
        if i >= num_samples:
            break

    return np.max(accs), np.mean(accs), np.median(accs)


def eval_gen_model(layer_name, num_samples, device):
    accs = []
    params_list = create_new_params_for_resnet(layer_name, num_samples, device)
    for params_dict in params_list:
        resnet = get_resnet_from_params(params_dict, device)
        acc = test_resnet(resnet)
        accs.append(acc)

    return np.max(accs), np.mean(accs), np.median(accs)


def get_result_table(num_samples=32, device='cuda'):
    df = {}
    for layer_name in ['10-14', '14-16', '16-18']:
        df['orig ' + layer_name] = eval_origin_models(layer_name, num_samples, device=device)
        df['gen ' + layer_name] = eval_gen_model(layer_name, num_samples, device=device)
    data = pd.DataFrame.from_dict(df, orient='index')
    data.columns = ['max', 'mean', 'median']
    data.to_csv('results.csv')
    return data


def get_trajectories(layer_name, num_samples, device):
    trajectory_path = f'trajs_{layer_name}.pt'
    ddpm_enocoder_path = f'ddpm_encoder_{layer_name}.pt'
    ddpm_encoder = DDPMEncoder(12, 1).to(device)
    ddpm_encoder.load_state_dict(torch.load(ddpm_enocoder_path, map_location=device))

    diffusion = DiffusionModel(eps_model=ddpm_encoder, betas=(1e-4, 2e-2), num_timesteps=1000).to(device)
    diffusion.sample(num_samples, (4, 3), device, trajectory_path=trajectory_path)  # (bs, T, 4, 3)


def get_params_points(layer_name, num_samples, device):
    points_path = f'points_{layer_name}.pt'
    autoencoder_path = f'autoencoder_{layer_name}.pt'
    autoencoder = AutoEncoder(2048, 0, 0).to(device)
    autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=device))

    if layer_name == '10-14':
        pathfile = 'param_checkpoints_layer3'
    elif layer_name == '14-16':
        pathfile = 'param_checkpoints_layer4.0'
    elif layer_name == '16-18':
        pathfile = 'param_checkpoints_layer4.1'
    dataset = ParamDataset(pathfile, device, layer_name)
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=False)
    batch = next(iter(dataloader))
    z = autoencoder.encode(batch)
    torch.save(z, points_path)


def make_tsne_vectors(trajs_path, points_path):
    traj_vecs = torch.load(trajs_path, map_location='cpu')
    points_vecs = torch.load(points_path, map_location='cpu').unsqueeze(1)
    vecs = torch.cat([traj_vecs, points_vecs], dim=1)
    bs = vecs.size(0)
    t = vecs.size(1)
    vecs = vecs.reshape(bs * t, -1).detach().numpy()

    tsne = TSNE()
    new_vecs = tsne.fit_transform(vecs).reshape(bs, t, 2)
    return new_vecs


def plot_trajs_and_points(vecs, colors):
    assert len(vecs.shape) == 3
    assert len(colors) == len(vecs)
    for v, c in zip(vecs, colors):
        plt.scatter(v[:-1, 0], v[:-1, 1], color=c)
        plt.scatter(v[-1, 0], v[-1, 1], color='red')
    plt.show()
    plt.savefig('traj_fig.pdf')


def make_visualization(layer_name='16-18', num_samples=4, device='cuda', colors=['black', 'blue', 'green', 'orange']):
    trajectory_path = f'trajs_{layer_name}.pt'
    points_path = f'points_{layer_name}.pt'
    get_trajectories(layer_name, num_samples, device)
    get_params_points(layer_name, num_samples, device)
    vecs = make_tsne_vectors(trajectory_path, points_path)
    plot_trajs_and_points(vecs, colors)


if __name__ == '__main__':
    get_result_table(num_samples=8, device='cpu')
    make_visualization(device='cpu', num_samples=4)
