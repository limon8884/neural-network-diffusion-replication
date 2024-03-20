import torch
from torchvision.models import resnet18
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import wandb
import click

from models.diffusion import DiffusionModel
from models.diffusion_encoder import DDPMEncoder


class LatentParamDataset(Dataset):
    def __init__(self, filepath: str, device: str) -> None:
        super().__init__()
        self.latents = torch.load(filepath, map_location=device)

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, index):
        return self.latents[index]


def train_step(model: DiffusionModel, x: torch.Tensor, optimizer: Optimizer, device: str):
    optimizer.zero_grad()
    x = x.to(device)
    loss = model(x)
    loss.backward()
    optimizer.step()
    return loss


def train_epoch(model: DiffusionModel, dataloader: DataLoader, optimizer: Optimizer, device: str):
    model.train()
    loss_ema = None
    for x in dataloader:
        train_loss = train_step(model, x, optimizer, device)
        loss_ema = train_loss if loss_ema is None else 0.9 * loss_ema + 0.1 * train_loss

    return loss_ema


# def generate_samples(model: DiffusionModel, device: str, path: str):
#     model.eval()
#     with torch.no_grad():
#         samples = model.sample(8, (2048,), device=device)
#         return samples


def train_diffusion(layer_name, num_ecpochs=60000, device='cuda'):
    ddpm_encoder = DDPMEncoder(in_dim=12, in_channel=1)
    diff_model = DiffusionModel(eps_model=ddpm_encoder, betas=(1e-4, 2e-2), num_timesteps=1000).to(device)
    opt = torch.optim.AdamW(diff_model.parameters(), lr=1e-3, weight_decay=2e-6)
    dataset = LatentParamDataset(f'latent_dataset_{layer_name}.pt', device=device)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    for epoch in range(num_ecpochs):
        loss = train_epoch(diff_model, dataloader, opt, device)
        if (epoch + 1) % 10 == 0:
            wandb.log({'train loss': loss.item()})
            torch.save(ddpm_encoder.state_dict(), f'ddpm_encoder_{layer_name}.pt')


@click.command()
@click.argument('layer_name')
def main(layer_name):
    wandb.init(
        project='NDN-replication',
        name='ddpm_encoder-' + layer_name,
    )
    train_diffusion(layer_name)


if __name__ == '__main__':
    main()
