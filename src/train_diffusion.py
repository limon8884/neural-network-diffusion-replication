import torch
from torchvision.models import resnet18
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

from models.diffusion import DiffusionModel
from models.diffusion_encoder import DDPMEncoder


class LatentParamDataset(Dataset):
    def __init__(self, filepath: str, device: str) -> None:
        super().__init__()
        self.latents = torch.load('latent.pt', map_location=device)

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


def generate_samples(model: DiffusionModel, device: str, path: str):
    model.eval()
    with torch.no_grad():
        samples = model.sample(8, (2048,), device=device)
        # grid = make_grid(samples, nrow=4)
        # save_image(grid, path)
        return samples


def train_diffusion(latent_params_path, num_ecpochs=10000, device='cuda'):
    ddpm_encoder = DDPMEncoder(in_dim=48, in_channel=1)
    diff_model = DiffusionModel(eps_model=ddpm_encoder, betas=(1e-4, 2e-2), num_timesteps=1000)
    opt = torch.optim.AdamW(diff_model.parameters(), lr=1e-3, weight_decay=2e-6)
    dataset = LatentParamDataset(latent_params_path, device=device)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True)
    for epoch in range(num_ecpochs):
        train_epoch(diff_model, dataloader, opt, device)
    torch.save(ddpm_encoder.state_dict(), 'ddpm_encoder.pt')


if __name__ == '__main__':
    train_diffusion('latent.pt', num_ecpochs=10, device='cpu')
