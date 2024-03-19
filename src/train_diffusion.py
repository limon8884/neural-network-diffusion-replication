import torch
from torchvision.models import resnet18
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

from models.diffusion import DiffusionModel
from models.diffusion_encoder import DiffusionEncoder


class ParamDataset(Dataset):
    def __init__(self, filepath: str) -> None:
        super().__init__()
        self.params = []
        pathlist = Path(filepath).rglob('*.pt')
        for path in pathlist:
            params_dict = torch.load(path)
            self.params.append(params_dict)

    def __len__(self):
        return len(self.params)

    def __getitem__(self, index):
        return self.params[index]


def train_step(model: DiffusionModel, x: torch.Tensor, optimizer: Optimizer, device: str):
    optimizer.zero_grad()
    x = x.to(device)
    loss = model(x)
    loss.backward()
    optimizer.step()
    return loss


def train_epoch(model: DiffusionModel, dataloader: DataLoader, optimizer: Optimizer, device: str):
    model.train()
    pbar = tqdm(dataloader)
    loss_ema = None
    for x, _ in pbar:
        train_loss = train_step(model, x, optimizer, device)
        loss_ema = train_loss if loss_ema is None else 0.9 * loss_ema + 0.1 * train_loss
        pbar.set_description(f"loss: {loss_ema:.4f}")

    return loss_ema


def generate_samples(model: DiffusionModel, device: str, path: str):
    model.eval()
    with torch.no_grad():
        samples = model.sample(8, (2048,), device=device)
        # grid = make_grid(samples, nrow=4)
        # save_image(grid, path)
        return samples


def train_diffusion(checkpoints_path, num_ecpochs=10000):
    encoder = DiffusionEncoder(in_dim=7808, input_noise_factor=0.001, latent_noise_factor=0.1)
    diff_model = DiffusionModel(eps_model=encoder, betas=(1e-4, 2e-2), num_timesteps=1000)
    opt = torch.optim.AdamW(diff_model.parameters(), lr=1e-3, weight_decay=2e-6)
    dataset = ParamDataset(checkpoints_path)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, pin_memory=True)
    for epoch in range(num_ecpochs):
        train_epoch(diff_model, dataloader, opt, 'cpu')


if __name__ == '__main__':
    train_diffusion('param_checkpoints')