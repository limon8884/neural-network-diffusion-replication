import torch
from torchvision.models import resnet18
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
import wandb

from models.autoencoder import AutoEncoder


class ParamDataset(Dataset):
    def __init__(self, filepath: str, device: str) -> None:
        super().__init__()
        self.params = []
        pathlist = Path(filepath).rglob('*.pt')
        for path in pathlist:
            params_dict = torch.load(path, map_location=device)
            self.params.append(params_dict)

    def __len__(self):
        return len(self.params)

    def __getitem__(self, index):
        return self.params[index]


def train_step(model: AutoEncoder, x: torch.Tensor, optimizer: Optimizer, device: str):
    optimizer.zero_grad()
    x = x.to(device)
    y = model(x)
    loss = torch.nn.functional.mse_loss(y, x)
    loss.backward()
    optimizer.step()
    return loss


def train_epoch(model: AutoEncoder, dataloader: DataLoader, optimizer: Optimizer, device: str):
    model.train()
    # pbar = tqdm(dataloader)
    loss_ema = None
    for x in dataloader:
        train_loss = train_step(model, x, optimizer, device)
        loss_ema = train_loss if loss_ema is None else 0.9 * loss_ema + 0.1 * train_loss
        # pbar.set_description(f"loss: {loss_ema:.4f}")

    return loss_ema


def eval(model: AutoEncoder, dataloader: DataLoader, device: str):
    model.eval()
    loss_ema = None
    for x in dataloader:
        with torch.no_grad():
            x = x.to(device)
            test_loss = torch.nn.functional.mse_loss(x, model(x))
        loss_ema = test_loss if loss_ema is None else 0.9 * loss_ema + 0.1 * test_loss

    return loss_ema


def train(checkpoints_path='param_checkpoints', latent_dataset_path='latent_dataset.pt', num_ecpochs=30000,
          device='cuda'):
    autoencoder = AutoEncoder(in_dim=2048, input_noise_factor=0.001, latent_noise_factor=0.1).to(device)
    opt = torch.optim.AdamW(autoencoder.parameters(), lr=1e-3, weight_decay=2e-6)
    dataset = ParamDataset(checkpoints_path, device)
    train_size = int(0.8 * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=False, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False, pin_memory=True)
    for epoch in range(num_ecpochs):
        train_loss = train_epoch(autoencoder, train_dataloader, opt, device)
        test_loss = eval(autoencoder, test_dataloader, device)
        if epoch % 100 == 0:
            wandb.log({'train loss:': train_loss.item(), 'test loss:': test_loss.item()})
            torch.save(autoencoder.state_dict(), 'autoencoder.pt')

    x = torch.stack(dataset.params, dim=0)
    with torch.no_grad():
        z = autoencoder.encode(x)
    torch.save(z, latent_dataset_path)
    print(z.shape)


if __name__ == '__main__':
    wandb.init(
        project='NDN-replication',
        name='autoencoder',
    )
    train()
