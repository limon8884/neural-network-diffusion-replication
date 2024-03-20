import torch
import click
from torchvision.models import resnet18
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
import wandb

from models.autoencoder import AutoEncoder


LAYERS = {
    '10-14': [
        'layer3.0.bn1.weight', 'layer3.0.bn1.bias', 'layer3.0.bn2.weight', 'layer3.0.bn2.bias',
        'layer3.1.bn1.weight', 'layer3.1.bn1.bias', 'layer3.1.bn2.weight', 'layer3.1.bn2.bias',
    ],
    '14-16': ['layer4.0.bn1.weight', 'layer4.0.bn1.bias', 'layer4.0.bn2.weight', 'layer4.0.bn2.bias'],
    '16-18': ['layer4.1.bn1.weight', 'layer4.1.bn1.bias', 'layer4.1.bn2.weight', 'layer4.1.bn2.bias'],
}


class ParamDataset(Dataset):
    def __init__(self, filepath: str, device: str, layer_name: str) -> None:
        super().__init__()
        self.params = []
        pathlist = Path(filepath).rglob('*.pt')
        for path in pathlist:
            params_dict = torch.load(path, map_location=device)
            for ln in LAYERS[layer_name]:
                self.params.append(params_dict[ln])

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


def train(layer_name, checkpoints_path='param_checkpoints', num_ecpochs=30000, device='cuda'):
    if layer_name == '10-14':
        checkpoints_path += '_layer3'
    elif layer_name == '14-16':
        checkpoints_path += '_layer4.0'
    elif layer_name == '16-18':
        checkpoints_path += '_layer4.1'
    autoencoder = AutoEncoder(in_dim=2048, input_noise_factor=0.001, latent_noise_factor=0.1).to(device)
    opt = torch.optim.AdamW(autoencoder.parameters(), lr=1e-3, weight_decay=2e-6)
    dataset = ParamDataset(checkpoints_path, device, layer_name)
    train_size = int(0.8 * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    for epoch in range(num_ecpochs):
        train_loss = train_epoch(autoencoder, train_dataloader, opt, device)
        test_loss = eval(autoencoder, test_dataloader, device)
        if epoch % 100 == 0:
            wandb.log({'train loss:': train_loss.item(), 'test loss:': test_loss.item()})
            torch.save(autoencoder.state_dict(), f'autoencoder_{layer_name}.pt')

    x = torch.stack(dataset.params, dim=0)
    with torch.no_grad():
        z = autoencoder.encode(x)
    torch.save(z, f'latent_dataset_{layer_name}.pt')
    print(z.shape)


@click.command()
@click.argument('layer_name')
def main(layer_name):
    wandb.init(
        project='NDN-replication',
        name='autoencoder-' + layer_name,
    )
    train(layer_name=layer_name)


if __name__ == '__main__':
    main()
