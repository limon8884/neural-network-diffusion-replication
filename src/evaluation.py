import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from models.diffusion import DiffusionModel
from models.diffusion_encoder import DDPMEncoder
from models.autoencoder import AutoEncoder


def test_resnet(model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 64

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    correct_pred = 0
    total_pred = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred += 1
                total_pred += 1
    return correct_pred / total_pred


def sample_resnet(
    resnet,
    autoencoder_path='autoencoder.pt',
    ddpm_enocoder_path='ddpm_encoder.pt',
    device='cuda',
):
    autoencoder = AutoEncoder(2048, 0, 0).to(device)
    autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=device))
    ddpm_encoder = DDPMEncoder(12, 1).to(device)
    ddpm_encoder.load_state_dict(torch.load(ddpm_enocoder_path, map_location=device))

    diffusion = DiffusionModel(eps_model=ddpm_encoder, betas=(1e-4, 2e-2), num_timesteps=1000).to(device)
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
    device = 'cuda'
    resnet = resnet18().to(device)
    resnet.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True).to(device)
    resnet.load_state_dict(torch.load('result_model.pt', map_location=device))
    # model = sample_resnet()
    print('accuracy before:', test_resnet(resnet))
    new_resnet = sample_resnet(resnet, device=device)
    print('accuracy after:', test_resnet(new_resnet))
