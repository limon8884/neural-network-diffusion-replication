import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import json
from torchvision.models import resnet18

EPOCHS = 50

def save_params_example(filename):
    model = resnet18()
    list_of_params = []
    dict_scheme = {}
    position = 0
    for name, param in model.named_parameters():
        if 'bn' in name and "layer4.1" in name:
            # print(name, param.shape)
            dict_scheme[name] = (position, position + len(param))
            position += len(param)
            assert len(param.shape) == 1
            list_of_params.append(param)
    vec_to_save = torch.cat(list_of_params, dim=0)
    torch.save(vec_to_save, filename)
    return dict_scheme

def save_params(model, filename, layer="layer4.1"):
    list_of_params = []
    dict_scheme = {}
    position = 0
    for name, param in model.named_parameters():
        if 'bn' in name and (layer in name or (layer == "layer3.1" and "layer3.0" in name)):
            # print(name, param.shape)
            dict_scheme[name] = (position, position + len(param))
            position += len(param)
            assert len(param.shape) == 1
            list_of_params.append(param)
    vec_to_save = torch.cat(list_of_params, dim=0)
    torch.save(vec_to_save, filename)
    return dict_scheme


def train_resnet():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 64

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    model = resnet18(pretrained=False).to(device)
    model.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    save_frec = len(trainloader) // 200
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0
            if epoch == EPOCHS - 1 and (i + 1) % save_frec == 0:
                for layer in ["layer3.1", "layer4.0", "layer4.1"]:
                    scheme = save_params(model.to("cpu"), f'param_checkpoints_{layer}/{i + 1}.pt', layer)
                    model.to(device)
                    with open(f'param_checkpoints_{layer}/scheme.json', 'w') as f:
                        json.dump(scheme, f)
    torch.save(model.state_dict(), "result_model.pt")

def test_resnet(model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
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

if __name__ == '__main__':
    train_resnet()
