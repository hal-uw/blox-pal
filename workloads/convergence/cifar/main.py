from __future__ import print_function
import argparse
import timeit
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
import torch.distributed as dist
import numpy as np
import os
import sys
import torchvision
import time
from torchvision import transforms

sys.path.append(os.path.abspath(
    os.path.join(
        os.path.join(
            os.path.join(os.path.dirname(__file__), os.path.pardir),
            os.path.pardir),
        os.path.pardir)
))

from workloads.lucid.cifar.models import *

# Benchmark settings
parser = argparse.ArgumentParser(
    description="PyTorch DP Synthetic Benchmark", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--batch-size', type=int, default=32, help='batch size')
parser.add_argument('--num-epochs', type=int, default=100, help='num epochs')
parser.add_argument('--data_dir', type=str, default="data/", help='Data directory')

args = parser.parse_args()


def main():
    model_name = 'ResNet18'
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    print('==> Building model..')
    if model_name == 'VGG11':
        model = VGG('VGG11')
    elif model_name == 'VGG13':
        model = VGG('VGG13')
    elif model_name == 'VGG16':
        model = VGG('VGG16')
    elif model_name == 'VGG19':
        model = VGG('VGG19')
    elif model_name == 'ShuffleNetV2':
        model = ShuffleNetV2(net_size=0.5)
    else:
        model = eval(model_name)()

    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.0133 ** (1.0 / args.num_epochs))

    # specify dataset
    ###### dataloader
    # print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform_test
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size,
        shuffle=True,
        num_workers=4
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size,
        shuffle=False,
        num_workers=4
    )

    with open(f"../info/cifar10_{batch_size}.txt", "w"):
        pass

    # Train
    total_images = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in trainloader:
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total_images += inputs.size(0)

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        with open(f"../info/cifar10_{batch_size}.txt", "a") as file:
            file.write(f"Epoch: {epoch}, "
                       f"Total Images: {total_images}, "
                       f"Accuracy: {100 * correct / total}% \n")

        print(f'Accuracy on the CIFAR-10 test images: {100 * correct / total}%')
        lr_scheduler.step()


if __name__ == "__main__":
    main()
