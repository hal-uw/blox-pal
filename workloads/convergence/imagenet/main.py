from __future__ import print_function
import argparse
import torch
import torch.optim as optim
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.distributed as dist
import os
import sys


# Benchmark settings
parser = argparse.ArgumentParser(
    description="PyTorch DP Synthetic Benchmark", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--model-name', type=str, default="vgg16")
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--num-epochs', type=int, default=90)
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--local_rank', type=int)
args = parser.parse_args()


def main():
    model_name = args.model_name
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(local_rank)

    model = getattr(torchvision.models, model_name)()
    model = model.to(local_rank)
    torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank]
    )

    print('==> Preparing data..')
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load the ImageNet dataset
    train_dataset = ImageFolder(root='/users/Master/imagenet/train', transform=train_transform)
    val_dataset = ImageFolder(root='/users/Master/imagenet/val', transform=val_transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # Create the DataLoader for the training and validation datasets
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=False,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    criterion = nn.CrossEntropyLoss().to(local_rank)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    if args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    elif args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        raise ValueError("the opt is not implemented")

    if local_rank == 0:
        with open(f"../info/imagenet_{model_name}_{dist.get_world_size() * batch_size}_{args.opt}.txt", "w"):
            pass

    total_images = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(local_rank), targets.to(local_rank)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total_images += inputs.size(0)

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, local_rank: {local_rank}")

        if local_rank == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(local_rank), labels.to(local_rank)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            with open(f"../info/imagenet_{model_name}_{dist.get_world_size() * batch_size}_{args.opt}.txt", "a") as file:
                file.write(f"Epoch: {epoch}, "
                           f"Total Images: {(epoch + 1) * len(train_dataset)}, "
                           f"Accuracy: {100 * correct / total}% \n")

            print(f'Accuracy on the ImageNet validation images: {100 * correct / total}%')

        scheduler.step()
        torch.distributed.barrier()


if __name__ == '__main__':
    main()


