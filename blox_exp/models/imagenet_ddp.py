from __future__ import print_function
import argparse
import timeit
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
import torch.distributed as dist
import numpy as np
import time
import os
import sys
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(
    os.path.join(
        os.path.join(os.path.dirname(__file__), os.path.pardir),
        os.path.pardir),
))

from torch.nn import DataParallel
from workloads.lucid.cifar.models import *

# Benchmark settings
parser = argparse.ArgumentParser(
    description="PyTorch DP Synthetic Benchmark", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--model-name', type=str, default="vgg16")
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--local_rank', type=int)
args = parser.parse_args()


# Training
def benchmark_imagenet(model_name, batch_size):
    cudnn.benchmark = True

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

    # data = torch.randn(batch_size, 3, 224, 224)
    # target = torch.LongTensor(batch_size).random_() % 1000
    # data, target = data.to(local_rank), target.to(local_rank)
    print('==> Preparing data..')
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load the ImageNet dataset
    train_dataset = ImageFolder(root='/global/cfs/cdirs/m4207/song/tiny-imagenet-200', transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # Create the DataLoader for the training and validation datasets
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=False,
        num_workers=2
    )

    criterion = nn.CrossEntropyLoss().to(local_rank)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    def benchmark_step():
        iter_num = 0
        while True:
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(local_rank), targets.to(local_rank)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                iter_num += 1

    # Benchmark
    print(f'==> Training {model_name} model with {batch_size} batchsize')
    benchmark_step()


if __name__ == '__main__':
    # the candidate model is vgg16 vgg19 resnet18 resnet50 shufflenet_v2_x1_0
    model_name = args.model_name
    batch_size = args.batch_size
    benchmark_imagenet(model_name, batch_size)
