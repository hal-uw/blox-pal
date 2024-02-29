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
parser.add_argument("--amp-fp16", action="store_true", default=False, help="Enables FP16 training with Apex.")
parser.add_argument('--warmup_epoch', type=int, default=10, help='number of warmup epochs')
parser.add_argument('--benchmark_epoch', type=int, default=50, help='number of training benchmark epochs')
parser.add_argument('--data_dir', type=str, default="~/data/", help='Data directory')
parser.add_argument('--total_time', type=int, default=30, help='Total time to run the code')
parser.add_argument('--local_rank', type=int)
# parser.add_argument('--master_addr', type=str, default='127.0.0.1', help='Total time to run the code')
# parser.add_argument('--master_port', type=str, default='47020', help='Total time to run the code')

args = parser.parse_args()


def benchmark_cifar(model_name, batch_size, mixed_precision, t_start):
    cudnn.benchmark = True

    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(local_rank)

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

    model = model.to(local_rank)
    torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank]
    )

    criterion = nn.CrossEntropyLoss().to(local_rank)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    # specify dataset
    ###### dataloader
    # print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform_train
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=False
    )

    # Train
    def benchmark_step():
        iter_num = 0
        exit_flag = False
        model.train()
        # Prevent total batch number < warmup+benchmark situation
        while True:
            for inputs, targets in trainloader:
                # Warm-up: previous 10 iters
                if iter_num == args.warmup_epoch - 1:
                    t_warmend = time.time()
                # Reach timeout: exit profiling
                if time.time() - t_start >= args.total_time:
                    t_end = time.time()
                    t_pass = t_end - t_warmend
                    exit_flag = True
                    break
                optimizer.zero_grad()
                if mixed_precision:
                    inputs, targets = inputs.to(local_rank), targets.to(local_rank)
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    inputs, targets = inputs.to(local_rank), targets.to(local_rank)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                iter_num += 1
            if exit_flag:
                break
        return t_pass, iter_num

    print(f'==> Training {model_name} model with {batch_size} batchsize, {mixed_precision} mp..')
    t_pass, iter_num = benchmark_step()
    img_sec = dist.get_world_size() * (iter_num - args.warmup_epoch) * batch_size / t_pass
    if rank == 0:
        print(f'speed: {img_sec}')

    dist.destroy_process_group()


if __name__ == "__main__":
    # since this example shows a single process per GPU, the number of processes is simply replaced with the
    # number of GPUs available for training.
    model_name = 'ResNet18'
    batch_size = 64
    mixed_precision = 0
    t_start = time.time()
    benchmark_cifar(model_name, batch_size, mixed_precision, t_start)
    # print(bench_list[0] * dist.get_world_size())
