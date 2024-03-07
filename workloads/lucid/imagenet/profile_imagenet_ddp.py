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
        os.path.join(
            os.path.join(os.path.dirname(__file__), os.path.pardir),
            os.path.pardir),
        os.path.pardir)
))

from torch.nn import DataParallel
from workloads.lucid.cifar.models import *

# Benchmark settings
parser = argparse.ArgumentParser(
    description="PyTorch DP Synthetic Benchmark", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# parser.add_argument('--warmup_epoch', type=int, default=20, help='number of warmup epochs')
parser.add_argument(
    "--amp-fp16", action="store_true", default=False,
    help="Enables FP16 training with Apex."
)
parser.add_argument('--warmup_time', type=int, default=30, help='Warmup time to run the code')
parser.add_argument('--total_time', type=int, default=100, help='Total time to run the code')
parser.add_argument('--model-name', type=str, default="vgg16")
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--port', type=int, default=None, help='port for profiling')
args = parser.parse_args()


# Training
def benchmark_imagenet(model_name, batch_size, mixed_precision):
    t_start = time.time()
    cudnn.benchmark = True

    # for slurm profiling
    # if 'SLURM_PROCID' in os.environ:
    #     world_size = int(os.environ['SLURM_NTASKS'])
    #     rank = int(os.environ['SLURM_PROCID'])
    #     local_rank = int(os.environ['SLURM_LOCALID'])
    #     if args.port is not None:
    #         os.environ["MASTER_PORT"] = str(args.port)
    # else:
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
    # train_dataset = ImageFolder(root='/users/Master/imagenet/train', transform=transform)
    train_dataset = ImageFolder(root='/global/cfs/cdirs/m4207/song/tiny-imagenet-200/train', transform=transform)
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

    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    def benchmark_step():
        warmup_iter_num = -1
        iter_num = 0
        exist_flag = False
        while True:
            for inputs, targets in train_loader:
                # Warm-up: previous 10 iters
                if time.time() - t_start >= args.warmup_time and warmup_iter_num == -1:
                    t_warmend = time.time()
                    warmup_iter_num = iter_num
                # Reach timeout: exit profiling
                if time.time() - t_start >= args.total_time:
                    t_end = time.time()
                    t_pass = t_end - t_warmend
                    exist_flag = True
                    break
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
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                iter_num += 1
                # save and load checkpoint
                # if rank == 0:
                #     start_save = time.time()
                #     torch.save(
                #         {'model_state_dict': model.state_dict()},
                #         f'/pscratch/sd/s/songbian/checkpoint/{model_name}.pth.tar'
                #     )
                #     end_save = time.time()
                #     print(f"save checkpoint: {end_save - start_save}")
                #     start_load = time.time()
                #     checkpoint = torch.load(
                #         f'/pscratch/sd/s/songbian/checkpoint/{model_name}.pth.tar'
                #     )
                #     model.load_state_dict(checkpoint['model_state_dict'])
                #     end_load = time.time()
                #     print(f"load checkpoint: {end_load - start_load}")
            if exist_flag:
                break
        return t_pass, iter_num, warmup_iter_num

    # Benchmark
    print(f'==> Training {model_name} model with {batch_size} batchsize, {mixed_precision} mp..')
    t_pass, iter_num, warmup_iter_num = benchmark_step()
    iter_sec = dist.get_world_size() * (iter_num - warmup_iter_num) / t_pass
    if rank == 0:
        print(f'speed: {iter_sec}')

    dist.destroy_process_group()


if __name__ == '__main__':
    # the candidate model is vgg16 vgg19 resnet18 resnet50 shufflenet_v2_x1_0
    model_name = args.model_name
    batch_size = args.batch_size
    mixed_precision = 0
    benchmark_imagenet(model_name, batch_size, mixed_precision)

