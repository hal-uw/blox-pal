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
import logging
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.join(os.path.dirname(__file__), os.path.pardir), os.path.pardir
        ),
    )
)

def setup_logging(job_id, rank, time):
    log_file = f'/scratch1/08503/rnjain/blox-pal/logs/profiler/imagenet_profile_{job_id}_{rank}_{time}.log'
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from workloads.lucid.cifar.models import *
# Benchmark settings
parser = argparse.ArgumentParser(
    description="PyTorch DP Synthetic Benchmark",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--model-name", type=str, default="vgg16")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--rank", type=int)
parser.add_argument("--world-size", type=int)
parser.add_argument("--master-ip-address", type=str)
parser.add_argument("--master-ip-port", type=str)
parser.add_argument("--job-id", type=int, default=0, help="job-id for blox scheduler")
args = parser.parse_args()
print("Parsed Args {}".format(args))


# Training
def benchmark_imagenet(model_name, batch_size):
    cudnn.benchmark = True
    job_id = args.job_id

    # initialize the process group
    logging.info(f"before init_process_group")
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{args.master_ip_address}:{args.master_ip_port}",
        rank=args.rank,
        world_size=args.world_size
    )
    logging.info(f"after init_process_group")

    model = getattr(torchvision.models, model_name)()
    model = model.cuda()
    torch.nn.parallel.DistributedDataParallel(model)

    print("==> Preparing data..")
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load the ImageNet dataset
    logging.info(f"before ImageFolder")
    train_dataset = ImageFolder(
        root="/scratch1/08503/rnjain/data-files/imagenet/ILSVRC2012_img_train", transform=transform
    )
    logging.info(f"after ImageFolder")
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # Create the DataLoader for the training and validation datasets
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=False,
        num_workers=2,
    )

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    def benchmark_step(job_id):
        iter_num = 0
        total_attained_service = 0
        while True:
            start = time.time()
            logging.info(f"Setup time end = {start}")
            for inputs, targets in train_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                end = time.time()
                iter_num += 1
                total_attained_service += end - start
                logging.info(f"attained_service {total_attained_service} iter_num {iter_num}")
                per_iter_time = end - start 
                logging.info(f"per_iter_time = {per_iter_time}")
                start = time.time()
                if iter_num >= 250:
                    logging.info("Job exit notify")
                    torch.cuda.empty_cache()
                    sys.exit()
                logging.info("Done iteration")

    # Benchmark
    print(f"==> Training {model_name} model with {batch_size} batchsize")
    logging.info(f"==> Training {model_name} model with {batch_size} batchsize")
    benchmark_step(job_id)


if __name__ == "__main__":
    # the candidate model is vgg16 vgg19 resnet18 resnet50 shufflenet_v2_x1_0
    warmup_time_start = time.time()
    setup_logging(args.job_id, args.rank, warmup_time_start)
    logging.info("Setup log file, start")
    logging.info(f"Setup time start = {warmup_time_start}") 
    model_name = args.model_name
    batch_size = args.batch_size
    benchmark_imagenet(model_name, batch_size)
