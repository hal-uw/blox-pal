from __future__ import print_function
import argparse
import time
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import numpy as np
import os
import sys
import pandas as pd
import torchvision
import time
import logging

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.join(os.path.dirname(__file__), os.path.pardir), os.path.pardir
        )
    )
)

from torch.nn import DataParallel
from torchvision import transforms
from workloads.lucid.pointnet.dataset import ShapeNetDataset
from workloads.lucid.pointnet.pointnet import PointNetCls, feature_transform_regularizer

def setup_logging(job_id, rank):
    log_file = f'/scratch1/08503/rnjain/blox-pal/logs/job-runs/training_worker_{job_id}_{rank}.log'
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Benchmark settings
parser = argparse.ArgumentParser(
    description="PyTorch Profile pointnet",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--amp-fp16",
    action="store_true",
    default=False,
    help="Enables FP16 training with Apex.",
)
parser.add_argument(
    "--feature_transform", action="store_true", help="use feature transform"
)
parser.add_argument(
    "--num_points", type=int, default=2500, help="num of points for dataset"
)
parser.add_argument(
    "--data_dir",
    type=str,
    default="/scratch1/08503/rnjain/data-files/pointnet/shapenetcore_partanno_segmentation_benchmark_v0/",
    help="Data directory",
)
parser.add_argument("--batch-size", type=int, default=64, help="batch size")
parser.add_argument("--rank", type=int)
parser.add_argument("--world-size", type=int)
parser.add_argument("--master-ip-address", type=str)
parser.add_argument("--master-ip-port", type=str)
parser.add_argument("--job-id", type=int, default=0, help="job-id for blox scheduler")

args = parser.parse_args()
print("Parsed Args {}".format(args))


def benchmark_pointnet(model_name, batch_size):
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

    # specify dataset
    logging.info(f"before ShapeNetDataset")
    trainset = ShapeNetDataset(
        root=args.data_dir, classification=True, npoints=args.num_points
    )
    logging.info(f"after ShapeNetDataset")
    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size,
        shuffle=(trainsampler is None),
        sampler=trainsampler,
        num_workers=2,
        pin_memory=False,
    )
    num_classes = len(trainset.classes)

    # Model
    model = PointNetCls(k=num_classes, feature_transform=args.feature_transform)
    model = model.cuda()
    torch.nn.parallel.DistributedDataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    criterion = nn.CrossEntropyLoss().cuda()

    # Train
    def benchmark_step(job_id):
        iter_num = 0
        # Prevent total batch number < warmup+benchmark situation
        total_attained_service = 0
        start = time.time()
        while True:
            for inputs, targets in trainloader:
                optimizer.zero_grad()
                targets = targets[:, 0]
                inputs = inputs.transpose(2, 1)
                inputs, targets = inputs.cuda(), targets.cuda()
                pred, trans, trans_feat = model(inputs)
                loss = criterion(pred, targets)
                if args.feature_transform:
                    loss += feature_transform_regularizer(trans_feat) * 0.001
                loss.backward()
                optimizer.step()
                end = time.time()
                iter_num += 1
                print(iter_num)
                total_attained_service += end - start
                start = time.time()
                if iter_num > 5:
                    logging.info("Job Exit Notify")
                    logging.info("Exit")
                    torch.cuda.empty_cache()
                    sys.exit()
                logging.info("Done iteration")

    print(f"==> Training {model_name} model with {batch_size} batchsize")
    logging.info(f"==> Training {model_name} model with {batch_size} batchsize")
    benchmark_step(job_id)


if __name__ == "__main__":
    setup_logging(args.job_id, args.rank)
    print("start")
    model_name = "PointNet"
    batch_size = args.batch_size
    benchmark_pointnet(model_name, batch_size)
