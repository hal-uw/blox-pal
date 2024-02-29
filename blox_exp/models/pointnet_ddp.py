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
from blox_enumerator import bloxEnumerate

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
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{args.master_ip_address}:{args.master_ip_port}",
        rank=args.rank,
        world_size=args.world_size
    )

    # specify dataset
    trainset = ShapeNetDataset(
        root=args.data_dir, classification=True, npoints=args.num_points
    )
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
        enumerator = bloxEnumerate(range(1000000), args.job_id)
        # Prevent total batch number < warmup+benchmark situation
        while True:
            for inputs, targets in trainloader:
                start = time.time()
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
                ictr, status = enumerator.__next__()
                logger.info(f"ictr {ictr} status {status}")
                enumerator.push_metrics(
                    {"attained_service": end - start,
                     "per_iter_time": end - start,
                     "iter_num": 1}
                )
                if status is False:
                    logger.info("Exit")
                    sys.exit()
                logger.info("Done iteration")

    print(f"==> Training {model_name} model with {batch_size} batchsize")
    benchmark_step(job_id)


if __name__ == "__main__":
    logging.basicConfig(filename=f"/dev/shm/training_worker_{args.job_id}_{args.rank}.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    print("start")
    model_name = "PointNet"
    batch_size = args.batch_size
    benchmark_pointnet(model_name, batch_size)
