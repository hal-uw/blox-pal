from __future__ import print_function
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torch.distributed as dist
import os
import sys
import pandas as pd
import time

sys.path.append(os.path.abspath(
    os.path.join(
        os.path.join(
            os.path.join(os.path.dirname(__file__), os.path.pardir),
            os.path.pardir),
        os.path.pardir)
))

from torchvision import transforms
from torch.nn import DataParallel
import workloads.lucid.ncf.models as models
import workloads.lucid.ncf.config as config
import workloads.lucid.ncf.data_utils as data_utils

# Benchmark settings
parser = argparse.ArgumentParser(
    description="PyTorch DP Synthetic Benchmark", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--batch-size', type=int, default=64, help='input batch size')
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
parser.add_argument("--factor_num", type=int, default=32, help="predictive factors numbers in the model")
parser.add_argument("--num_layers", type=int, default=3, help="number of layers in MLP model")
parser.add_argument("--num_ng", type=int, default=4, help="sample negative items for training")
parser.add_argument('--warmup_time', type=int, default=30, help='Warmup time to run the code')
parser.add_argument('--total_time', type=int, default=100, help='Total time to run the code')
parser.add_argument('--local_rank', type=int)
args = parser.parse_args()


def benchmark_ncf(model_name, batch_size, mixed_precision):
    t_start = time.time()
    cudnn.benchmark = True

    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(local_rank)

    ############################## PREPARE DATASET ##########################
    train_data, test_data, user_num, item_num, train_mat = data_utils.load_all()

    # construct the train and test datasets
    train_dataset = data_utils.NCFData(
        train_data, item_num, train_mat, args.num_ng, True
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=False,
        num_workers=2
    )

    ########################### CREATE MODEL #################################
    if model_name == 'NeuMF-end':
        assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
        assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
        GMF_model = torch.load(config.GMF_model_path)
        MLP_model = torch.load(config.MLP_model_path)
    else:
        GMF_model = None
        MLP_model = None

    model = models.NCF(
        user_num, item_num, args.factor_num, args.num_layers, args.dropout, config.model, GMF_model, MLP_model
    )
    model = model.to(local_rank)
    torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank]
    )

    criterion = nn.BCEWithLogitsLoss()

    if config.model == 'NeuMF-pre':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    ########################### TRAINING #####################################
    def benchmark_step():
        warmup_iter_num = -1
        iter_num = 0
        exit_flag = False
        model.train()
        train_loader.dataset.ng_sample()
        while True:
            for idx, (user, item, label) in enumerate(train_loader):
                # Warm-up: previous 10 iters
                if time.time() - t_start >= args.warmup_time and warmup_iter_num == -1:
                    t_warmed = time.time()
                    warmup_iter_num = iter_num
                # # Reach timeout: exit profiling
                if time.time() - t_start >= args.total_time:
                    t_end = time.time()
                    t_pass = t_end - t_warmed
                    exit_flag = True
                    break
                user = user.to(local_rank)
                item = item.to(local_rank)
                label = label.float().to(local_rank)
                optimizer.zero_grad()
                if mixed_precision:
                    with torch.cuda.amp.autocast():
                        prediction = model(user, item)
                        loss = criterion(prediction, label)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    prediction = model(user, item)
                    loss = criterion(prediction, label)
                    loss.backward()
                    optimizer.step()
                iter_num += 1
            if exit_flag:
                break
        return t_pass, iter_num, warmup_iter_num

    print(f'==> Training {model_name} model with {batch_size} batchsize, {mixed_precision} mp..')
    t_pass, iter_num, warmup_iter_num = benchmark_step()
    img_sec = dist.get_world_size() * (iter_num - warmup_iter_num) * batch_size / t_pass
    if rank == 0:
        print(f'speed: {img_sec}')

    dist.destroy_process_group()


if __name__ == '__main__':
    model_name = 'NeuMF-pre'
    batch_size = args.batch_size
    mixed_precision = 0
    benchmark_ncf(model_name, batch_size, mixed_precision)
