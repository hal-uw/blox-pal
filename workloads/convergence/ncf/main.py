from __future__ import print_function
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

sys.path.append(os.path.abspath(
    os.path.join(
        os.path.join(
            os.path.join(os.path.dirname(__file__), os.path.pardir),
            os.path.pardir),
        os.path.pardir)
))

import workloads.lucid.ncf.evaluate as evaluate
import workloads.lucid.ncf.models as models
import workloads.lucid.ncf.config as config
import workloads.lucid.ncf.data_utils as data_utils

# Benchmark settings
parser = argparse.ArgumentParser(
    description="PyTorch DP Synthetic Benchmark", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--batch-size', type=int, default=64,
                    help='input batch size')
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate")
parser.add_argument("--dropout", type=float, default=0.0,
                    help="dropout rate")
parser.add_argument("--factor_num", type=int, default=32,
                    help="predictive factors numbers in the model")
parser.add_argument("--num_layers", type=int, default=3,
                    help="number of layers in MLP model")
parser.add_argument("--num_ng", type=int, default=4,
                    help="sample negative items for training")
parser.add_argument("--test_num_ng", type=int, default=99,
                    help="sample part of negative items for testing")
parser.add_argument("--num-epochs", type=int, default=10,
                    help="number of epochs")
parser.add_argument("--top_k", type=int, default=10,
                    help="compute metrics@top_k")
args = parser.parse_args()


def main():
    model_name = 'NeuMF-end'
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ############################## PREPARE DATASET ##########################
    train_data, test_data, user_num, item_num, train_mat = data_utils.load_all()

    # construct the train and test datasets
    train_dataset = data_utils.NCFData(
        train_data, item_num, train_mat, args.num_ng, True
    )
    test_dataset = data_utils.NCFData(
        test_data, item_num, train_mat, 0, False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_num_ng+1,
        shuffle=False,
        num_workers=0
    )

    ########################### CREATE MODEL #################################
    if model_name == 'NeuMF-pre':
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
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()

    if config.model == 'NeuMF-pre':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=args.lr)

    with open(f"../info/ncf_{batch_size}.txt", "w"):
        pass

    ########################### TRAINING #####################################
    total_items = 0
    for epoch in range(num_epochs):
        model.train()
        train_loader.dataset.ng_sample()
        for idx, (user, item, label) in enumerate(train_loader):
            user = user.to(device)
            item = item.to(device)
            label = label.float().to(device)
            optimizer.zero_grad()
            prediction = model(user, item)
            loss = criterion(prediction, label)
            loss.backward()
            optimizer.step()
            total_items += label.size(0)

        model.eval()
        HR, NDCG = evaluate.metrics(model, test_loader, args.top_k)
        print(f"Epoch: {epoch}, HR: {HR * 100}%, NDCG: {NDCG * 100}%")

        with open(f"../info/ncf_{batch_size}.txt", "a") as file:
            file.write(f"Epoch: {epoch}, "
                       f"Total Itemes: {total_items}, "
                       f"HR: {HR * 100}% "
                       f"NDCG: {NDCG * 100}% \n")


if __name__ == '__main__':
    main()
