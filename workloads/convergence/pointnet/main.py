from __future__ import print_function
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import os
import sys

sys.path.append(os.path.abspath(
    os.path.join(
        os.path.join(
            os.path.join(os.path.dirname(__file__), os.path.pardir),
            os.path.pardir),
        os.path.pardir)
))

from torchvision import transforms
from workloads.lucid.pointnet.dataset import ShapeNetDataset
from workloads.lucid.pointnet.pointnet import PointNetCls, feature_transform_regularizer

# Benchmark settings
parser = argparse.ArgumentParser(
    description="PyTorch Profile pointnet", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--feature_transform", action="store_true", help="use feature transform")
parser.add_argument("--num_points", type=int, default=2500, help="num of points for dataset")
parser.add_argument(
    '--data_dir',
    type=str,
    default="/users/Master/Megatron-Resource/workloads/lucid/pointnet/shapenetcore_partanno_segmentation_benchmark_v0/",
    help='Data directory'
)
parser.add_argument('--batch-size', type=int, default=64, help='batch_size')
parser.add_argument('--num-epochs', type=int, default=100, help='num_epochs')

args = parser.parse_args()


def main():
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # specify dataset
    print('==> Preparing data..')
    trainset = ShapeNetDataset(
        root=args.data_dir,
        classification=True,
        npoints=args.num_points,
        split='train'
    )
    testset = ShapeNetDataset(
        root=args.data_dir,
        classification=True,
        npoints=args.num_points,
        split='test'
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
    num_classes = len(trainset.classes)
    print("classes", num_classes)

    # Model
    print('==> Building model..')
    model = PointNetCls(k=num_classes, feature_transform=args.feature_transform)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    criterion = nn.CrossEntropyLoss().to(device)

    with open(f"../info/pointnet_{batch_size}.txt", "w"):
        pass

    total_images = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in trainloader:
            optimizer.zero_grad()
            targets = targets[:, 0]
            inputs = inputs.transpose(2, 1)
            inputs, targets = inputs.to(device), targets.to(device)
            pred, trans, trans_feat = model(inputs)
            loss = criterion(pred, targets)
            if args.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
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
                targets = targets[:, 0]
                inputs = inputs.transpose(2, 1)
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, trans, trans_feat = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        with open(f"../info/pointnet_{batch_size}.txt", "a") as file:
            file.write(f"Epoch: {epoch}, "
                       f"Total Images: {total_images}, "
                       f"Accuracy: {100 * correct / total}% \n")

        print(f'Accuracy on the ShapeNet validation images: {100 * correct / total}%')


if __name__ == '__main__':
    main()
