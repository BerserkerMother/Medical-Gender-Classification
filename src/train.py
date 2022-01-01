from argparse import ArgumentParser
from tqdm import tqdm
import logging
import wandb

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from data import MedicalDataset
from data import IXI_MEAN, IXI_STD
from model import SimNet
from utils import AverageMeter, set_seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    if args.seed:
        set_seed(args.seed)
    # wandb logging
    wandb.init(config=args, mode="disable")

    # dataset
    transform = transforms.Normalize(mean=IXI_MEAN, std=IXI_STD)
    train_set = MedicalDataset(args.data, split='train', transform=transform)
    val_set = MedicalDataset(args.data, split='val')
    test1_set = MedicalDataset(args.data, split='test1')
    test2_set = MedicalDataset(args.data, split='test2')
    test3_set = MedicalDataset(args.data, split='test3')

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    t1_loader = DataLoader(
        test1_set,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    t2_loader = DataLoader(
        test2_set,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    t3_loader = DataLoader(
        test3_set,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # model and optimizer
    model = SimNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=5e-4)

    for e in range(1, args.epochs):
        train_acc = train(train_loader, model, optimizer, e, args)
        val_acc = val(val_loader, model)

        # logging
        wandb.log({
            "accuracy": {
                "train": train_acc,
                "val": val_acc
            }
        }
        )
        logging.info('train accuracy: %.2f%%, val accuracy: %.2f%%' %
                     (train_acc, val_acc))
        torch.save(model.state_dict(), './model.pth')

    test(t1_loader, t2_loader, t3_loader, model)


def train(loader, model, optimizer, epoch, args):
    model.train()
    acc_meter = AverageMeter()
    total_loss = 0.
    for i, data in enumerate(loader):
        images, ages, targets = data
        images = images.to(device)
        ages = ages.to(device)
        targets = targets.to(device).view(-1, 1)

        batch_size = images.size()[0]

        output = model(images, ages)

        loss = F.binary_cross_entropy_with_logits(output, targets)
        total_loss += loss.item()

        # opt
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # acc
        pred = torch.where(output >= 0, 1., 0.)
        num_correct = (pred == targets).sum()
        acc_meter.update(num_correct, batch_size)

        if (i + 1) % args.log_freq == 0:
            logging.info('epoch [%3d/%3d][%4d/%4d], loss: %f' % (
                epoch, args.epochs, i, loader.__len__(),
                total_loss / args.log_freq))
            total_loss = 0.0

    return acc_meter.avg() * 100


def val(loader, model):
    model.eval()
    acc_meter = AverageMeter()
    for data in tqdm(loader):
        with torch.no_grad():
            images, ages, targets = data
            images = images.to(device)
            ages = ages.to(device)
            targets = targets.to(device).view(-1, 1)

            batch_size = images.size()[0]

            output = model(images, ages)

            # acc
            pred = torch.where(output >= 0, 1., 0.)
            num_correct = (pred == targets).sum()
            acc_meter.update(num_correct, batch_size)

    return acc_meter.avg() * 100


def test(t1_loader, t2_loader, t3_loader, model):
    model.eval()
    t1_meter, t2_meter, t3_meter = \
        AverageMeter(), AverageMeter(), AverageMeter()

    for data in tqdm(t1_loader):
        with torch.no_grad():
            images, ages, targets = data
            images = images.to(device)
            ages = ages.to(device)
            targets = targets.to(device).view(-1, 1)

            batch_size = images.size()[0]

            output = model(images, ages)

            # acc
            pred = torch.where(output >= 0, 1., 0.)
            num_correct = (pred == targets).sum()
            t1_meter.update(num_correct, batch_size)

    for data in tqdm(t2_loader):
        with torch.no_grad():
            images, ages, targets = data
            images = images.to(device)
            ages = ages.to(device)
            targets = targets.to(device).view(-1, 1)

            batch_size = images.size()[0]

            output = model(images, ages)

            # acc
            pred = torch.where(output >= 0, 1., 0.)
            num_correct = (pred == targets).sum()
            t2_meter.update(num_correct, batch_size)

    for data in tqdm(t3_loader):
        with torch.no_grad():
            images, ages, targets = data
            images = images.to(device)
            ages = ages.to(device)
            targets = targets.to(device).view(-1, 1)

            batch_size = images.size()[0]

            output = model(images, ages)

            # acc
            pred = torch.where(output >= 0, 1., 0.)
            num_correct = (pred == targets).sum()
            t3_meter.update(num_correct, batch_size)

    print("t1 acc: %.2f, t2 acc: %.2f, t3 acc: %.2f" %
          (t1_meter.avg(), t2_meter.avg(), t3_meter.avg()))


# data related
arg_parser = ArgumentParser(description='Medical Gender Classification')
arg_parser.add_argument('--data', type=str, default='', required=True,
                        help='path to data folder')
arg_parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size')
arg_parser.add_argument('--num_workers', default=2, type=int,
                        help='number of data loader workers')
# optimization related
arg_parser.add_argument('--lr', type=float, default=1e-4,
                        help='optimizer learning rate')
# training related
arg_parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs')
arg_parser.add_argument('--seed', type=int, help='number of training epochs')
arg_parser.add_argument('--log_freq', type=int, default=2,
                        help='frequency of logging')
arg = arg_parser.parse_args()
# config logger
logging.basicConfig(
    format='[%(levelname)s] %(module)s - %(message)s',
    level=logging.INFO
)
if __name__ == '__main__':
    main(arg)