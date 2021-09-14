import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from argparse import ArgumentParser
from tqdm import tqdm

from data import MedicalDataset
from model import SimNet
from utils import AverageMeter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    train_set = MedicalDataset(args.data, split='train')
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
        train_set,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    t2_loader = DataLoader(
        train_set,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    t3_loader = DataLoader(
        train_set,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # model and optimizer
    model = SimNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    for e in range(1, args.epochs):
        train_acc = train(train_loader, model, optimizer, e, args)
        val_acc = val(val_loader, model)
        print('train accuracy: %.2f%%, val accuracy: %.2f%%' % (train_acc * 100, val_acc * 100))
        print('_' * 100)
        torch.save(model.state_dict(), './model.pth')


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
            print('epoch [%3d/%3d][%4d/%4d], loss: %f' % (
                epoch, args.epochs, i, loader.__len__(), total_loss / args.log_freq))
            total_loss = 0.0

    return acc_meter.avg()


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

    return acc_meter.avg()


# data related
arg_parser = ArgumentParser(description='Medical Gender Classification')
arg_parser.add_argument('--data', type=str, default='', required=True, help='path to data folder')
arg_parser.add_argument('--batch_size', default=32, type=int, help='batch size')
arg_parser.add_argument('--num_workers', default=2, type=int, help='number of data loader workers')
# optimization related
arg_parser.add_argument('--lr', type=float, default=1e-4, help='optimizer learning rate')
# training related
arg_parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
arg_parser.add_argument('--log_freq', type=int, default=2, help='frequency of logging')

arg = arg_parser.parse_args()
main(arg)
