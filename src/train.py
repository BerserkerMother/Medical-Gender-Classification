from argparse import ArgumentParser
from tqdm import tqdm
import logging
import wandb

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.cuda import amp

import pandas as pd
from pandas import ExcelWriter

from data import MedicalDataset, transforms
from model import SimNet, SimNetExtra
from utils import AverageMeter, set_seed, log_and_display, plot_hist, plot_target_distri, sampling_ratio
from schedular import CosineSchedularLinearWarmup


def main(args):
    if args.seed:
        set_seed(args.seed)
    # wandb logging
    wandb.init(entity="berserkermother", project="MGC", config=args,
               name=args.name)

    # dataset
    if args.mix_split:
        dataset = MedicalDataset(args.data, splits='train+val', ram=args.ram)
        data_length = len(dataset)
        train_length = int(data_length * 0.7)  # uses 70% for training
        train_set, val_set = random_split(
            dataset, lengths=[train_length, data_length - train_length])
    else:
        train_set = MedicalDataset(args.data, splits='train', ram=args.ram)
        val_set = MedicalDataset(args.data, splits='val', ram=args.ram)
    test1_set = MedicalDataset(args.data, splits='test1')
    test2_set = MedicalDataset(args.data, splits='test2')
    test3_set = MedicalDataset(args.data, splits='test3')
    datasets_targets = []
    for ds in (train_set, val_set, test1_set, test2_set, test3_set):
        datasets_targets.append([i[-2] for i in ds])
    plot_target_distri(datasets_targets, args.name)
    loss_sampling_weight = sampling_ratio(train_set)

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
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
    if args.model == 1:
        model = SimNet(dropout=args.dropout).to(device)
    else:
        model = SimNetExtra(dropout=args.dropout,
                            model_num=args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    if args.use_schedular:
        schedular = CosineSchedularLinearWarmup(
            optimizer, len(train_set) // args.batch_size,
            10, args.epochs, lr=args.lr)
    else:
        schedular = None
    scaler = amp.GradScaler()
    acc_best = 0.
    for e in range(1, args.epochs):
        train_acc, train_loss = train(train_loader, model, optimizer,
                                      schedular, scaler, loss_sampling_weight, e, args)
        val_acc, val_loss = val(val_loader, model, args)

        # logging
        log_and_display(train_acc, val_acc, train_loss, val_loss)

        if val_acc > acc_best:
            torch.save(model.state_dict(), './model.pth')
            acc_best = val_acc

    # load best model
    model.load_state_dict(torch.load("model.pth"))
    test(
        (["train", "val", "test1", "test2", "test3"]
         , [train_loader, val_loader, t1_loader, t2_loader, t3_loader]
         )
        , model, args)
    torch.save(model.state_dict(), "%s.pth" % args.name)


def train(loader, model, optimizer, schedular, scaler, ratio, epoch, args):
    model.train()
    acc_meter, loss_meter = AverageMeter(), AverageMeter()
    total_loss = 0.
    for i, data in enumerate(loader):
        images, age, TIV, GMv, GMn, WMn, CSFn, targets, name = data
        images = images.to(device)
        age = age.to(device).unsqueeze(1).type(torch.float)
        TIV = TIV.to(device).unsqueeze(1).type(torch.float)
        GMv = GMv.to(device).unsqueeze(1).type(torch.float)
        GMn = GMn.to(device).unsqueeze(1).type(torch.float)
        WMn = WMn.to(device).unsqueeze(1).type(torch.float)
        CSFn = CSFn.to(device).unsqueeze(1).type(torch.float)
        targets = targets.to(device).unsqueeze(1)

        batch_size = images.size()[0]

        sampling_weights = torch.ones_like(targets)
        sampling_weights[targets == 1] = ratio
        with amp.autocast():
            if args.model == 1:
                output = model(images)
            else:
                output = model.forward_with_extra(images, age, TIV, GMv, GMn, WMn, CSFn)
            loss = F.binary_cross_entropy_with_logits(output, targets)

        total_loss += loss.item()
        loss_meter.update(loss.item())

        # opt
        if args.use_schedular:
            lr = schedular.update()
        else:
            lr = args.lr
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # acc
        pred = torch.where(output >= 0, 1., 0.)
        num_correct = (pred == targets).sum()
        acc_meter.update(num_correct, batch_size)

        if (i + 1) % args.log_freq == 0:
            logging.info('epoch [%3d/%3d][%4d/%4d], loss: %f, lr: %f' % (
                epoch, args.epochs, i, loader.__len__(),
                total_loss / args.log_freq, lr))
            total_loss = 0.0

    return acc_meter.avg() * 100, loss_meter.avg()


def val(loader, model, args):
    model.eval()
    acc_meter, loss_meter = AverageMeter(), AverageMeter()
    for data in tqdm(loader):
        with torch.no_grad():
            images, age, TIV, GMv, GMn, WMn, CSFn, targets, name = data
            images = images.to(device)
            age = age.to(device).unsqueeze(1).type(torch.float)
            TIV = TIV.to(device).unsqueeze(1).type(torch.float)
            GMv = GMv.to(device).unsqueeze(1).type(torch.float)
            GMn = GMn.to(device).unsqueeze(1).type(torch.float)
            WMn = WMn.to(device).unsqueeze(1).type(torch.float)
            CSFn = CSFn.to(device).unsqueeze(1).type(torch.float)
            targets = targets.to(device).unsqueeze(1)

            batch_size = images.size()[0]

            if args.model == 1:
                output = model(images)
            else:
                output = model.forward_with_extra(images, age, TIV, GMv, GMn, WMn, CSFn)
            loss = F.binary_cross_entropy_with_logits(output, targets)
            loss_meter.update(loss.item())

            # acc
            pred = torch.where(output >= 0, 1., 0.)
            num_correct = (pred == targets).sum()
            acc_meter.update(num_correct, batch_size)

    return acc_meter.avg() * 100, loss_meter.avg()


def test(loaders, model, args):
    model.eval()
    splits, loaders = loaders
    meters = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]
    plot_scores = []  # save scores to later plot their distribution with kdeplot

    writer = ExcelWriter("predictions.xlsx")
    for split, loader, meter in zip(splits, loaders, meters):
        names, prediction = [], []
        for data in tqdm(loader):
            with torch.no_grad():
                images, age, TIV, GMv, GMn, WMn, CSFn, targets, name = data
                images = images.to(device)
                age = age.to(device).unsqueeze(1).type(torch.float)
                TIV = TIV.to(device).unsqueeze(1).type(torch.float)
                GMv = GMv.to(device).unsqueeze(1).type(torch.float)
                GMn = GMn.to(device).unsqueeze(1).type(torch.float)
                WMn = WMn.to(device).unsqueeze(1).type(torch.float)
                CSFn = CSFn.to(device).unsqueeze(1).type(torch.float)
                targets = targets.to(device).unsqueeze(1)

                batch_size = images.size()[0]

                if args.model == 1:
                    output = model(images)
                else:
                    output = model.forward_with_extra(images, age, TIV, GMv, GMn, WMn, CSFn)
                # acc
                pred = torch.where(output >= 0, 1., 0.)
                num_correct = (pred == targets).sum()
                meter.update(num_correct, batch_size)

                # saving data for logging predictions to xlsx
                names += name
                prediction += (torch.sigmoid(output.view(-1)).tolist())

        plot_scores.append(prediction)
        # save predictions to xlsx
        data_frame = pd.DataFrame({
            "IDs": names,
            "Score": prediction
        })
        data_frame.to_excel(writer, split, index=False)
    writer.save()
    plot_hist(plot_scores, args.name)
    logging.info("train acc: %.2f, val acc: %.2f, t1 acc: %.2f, t2 acc: %.2f, t3 acc: %.2f" %
                 (meters[0].avg(), meters[1].avg(), meters[2].avg(), meters[3].avg(), meters[4].avg()))


# data related
arg_parser = ArgumentParser(description='Medical Gender Classification')
arg_parser.add_argument('--data', type=str, default='', required=True,
                        help='path to data folder')
arg_parser.add_argument("--mix_split", default=False, action="store_true",
                        help="if True, uses random split to create "
                             "train and val set")
arg_parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size')
arg_parser.add_argument('--num_workers', default=2, type=int,
                        help='number of data loader workers')
# network related
arg_parser.add_argument("--dropout", default=0.2, type=float,
                        help="fully connect layer dropout")
arg_parser.add_argument("--model", default=1, type=int,
                        help="which model to use")
# optimization related
arg_parser.add_argument('--lr', type=float, default=1e-4,
                        help='optimizer learning rate')
arg_parser.add_argument('--weight_decay', type=float, default=5e-5,
                        help='optimizer weight_decay')
arg_parser.add_argument('--use_schedular', action='store_true',
                        help='uses learning rate warmup and decay')
# training related
arg_parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs')
arg_parser.add_argument('--seed', type=int, default=34324
                        , help='number of training epochs')
arg_parser.add_argument('--log_freq', type=int, default=2,
                        help='frequency of logging')
# others
arg_parser.add_argument('--name', type=str, default='', required=True,
                        help='experiment name')
arg_parser.add_argument('--ram', default=False, action="store_true",
                        help="if True transfers images to RAM for faster IO")
arg = arg_parser.parse_args()

# config logger
logging.basicConfig(
    format='[%(levelname)s] %(module)s - %(message)s',
    level=logging.INFO
)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    logging.info('Training on GPU')
else:
    device = torch.device("cpu")
    logging.info("WARNING!!!")
    logging.info("Training on CPU")
if __name__ == '__main__':
    main(arg)
