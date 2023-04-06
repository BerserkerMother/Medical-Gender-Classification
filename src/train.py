from argparse import ArgumentParser
import logging
import wandb

import torch
from torch.utils.data import DataLoader

from trainer import Trainer
from data import MedicalDataset
from model import SimNetExtra
from utils import AverageMeter, set_seed


def main(args):
    # wandb init
    if args.wandb:
        wandb.init(
            entity="berserkermother",
            project="medical",
            name=f"model_{args.model}",
            config=args
        )

    if args.seed:
        set_seed(args.seed)

    # dataset
    if args.mix_split:
        dataset = MedicalDataset(args.data, splits='train+val+test1',
                                 transform=None, ram=args.ram, train=True)
        train_set, val_set, test1_set = dataset.split_train_test(
            ratios=[.8, .1])
    else:
        train_set = MedicalDataset(args.data, splits='train', ram=args.ram)
        val_set = MedicalDataset(args.data, splits='val', ram=args.ram)
        test1_set = MedicalDataset(args.data, splits='test1')
    test2_set = MedicalDataset(args.data, splits='test2')
    test3_set = MedicalDataset(args.data, splits='test3')

    datasets_name = ["train", "val", "healthy", "MCI", "Alz"]
    datasets = [train_set, val_set, test1_set, test2_set, test3_set]
    datasets = {k: v for k, v in zip(datasets_name, datasets)}

    loaders = {}
    for key, value in datasets.items():
        loader = DataLoader(
            dataset=value,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True
        )
        loaders[key] = loader

    # model and optimizer
    model = SimNetExtra(dropout=args.dropout,
                        model_num=args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loaders=loaders,
        args=args
    )
    trainer.train()


# data related
arg_parser = ArgumentParser(description='Medical Gender Classification')
arg_parser.add_argument('--data', type=str, default='', required=True,
                        help='path to data folder')
arg_parser.add_argument("--mix_split", default=False, action="store_true",
                        help="if True, uses random split to create "
                             "train and val set")
arg_parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size')
arg_parser.add_argument('--num_workers', default=8, type=int,
                        help='number of data loader workers')
# network related
arg_parser.add_argument("--dropout", default=0.2, type=float,
                        help="fully connect layer dropout")
arg_parser.add_argument("--model", default=1, type=int,
                        help="which model to use")
arg_parser.add_argument("--fp16", default=False, action="store_true",
                        help="whether to use fp16")
# optimization related
arg_parser.add_argument('--lr', type=float, default=1e-4,
                        help='optimizer learning rate')
arg_parser.add_argument('--weight_decay', type=float, default=5e-5,
                        help='optimizer weight_decay')
arg_parser.add_argument('--schedular', action='store_true',
                        help='uses learning rate warmup and decay')
arg_parser.add_argument('--warmup_steps', default=100, type=int,
                        help='uses learning rate warmup and decay')
# training related
arg_parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs')
arg_parser.add_argument('--seed', type=int, default=34324,
                        help='number of training epochs')
arg_parser.add_argument('--log_freq', type=int, default=300,
                        help='frequency of logging')
# others
arg_parser.add_argument('--name', type=str, default='', required=True,
                        help='experiment name')
arg_parser.add_argument('--wandb', default=False, action="store_true",
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
    arg.device = device
    main(arg)
