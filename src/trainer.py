import logging

import torch
import torch.nn.functional as F
from torch.cuda import amp
from schedular import CosineSchedularLinearWarmup

import pandas as pd
from pandas import ExcelWriter
from sklearn.manifold import TSNE
from tqdm import tqdm

from utils import AverageMeter, plot_tsne, wandb_log

device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)


class Trainer:
    def __init__(self, model, optimizer, loaders, args):
        self.model_num = args.model
        self.model = model
        self.optimizer = optimizer
        self.loaders = loaders
        self.num_epoch_steps = len(loaders["train"])
        self.progress_bar = tqdm(total=self.num_epoch_steps * args.epochs)
        self.args = args
        self.best_accuracy = 0

        if args.schedular:
            self.schedular = CosineSchedularLinearWarmup(
                optimizer=self.optimizer,
                num_training_steps=args.epochs * self.num_epoch_steps,
                warmup_steps=args.warmup_steps,
                lr=args.lr
            )
        if args.fp16:
            self.grad_scaler = amp.GradScaler()

        self.num_steps = 0  # tracks current step
        # tracks loss and accuracy for steps
        self.loss_meter, self.accuracy_meter = AverageMeter(), AverageMeter()

    def train(self):
        for e in range(self.args.epochs):
            self.train_one_epoch()
        self.load_model()
        self.test()

    def train_one_epoch(self):
        self.model.train()
        for i, data in enumerate(self.loaders["train"]):
            self.model.train()
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

            with amp.autocast():
                output, _ = self.model.forward_with_extra(
                    images, age, TIV, GMv, GMn, WMn, CSFn)
                loss = F.binary_cross_entropy_with_logits(output, targets)

            self.loss_meter.update(loss.item(), batch_size)

            # opt
            if self.args.schedular:
                lr = self.schedular.update()
            else:
                lr = self.args.lr
            self.optimizer.zero_grad()
            if self.args.fp16:
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # acc
            pred = torch.where(output >= 0, 1., 0.)
            num_correct = (pred == targets).sum()
            self.accuracy_meter.update(num_correct, batch_size)
            self.num_steps += 1

            # progress bar
            self.progress_bar.update(1)
            self.progress_bar.set_description(
                desc=f"loss: {loss.item(): 4.5f}, lr: {lr : .6f}"
            )

            if self.num_steps % self.args.log_freq == 0:
                train_acc, train_loss = self.eval(self.loaders["train"])
                val_acc, val_loss = self.eval(self.loaders["val"])
                logging.info(f"Epoch: {self.num_steps / self.num_epoch_steps: 3.1f}, "
                             f"train loss: {train_loss: 3.3f}, train accuracy: {train_acc: 2.2f}%, "
                             f"val loss: {val_loss: 3.3f}, val accuracy: {val_acc: 2.2f}%")
                if self.args.wandb:
                    wandb_log(train_acc, val_acc, train_loss, val_loss)
                if val_acc > self.best_accuracy:
                    self.best_accuracy = val_acc
                    self.save_model()

    def eval(self, loader):
        self.model.eval()
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

                output, _ = self.model.forward_with_extra(
                    images, age, TIV, GMv, GMn, WMn, CSFn)
                loss = F.binary_cross_entropy_with_logits(output, targets)
                loss_meter.update(loss.item())

                # acc
                pred = torch.where(output >= 0, 1., 0.)
                num_correct = (pred == targets).sum()
                acc_meter.update(num_correct, batch_size)
        return acc_meter.avg() * 100, loss_meter.avg()

    def test(self):
        self.model.eval()
        meters = [AverageMeter(), AverageMeter(), AverageMeter(),
                  AverageMeter(), AverageMeter()]
        cls_category = []
        plot_scores = []  # save scores to later plot their distribution with kdeplot

        writer = ExcelWriter("predictions_%s.xlsx" % self.args.name)
        for (split, loader), meter in zip(self.loaders.items(), meters):
            c_male, c_female = 0., 0.
            t_male, t_female = 0., 0.
            names, prediction = [], []
            t_sne = []
            t_sne_labels = []

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

                    output, emd = self.model.forward_with_extra(
                        images, age, TIV, GMv, GMn, WMn, CSFn)
                    t_sne.append(emd.cpu())
                    t_sne_labels.append(targets.to(torch.long).cpu())
                    # acc
                    pred = torch.where(output >= 0, 1., 0.)
                    num_correct = (pred == targets).sum()
                    c_male += (pred[targets == 0] ==
                               targets[targets == 0]).sum()
                    c_female += (pred[targets == 1] ==
                                 targets[targets == 1]).sum()
                    meter.update(num_correct, batch_size)

                    t_male += (targets[targets == 0] == 0).sum()
                    t_female += targets[targets == 1].sum()
                    # saving data for logging predictions to xlsx
                    names += name
                    prediction += (torch.sigmoid(output.view(-1)).tolist())
            t_sne_labels = torch.cat(t_sne_labels, dim=0).numpy()
            t_sne = torch.cat(t_sne, dim=0).numpy()
            emd_x = TSNE(perplexity=20).fit_transform(t_sne)
            plot_tsne(emd_x, t_sne_labels, split, self.args.name)

            cls_category.append((c_male / t_male, c_female / t_female))
            plot_scores.append(prediction)
            # save predictions to xlsx
            data_frame = pd.DataFrame({
                "IDs": names,
                "Score": prediction
            })
            data_frame.to_excel(writer, split, index=False)
        writer.save()
        # plot_hist(plot_scores, args.name)
        # log to text
        with open("res.txt", 'a') as f:
            f.write(self.args.name)
            f.write("train acc: %.2f, val acc: %.2f, t1 acc: %.2f, t2 acc: %.2f, t3 acc: %.2f\n" %
                    (meters[0].avg(), meters[1].avg(), meters[2].avg(), meters[3].avg(), meters[4].avg()))
            f.write("Male vs. Female\n"
                    "%2.2f, %2.2f\n"
                    "%2.2f, %2.2f\n"
                    "%2.2f, %2.2f\n"
                    "%2.2f, %2.2f\n"
                    "%2.2f, %2.2f\n" %
                    (cls_category[0][0], cls_category[0][1], cls_category[1][0], cls_category[1][1],
                     cls_category[2][0], cls_category[2][1], cls_category[3][0], cls_category[3][1],
                     cls_category[4][0], cls_category[4][1]))

    def save_model(self, path=None):
        if not path:
            path = f"{self.model_num}.pth"
        torch.save(self.model.state_dict(), path)

    def load_model(self, path=None):
        if not path:
            path = f"{self.model_num}.pth"
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
