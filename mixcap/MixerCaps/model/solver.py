import torch.nn as nn
from model import *
from config import cfg
from dataloader import *
import os
from torch.optim import Adam, lr_scheduler, SGD
import scipy.stats as stats
import numpy as np
import logging
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


class Sovler:
    def __init__(self, dataset, idx):
        self.model = nn.DataParallel(QCaps())
        self.model = self.model.cuda()

        self.smooth_l1_loss = nn.SmoothL1Loss().cuda()
        self.cross_entropy_loss = nn.CrossEntropyLoss().cuda()
        self.dataset = dataset
        self.idx = idx
        self.loader = Loader(dataset=self.dataset, idx=self.idx)
        self.train_loader, self.test_loader = self.loader.get()
        if self.dataset == "LIVE":
            self.scale = cfg.DATASET.LIVE.SCALE
            self.save_dir = '/home/liuxiaolong/IQA/QCaps/log/checkpoint/live'
        elif self.dataset == "LIVEC":
            self.scale = cfg.DATASET.LIVEC.SCALE
            self.save_dir = '/home/liuxiaolong/IQA/QCaps/log/checkpoint/livec'
        elif self.dataset == "TID2013":
            self.scale = cfg.DATASET.TID2013.SCALE
            self.save_dir = '/home/liuxiaolong/IQA/QCaps/log/checkpoint/tid2013'
        elif self.dataset == "KonIQ-10K":
            self.scale = cfg.DATASET.KONIQ.SCALE
            self.save_dir = '/home/liuxiaolong/IQA/QCaps/log/checkpoint/koniq'
        elif self.dataset == "KADID-10K":
            self.scale = cfg.DATASET.KADID.SCALE
            self.save_dir = '/home/liuxiaolong/IQA/QCaps/log/checkpoint/kadid'
        elif self.dataset == 'SPAQ':
            self.scale = cfg.DATASET.SPAQ.SCALE
            self.save_dir = '/home/liuxiaolong/IQA/QCaps/log/checkpoint/spaq'
        
        self.savename = cfg.LOGFILE.SAVENAME
        self.epoch = cfg.SOLVER.EPOCH
        self.lr = cfg.SOLVER.BASE_LR
        self.weight_decay = cfg.SOLVER.WEIGHT_DECAY
        # resnet_params = list(map(id, self.model.module.resnet.parameters()))
        # capsnet_params = filter(lambda p: id(p) not in resnet_params, self.model.parameters())
        # params = [
        #     {'params': self.model.module.resnet.parameters(), 'lr': self.lr * 0.5},
        #     {'params': capsnet_params, 'lr': self.lr}
        # ]
        # self.optimizer = Adam(params=params, weight_decay=self.weight_decay)
        # self.optimizer = Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.optimizer = SGD(params=self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)

    def find_lr(self, init_value=1e-6, final_value=10., beta=0.98):
        num = len(self.train_loader) - 1
        mult = (final_value / init_value) ** (1 / num)
        lr = init_value
        self.optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        for data in self.train_loader:
            batch_num += 1
            inputs = data[0].cuda()
            score_label = data[1] * self.scale
            score_label = data[1]
            labels = score_label.cuda()
            self.optimizer.zero_grad()
            outputs, _ = self.model(inputs)
            outputs = outputs.squeeze()
            loss = self.smooth_l1_loss(outputs, labels)


            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta ** batch_num)
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                return log_lrs, losses

            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss

            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))

            loss.backward()
            self.optimizer.step()

            lr *= mult
            self.optimizer.param_groups[0]['lr'] = lr
        return log_lrs, losses

    def train(self, train_num):
        best_srcc = 0.0
        best_plcc = 0.0
        lr_decay = lr_scheduler.StepLR(self.optimizer, step_size=cfg.SOLVER.LR_DECAY_STEP_SIZE,
                                       gamma=cfg.SOLVER.LR_DECAY_RATIO)
        logging.info('#%d' % train_num)
        logging.info('Epoch\tTrain_loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC')
        for t in range(self.epoch):
            self.model.train(True)
            epoch_loss = []
            pred_scores = []
            gt_scores = []

            for i, data in enumerate(self.train_loader):
                x = data[0].cuda()
                score_label = data[1] * self.scale
                # score_label = data[1]
                score_label = score_label.cuda()

                self.optimizer.zero_grad()

                y_pred = self.model(x)

                y_pred = y_pred.squeeze()
                pred_scores = pred_scores + y_pred.cpu().tolist()
                gt_scores = gt_scores + score_label.cpu().tolist()

                loss = self.smooth_l1_loss(y_pred, score_label)

                iteration = t * len(self.train_loader) + i

                epoch_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            lr_decay.step()
            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)

            test_srcc, test_plcc = self.test()
            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc


            logging.info('%d\t\t%4.3f\t\t%4.3f\t\t%4.4f\t\t%4.4f' % (
                t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc))
            save_dir = os.path.join(self.save_dir, str(train_num))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if test_srcc == best_srcc:
                self.model.eval()
                torch.save(self.model.state_dict(), '%s/%s/%s' % (self.save_dir, str(train_num), self.savename))
                self.model.train(True)

        logging.info('Best test SRCC: %f, PLCC: %f' % (best_srcc, best_plcc))
        logging.info('')

        return best_srcc, best_plcc

    def test(self):
        self.model.eval()

        pred_scores = []
        gt_scores = []
        all_loss = []

        for i, data in enumerate(self.test_loader):
            img = data[0].cuda()
            data[1] = data[1] * self.scale
            score_label = data[1]

            label = score_label
            img = img.cuda()
            label = label.cuda()

            y_pred = self.model(img)

            y_pred = y_pred.squeeze()

            loss = self.smooth_l1_loss(y_pred, label)
            all_loss.append(loss.item())
            pred_scores = pred_scores + y_pred.cpu().tolist()
            gt_scores = gt_scores + label.cpu().tolist()



        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, cfg.DATASET.AUGMENTATION)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, cfg.DATASET.AUGMENTATION)), axis=1)

        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

        self.model.train(True)

        return test_srcc, test_plcc

