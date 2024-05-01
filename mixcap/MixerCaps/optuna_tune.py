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
import optuna
import logging

# from functools import partial

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

model = nn.DataParallel(QCaps())
model = model.cuda()
smooth_l1_loss = nn.SmoothL1Loss().cuda()


def test():
    model.eval()

    pred_scores = []
    gt_scores = []
    all_loss = []

    for i, data in enumerate(test_loader):
        img = data[0].cuda()
        data[1] = data[1] * scale
        score_label = data[1]

        label = score_label
        img = img.cuda()
        label = label.cuda()

        y_pred, _ = model(img)
        y_pred = y_pred.squeeze()

        loss = smooth_l1_loss(y_pred, label)
        all_loss.append(loss.item())
        pred_scores = pred_scores + y_pred.cpu().tolist()
        gt_scores = gt_scores + label.cpu().tolist()

    pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, cfg.DATASET.AUGMENTATION)), axis=1)
    gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, cfg.DATASET.AUGMENTATION)), axis=1)

    test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
    test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

    model.train(True)

    return test_srcc, test_plcc

def objective(trial):
    epoch = 4
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    momentum = 0.95
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-2)
    optimizer = SGD(params=model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    best_srcc = 0.0
    best_plcc = 0.0
    for t in range(epoch):
        model.train(True)
        epoch_loss = []
        pred_scores = []
        gt_scores = []

        for i, data in enumerate(train_loader):
            x = data[0].cuda()
            score_label = data[1] * scale
            score_label = score_label.cuda()

            optimizer.zero_grad()

            y_pred, cla = model(x)

            y_pred = y_pred.squeeze()
            pred_scores = pred_scores + y_pred.cpu().tolist()
            gt_scores = gt_scores + score_label.cpu().tolist()

            loss = smooth_l1_loss(y_pred, score_label)

            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        # lr_decay.step()
        train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)

        test_srcc, test_plcc = test()
        
    return test_srcc

def tune():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler("tune.log", mode='w'))
    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()

    study = optuna.create_study(direction="maximize")

    logger.info("Start optimization.")
    study.optimize(objective, n_trials=10)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    print("Best hyperparameters: {}".format(trial.params))



if __name__ == '__main__':
    # dataset = "LIVE"
    # dataset = "KonIQ-10K"
    dataset = "SPAQ"
    if dataset == 'SPAQ':
        idx = list(range(0, 11124))
    if dataset == 'KonIQ-10K':
        idx = list(range(0, 10073))
    if dataset == 'LIVEC':
        idx = list(range(0, 1162))
    if dataset == 'LIVE':
        idx = list(range(0, 29))
    if dataset == 'TID2013':
        idx = list(range(0, 25))
    random.shuffle(idx)
    
    loader = Loader(dataset=dataset, idx=idx)
    train_loader, test_loader = loader.get()
    if dataset == "LIVE":
        scale = cfg.DATASET.LIVE.SCALE
        save_dir = '/home/liuxiaolong/IQA/QCaps/log/checkpoint/live'
    elif dataset == "LIVEC":
        scale = cfg.DATASET.LIVEC.SCALE
        save_dir = '/home/liuxiaolong/IQA/QCaps/log/checkpoint/livec'
    elif dataset == "TID2013":
        scale = cfg.DATASET.TID2013.SCALE
        save_dir = '/home/liuxiaolong/IQA/QCaps/log/checkpoint/tid2013'
    elif dataset == "KonIQ-10K":
        scale = cfg.DATASET.KONIQ.SCALE
        save_dir = '/home/liuxiaolong/IQA/QCaps/log/checkpoint/koniq'
    elif dataset == "KADID-10K":
        scale = cfg.DATASET.KADID.SCALE
        save_dir = '/home/liuxiaolong/IQA/QCaps/log/checkpoint/kadid'
    elif dataset == 'SPAQ':
        scale = cfg.DATASET.SPAQ.SCALE
        save_dir = '/home/liuxiaolong/IQA/QCaps/log/checkpoint/spaq'
    
    tune()

