1 from model.solver import *
from config import cfg
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging
import random
import time

random.seed(256)

def main():
    torch.backends.cudnn.benchmark = True
    loop_num = cfg.SOLVER.TRAIN_TEST_LOOP
    srcc_all = np.zeros(loop_num, dtype=float)
    plcc_all = np.zeros(loop_num, dtype=float)
    dataset = "SPAQ"
    # dataset = "KonIQ-10K"
    # dataset = "LIVEC"
    # dataset = "LIVE"
    # dataset = 'TID2013'
    # dataset = 'KADID-10K'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        filename='/home/liuxiaolong/IQA/MixerCaps/log/logfile/param_%s_%s.log' % (cfg.LOGFILE.NAME, dataset))
    logging.info("FIX: %s" % cfg.LOGFILE.CAPTION)
    for i in range(loop_num):
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
        solver = Sovler(dataset=dataset, idx=idx)
        srcc_all[i], plcc_all[i] = solver.train(train_num=i + 1)
        # logs, losses = solver.find_lr()
        # plt.plot(logs[10:-5], losses[10:-5])
        # plt.savefig('./lr.jpg')
        # solver.test()
    srcc_med = np.median(srcc_all)
    plcc_med = np.median(plcc_all)

    srcc_mean = np.mean(srcc_all)
    plcc_mean = np.mean(plcc_all)
    logging.info('Testing median SRCC: %4.4f, median PLCC: %4.4f' % (srcc_med, plcc_med))
    logging.info('Testing mean SRCC: %4.4f, mean PLCC: %4.4f' % (srcc_mean, plcc_mean))

if __name__ == '__main__':
    main()

