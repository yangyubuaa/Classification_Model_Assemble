# -*- utf-8 -*-
# @Time: 2021/4/20 5:58 下午
# @Author: yang yu
# @File: train_record.py.py
# @Software: PyCharm

import torch

class TrainProcessRecord:
    def __init__(self, model_save_threshold, loss_save_threshold):
        self.model_save_threshold = model_save_threshold
        self.loss_save_threshold = loss_save_threshold


    def __call__(self, model, params_save, epoch, batch, train_loss, eval_loss, train_accu, eval_accu):
        if batch % self.model_save_threshold == 0:
            save_path = "epoch{}batch{}".format(str(epoch), str(batch))
            torch.save(model, save_path)

        if batch % self.loss_save_threshold == 0:
            params_save.append((epoch, batch, train_loss, eval_loss, train_accu, eval_accu))
