# -*- coding: utf-8 -*-
import os
import sys
# root_dir = os.path.dirname(os.path.dirname(__file__))
# sys.path.insert(0, root_dir)
sys.path.insert(0, r'D:\personal\code\tf2.0\FacialLandmark')
from log.logger import Logger
from callback.callback import CallBack
import tensorflow as tf


class TrainCallBackLog(CallBack):

    def __init__(self, name='train-callback-log'):
        self.name = name

    def call(self, cfg, dataset, train_brick):
        acc = train_brick.metric['train_accurency'].get_avg()
        loss = train_brick.metric['train_loss'].get_avg()
        message = f'[{train_brick.epoch}/{cfg.MAX_EPOCH} {train_brick.step}]  acc={acc[0]}/{acc[1]}  loss={loss[0]}/{loss[1]}'
        Logger.debug(
            message,
            show_type=Logger.LOG_STYLE.DEFAULT,
            forground=Logger.LOG_FRONT_COLOR.GREEN,
            background=Logger.LOG_BACK_COLOR.DEFAULT
        )


class TrainCallBackSaveCheckPoints(CallBack):

    def __init__(self, name='train-callback-save-checkpoints'):
        self.name = name

    def call(self, cfg, dataset, train_brick):
        message = f'[{train_brick.epoch}/{cfg.MAX_EPOCH} {train_brick.step}]  start saving checkpoints...'
        Logger.debug(
            message,
            show_type=Logger.LOG_STYLE.DEFAULT,
            forground=Logger.LOG_FRONT_COLOR.GREEN,
            background=Logger.LOG_BACK_COLOR.DEFAULT
        )
        ##  TODO
        checkpoint_file = cfg.save_checkpoint_prefix + '_' + str(train_brick.epoch) + '_' + str(train_brick.step)
        checkpoint_file_path = os.path.join(cfg.save_checkpoint_path, checkpoint_file)
        train_brick.model.save_weights(checkpoint_file_path)
        message = f'[{train_brick.epoch}/{cfg.MAX_EPOCH} {train_brick.step}]  complete save checkpoints.'
        Logger.debug(
            message,
            show_type=Logger.LOG_STYLE.DEFAULT,
            forground=Logger.LOG_FRONT_COLOR.GREEN,
            background=Logger.LOG_BACK_COLOR.DEFAULT
        )



if __name__ == '__main__':
    from metric.acc_metric import AccMetric
    import numpy as np
    acc_metric = AccMetric()
    predictions = np.array(
        [
            [0.9, 0.1, 0.0, 0.0, 0.0],
            [0.1, 0.9, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.9, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.9, 0.0],
            [0.0, 0.0, 0.0, 0.1, 0.9],
            [0.9, 0.1, 0.0, 0.0, 0.0],
            [0.1, 0.9, 0.0, 0.0, 0.0]
        ]
    )
    labels = np.array([0, 1, 2, 3, 4, 0, 0])
    acc_metric.update(labels, predictions)
    acc = acc_metric.get_avg()

    cb = TrainCallBackLog()
    class TrainBrick():
        def __init__(self):
            self.step = 10
            self.epoch = 1
            self.metric = {
                'train_accurency':acc_metric,
                'train_loss':acc_metric
            }
    class Cfg():
        def __init__(self):
            self.MAX_EPOCH = 10
    train_brick = TrainBrick()
    cfg = Cfg()
    cb.call(cfg, None, train_brick)