#! /usr/bin/env python
# coding: utf-8
import os, sys
import numpy as np
import cv2 as cv
import mxnet as mx
import matplotlib.pyplot as plt
import glob
import mxnet.metric
import warnings
from logger.logger_v1 import LogHandler
logging = LogHandler()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.file_func import get_class_name, get_function_name
from train.trainer_v002 import TrainerWithEpoch
from config_v002 import FMTrainConfig, FMEvalConfig
from config.config_v001 import Config

from model_zoo.mobilenet_v2 import MobileNetV20Transform
from model.mobilenet_v001 import get_mobilenet_v2
from callback.callback import CheckpointCallback, Speedometer
from dataset_v002 import RecDataIterV1
from fm_metric import MultiMetric
from eval_v002 import FMEvalParams, FMEvaluator
import utils.cv_show as CVShow
import utils.cv_save as CVSave


class FMTrainerWithEpoch(TrainerWithEpoch):

    def __init__(
        self,
        config,
        batch_end_callback=None,
        epoch_end_callback=None,
    ):
        super(FMTrainerWithEpoch, self).__init__(
            config=config,
            batch_end_callback=batch_end_callback,
            epoch_end_callback=epoch_end_callback,
        )
        assert isinstance(config, Config), f'config file is not correct!!!'
        self.config = config

        if len(self.config.gpus) <= 0:
            self.ctxs = [mx.cpu()]
        else:
            self.ctxs = [mx.gpu(int(i)) for i in config.gpus.split(',')]

        self.init()

    def init(self):
        self.init_dataset()
        self.init_optimizer()
        self.init_loss()
        assert isinstance(self.losses, list)
        self.init_metric()
        assert isinstance(self.metrics, list)
        self.init_eval()

    def init_optimizer(self, optimizer='sgd', optimizer_params=None):
        self.lr_schedule = mx.lr_scheduler.FactorScheduler(
            step=5,
            factor=0.9,
            stop_factor_lr=0.000001,
            base_lr=self.config.learning_rate,
            warmup_steps=0
        )
        logging.info(f'[init_optimizer] lr-schedule: {self.lr_schedule}')
        self.trainer = mx.gluon.Trainer(
            self.net.collect_params(),
            optimizer,
            optimizer_params={
                'learning_rate': self.config.learning_rate,
                'wd': self.config.weight_decay,
                # 'lr_scheduler': self.lr_schedule
            }
        )

    def init_loss(self):
        if self.losses is None:
            self.losses = []
        self.losses.append(mx.gluon.loss.L1Loss(self.config.loss_weights[0]))

    def init_metric(self):
        if self.metrics is None:
            self.metrics = []
        self.metrics.append(MultiMetric(name='train-L2-68'))

    def init_eval(self):
        data_process_func = lambda x, y: (x, y)
        self.eval_params = FMEvalParams(
            data_process_func=data_process_func,
            log_interval=1,
            metrics=[MultiMetric(name='eval-L2-68')],
            show=self.config.show,
            show_func=CVShow.cv_show_lm_rets,
            save=self.config.save,
            save_func=CVSave.cv_save_lm_rets,
            save_dir=self.config.save_prefix,
            dataloader=self.eval_dataloader,
            batch_size=self.config.batch_size,
            batch_end_callback_list=[Speedometer(self.config.batch_size, 1)],
            ctxs=self.ctxs,
            dtype=self.config.dtype
            )
        self.evaluator = FMEvaluator()

    def init_dataset(self):
        # self.train_dataset = FMDataset(
        #     root=self.config.image_dir,
        #     data_shape=self.config.data_shape,
        #     label_shape=self.config.label_shape,
        #     data_names=self.config.data_names,
        #     label_names=self.config.label_names,
        #     flag=1,
        #     transform=[lambda data, label: (data.astype(np.float32) / 255, label)],
        #     batch_size=self.config.batch_size
        # )
        # self.train_dataloader = mx.gluon.data.DataLoader(
        #     dataset=self.train_dataset,
        #     batch_size=self.config.batch_size,
        #     shuffle=True,
        #     num_workers=4,
        #     last_batch='discard'
        # )
        # self.eval_dataset = FMDataset(
        #     root=self.config.eval_image_dir,
        #     data_shape=self.config.data_shape,
        #     label_shape=self.config.label_shape,
        #     data_names=self.config.data_names,
        #     label_names=self.config.label_names,
        #     flag=1,
        #     transform=[lambda data, label: (data.astype(np.float32) / 255, label)],
        #     batch_size=self.config.batch_size
        # )
        # self.eval_dataloader = mx.gluon.data.DataLoader(
        #     dataset=self.eval_dataset,
        #     batch_size=self.config.batch_size,
        #     shuffle=False,
        #     num_workers=4,
        #     last_batch='discard'
        # )

        logging.info(f'[init_dataset]')
        self.train_dataloader = iter(RecDataIterV1(
            rec_dir=self.config.train_rec_dir,
            rec_prefix=self.config.train_rec_prefix,
            batch_size=self.config.batch_size,
            shuffle=True
        ))
        self.eval_dataloader = iter(RecDataIterV1(
            rec_dir=self.config.eval_rec_dir,
            rec_prefix=self.config.eval_rec_prefix,
            batch_size=self.config.batch_size,
            shuffle=False
        ))

    def compute_losses(self, preds, labels):
        losses = self.losses[0](preds, labels / 64.0)
        return losses


def TEST_TrainerWithEpoch():
    config = FMTrainConfig()

    trainer = FMTrainerWithEpoch(
        config,
        batch_end_callback=[
            Speedometer(config.batch_size, config.log_interval)
        ],
        epoch_end_callback=[CheckpointCallback()]
    )

    trainer.train()


if __name__ == '__main__':


    TEST_TrainerWithEpoch()
