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
from dataset.FM_dataset_v001 import FMDataset, FMDataLoader
from utils.file_func import get_class_name, get_function_name
from train.trainer_v001 import TrainerWithEpoch
from config.FM_config_v001 import FMTrainConfig, FMEvalConfig

from model_zoo.mobilenet_v2 import MobileNetV20Transform
from model.mobilenet_v001 import get_mobilenet_v2
from callback.callback import CheckpointCallback, Speedometer
from dataset.mx_data_iter_v012 import RecDataIterV1
from metric.FM_metric import MultiMetric



class FMTrainerWithEpoch(TrainerWithEpoch):

    def __init__(
        self,
        config,
        batch_end_callback=None,
        epoch_end_callback=None,
        for_train=True
    ):
        super(FMTrainerWithEpoch, self).__init__(
            config=config,
            batch_end_callback=batch_end_callback,
            epoch_end_callback=epoch_end_callback,
            for_train=True
        )
        # assert isinstance(config, Config), f'config file is not correct!!!'
        self.config = config

        if len(self.config.gpus) <= 0:
            self.ctxs = [mx.cpu()]
        else:
            self.ctxs = [mx.gpu(int(i)) for i in config.gpus.split(',')]

        self.init()


    def init(self):
        self.load_model(self.config.pretrained_prefix, self.config.pretrained_epoch)
        self.init_dataset()
        self.init_optimizer()
        self.init_loss()
        assert isinstance(self.losses, list)
        self.init_metric()
        assert isinstance(self.metrics, list)
        self.init_eval_metric()

    def init_optimizer(self, optimizer='sgd', optimizer_params=None):
        self.lr_schedule = mx.lr_scheduler.FactorScheduler(
            step=3,
            factor=0.9,
            stop_factor_lr=0.000001,
            base_lr=self.config.learning_rate,
            warmup_steps=self.config.min_epoch
        )
        logging.info(f'[init_optimizer] lr-schedule: {self.lr_schedule}')
        self.trainer = mx.gluon.Trainer(
            self.net.collect_params(),
            optimizer,
            optimizer_params={
                'learning_rate': self.config.learning_rate,
                'wd': self.config.weight_decay,
                'lr_scheduler': self.lr_schedule
            }
        )

    def init_loss(self):
        if self.losses is None:
            self.losses = []
        self.losses.append(mx.gluon.loss.L2Loss(self.config.loss_weights[0]))
        self.losses.append(mx.gluon.loss.L2Loss(self.config.loss_weights[1]))

    def init_metric(self):
        if self.metrics is None:
            self.metrics = []
        self.metrics.append(MultiMetric(name='L2-68'))
        self.metrics.append(MultiMetric(name='L2-5'))

    def init_eval_metric(self):
        if self.eval_metrics is None:
            self.eval_metrics = []
        self.eval_metrics.append(MultiMetric())

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
        rec_dir = '/data1/dataset/300/rec/002'
        rec_prefix = '300vw_1_1'
        batch_size = 2
        shuffle = True
        self.train_dataloader = iter(RecDataIterV1(
            rec_dir=rec_dir,
            rec_prefix=rec_prefix,
            batch_size=batch_size,
            shuffle=shuffle
        ))

    def train_batch(self, datas, labels):
        self.net.cast('float64')
        gpu_datas = mx.gluon.utils.split_and_load(datas, self.ctxs)
        gpu_labels0 = mx.gluon.utils.split_and_load(labels[0], self.ctxs)
        gpu_labels1 = mx.gluon.utils.split_and_load(labels[1], self.ctxs)
        with mx.autograd.record():
            losses = []
            # losses = [self.sec_loss(self.net(x), y) for x, y in zip(gpu_datas, gpu_labels)]
            for x, y1, y2 in zip(gpu_datas, gpu_labels0, gpu_labels1):
                # logging.info(f'[train] x: {x.shape}  {x.dtype}  y1: {y1.shape}  y1: {y1.dtype}  y2: {y2.shape}  y2: {y2.dtype}')
                x = x.astype(np.float64)
                preds = self.net(x)
                loss12 = self.batch_compute(preds=preds, labels=(y1.astype(np.float64), y2.astype(np.float64)))
                losses.append(loss12[0] + loss12[1])
        for loss in losses:
            loss.backward()
        self.trainer.step(self.config.batch_size)
        mx.nd.waitall()

    def batch_compute(self, preds, labels):
        logging.info(f'preds: {len(preds)}  {preds[0].shape}  {preds[0].dtype}  {preds[1].shape}  {preds[1].dtype}')
        logging.info(f'labels: {len(labels)}  {labels[0].shape}  {labels[0].dtype}  {np.max(labels[0])}  {labels[1].shape}  {labels[1].dtype}')
        losses = [loss(preds[idx], labels[idx] / 128) for idx, loss in enumerate(self.losses)]
        if len(self.metrics) > 0:
            for idx, metric in enumerate(self.metrics):
                metric.update(labels[idx], preds[idx])
        return losses
        for idx, loss_func in enumerate(self.losses):
            loss = loss_func(preds[idx], labels[idx] / 128.0)
            self.metrics[idx].update()


def TEST_TrainerWithEpoch():
    config = FMTrainConfig()

    # MobileNetV20Transform().transform(
    #     num_class=10,  # config.num_class,
    #     prefix=config.pretrained_prefix,
    #     epoch=config.pretrained_epoch
    # )

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
