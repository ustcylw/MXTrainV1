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
from mxnet.model import BatchEndParam
from logger.logger_v4 import Log
logging = Log()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset.image_folder_dataset import ImageFolderDataIter, ImageFolderDataset, ImageFolderDataLoader
from utils.file_func import get_class_name, get_function_name
from model_zoo.mobilenet_v2 import MobileNetV20Transform
from config.config_v001 import Config
# from config.config_v3 import Config, TrainConfig, EvalConfig


class Trainer(object):
    def __init__(
        self,
        config,  # training or eval params
        batch_end_callback=None,
        epoch_end_callback=None,
        for_train=True
    ):
        self.config = config
        self.batch_end_callback = batch_end_callback
        self.epoch_end_callback = epoch_end_callback
        self.for_train = for_train
        self.ctxs = [mx.cpu()] if self.config.gpus is None else [mx.gpu(int(i)) for i in self.config.gpus.split(',')]

    def init_metric(self):
        raise NotImplementedError

    def init_dataset(self):
        raise NotImplementedError

    def init_optimizer(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def save_model(self, model_prefix, model_epoch=0, state=True):
        logging.info(f'saving model ... {model_prefix} -- {model_epoch}')
        self.net.export(model_prefix, model_epoch)
        logging.info(f'save model complete.')

    def load_model(self, model_prefix, model_epoch=0, state=False):
        pretrained_json = model_prefix + '-symbol.json'
        pretrained_params = model_prefix + f'-{model_epoch:04d}.params'
        logging.info(f'loading pretraned model ... \n{pretrained_json}\n{pretrained_params}')
        net = mx.gluon.SymbolBlock.imports(pretrained_json, ['data'], pretrained_params, ctx=self.ctxs)
        logging.info(f'loading pretraned model complete.')
        return net


class TrainerWithEpoch(Trainer):

    def __init__(
        self,
        config,
        batch_end_callback=None,
        epoch_end_callback=None,
        for_train=True
    ):
        super(TrainerWithEpoch, self).__init__(
            config=config,
            batch_end_callback=batch_end_callback,
            epoch_end_callback=epoch_end_callback,
            for_train=for_train
        )
        assert isinstance(config, Config), f'config file is not correct!!!'
        self.config = config
        self.net = None

        # self.load_model(self.config.pretrained_prefix, self.config.pretrained_epoch)
        self.train_dataset = None
        self.train_dataloader = None
        self.eval_dataset = None
        self.eval_dataloader = None
        # self.init_dataset()
        # self.init_optimizer()
        self.losses = None
        # self.init_loss()
        # assert isinstance(self.losses, list)
        self.metrics = None
        # self.init_metric()
        # assert isinstance(self.metrics, list)
        self.eval_metrics = None
        # self.init_eval_metric()
        self.lr_schedule = None
        self.evaluator = None
        self.eval_params = None

    def init_optimizer(self, optimizer='sgd', optimizer_params=None):
        raise NotImplementedError('not implemented!!!')

    def init_loss(self):
        raise NotImplementedError('not implemented!!!')

    def init_metric(self):
        raise NotImplementedError('not implemented!!!')

    def init_eval_metric(self):
        raise NotImplementedError('not implemented!!!')

    def init_dataset(self):
        raise NotImplementedError('not implemented!!!')

    def init_eval(self, evaluator):
        raise NotImplementedError('not implemented!!!')

    def train(self):
        for epoch in range(self.config.min_epoch, self.config.max_epoch):
            self.train_epoch(epoch=epoch)
            if self.lr_schedule is not None:
                self.lr_schedule(epoch)

            for callback in self.epoch_end_callback:
                callback(self.net, self.config.checkpoint_dir, (epoch+1), self.metrics)

            model = self.load_model(self.config.pretrained_prefix, (epoch+1))
            self.evaluator.set_model(model)
            self.evaluator.eval(self.eval_params)

    def train_epoch(self, epoch):
            pass
            # # for nbatch, (datas, labels) in enumerate(self.train_dataloader):
            # for nbatch, databatch in enumerate(self.train_dataloader):
            #     # logging.info(f'datas: {type(datas)}  {datas.shape}')
            #     # logging.info(f'labels: {type(labels)}  {labels[0].shape}  {labels[1].shape}')
            #     # forward and backward
            #     datas, labels = databatch.data[0], databatch.label[0]
            #     self.train_batch(datas, labels)
            #
            #     # batch-end-callback
            #     if self.batch_end_callback is not None:
            #         batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch, eval_metric=self.metrics, locals=locals())
            #         for callback in self.batch_end_callback:
            #             callback(batch_end_params)

    def train_batch(self, datas, labels):
        print(f'[train_batch]  datas: {type(datas)}  labels: {type(labels)}')
        gpu_datas = mx.gluon.utils.split_and_load(datas, self.ctxs)
        gpu_labels = mx.gluon.utils.split_and_load(labels, self.ctxs)
        with mx.autograd.record():
            losses = []
            # losses = [self.sec_loss(self.net(x), y) for x, y in zip(gpu_datas, gpu_labels)]
            for x, y in zip(gpu_datas, gpu_labels):
                # logging.info(f'[train] x: {x.shape}  y: {y.shape}  {x.context}')
                preds = self.net(x)
                losses.append(self.batch_compute(preds=preds, labels=y))
        for loss in losses:
            loss.backward()
        self.trainer.step(self.config.batch_size)
        mx.nd.waitall()

    def batch_compute(self, preds, labels):
        # logging.info(f'preds: {len(preds)}  {preds[0].shape}  {preds[1].shape}')
        # logging.info(f'labels: {labels}')
        # logging.info(f'labels: {len(labels)}  {labels[0].shape}  {labels[1].shape}')
        losses = [loss(preds[idx], labels[idx]) for idx, loss in enumerate(self.losses)]
        if len(self.metrics) > 0:
            for idx, metric in enumerate(self.metrics):
                metric.update(labels[idx], preds[idx])
        return losses

    def forward(self, datas, labels=None, loss_funcs=None, metrics=None):
        print(f'[forward]  datas: {type(datas)}  {type(labels)}')
        gpu_datas = mx.gluon.utils.split_and_load(datas, self.ctxs)
        gpu_labels = mx.gluon.utils.split_and_load(labels, self.ctxs)
        losses = None
        # losses = [self.sec_loss(self.net(x), y) for x, y in zip(gpu_datas, gpu_labels)]
        for x, y in zip(gpu_datas, gpu_labels):
            # logging.info(f'[train] x: {x.shape}  y: {y.shape}  {x.context}')
            preds = self.net(x)

            if loss_funcs is None:
                continue
            losses = [loss(preds, y) for loss in loss_funcs]

            if metrics is None:
                continue
            if len(metrics) > 0:
                for metric in metrics:
                    metric.update(y, preds)
        return losses



def TEST_TrainerWithEpoch():
    config = TrainConfig()

    MobileNetV20Transform().transform(
        num_class=config.num_class,
        prefix=config.transform_prefix,
        epoch=config.transform_epoch
    )

    trainer = TrainerWithEpoch(config)

    trainer.train()



if __name__ == '__main__':


    TEST_TrainerWithFit()

    TEST_TrainerWithEpoch()