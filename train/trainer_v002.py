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
from logger.logger_v1 import LogHandler
logging = LogHandler()

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
        epoch_end_callback=None
    ):
        self.config = config
        self.batch_end_callback = batch_end_callback
        self.epoch_end_callback = epoch_end_callback
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
    ):
        super(TrainerWithEpoch, self).__init__(
            config=config,
            batch_end_callback=batch_end_callback,
            epoch_end_callback=epoch_end_callback,
        )
        assert isinstance(config, Config), f'config file is not correct!!!'
        self.config = config
        self.net = None

        self.net = self.load_model(self.config.pretrained_prefix, self.config.pretrained_epoch)
        self.net.cast(self.config.dtype)
        self.train_dataset = None
        self.train_dataloader = None
        self.eval_dataset = None
        self.eval_dataloader = None
        self.losses = None
        self.metrics = None
        self.lr_schedule = None
        self.evaluator = None
        self.eval_params = None

    def init_optimizer(self, optimizer='sgd', optimizer_params=None):
        raise NotImplementedError('not implemented!!!')

    def init_loss(self):
        raise NotImplementedError('not implemented!!!')

    def init_metric(self):
        raise NotImplementedError('not implemented!!!')

    def init_dataset(self):
        raise NotImplementedError('not implemented!!!')

    def init_eval(self):
        raise NotImplementedError('not implemented!!!')

    def train(self):
        for epoch in range(self.config.min_epoch, self.config.max_epoch):
            self.train_dataloader.reset()
            if self.eval_params is not None:
                self.eval_params.reset_metrics()

            self.train_epoch(epoch=epoch)

            if self.lr_schedule is not None:
                self.lr_schedule(epoch - self.config.min_epoch)
                self.trainer.set_learning_rate(self.lr_schedule.base_lr)

            for callback in self.epoch_end_callback:
                callback(self.net, self.config.checkpoint_dir, (epoch+1), self.metrics)

            model = self.load_model(self.config.checkpoint_dir, (epoch+1))
            self.evaluator.set_model(model)
            self.evaluator.eval(self.eval_params)

    def train_epoch(self, epoch):
        # for nbatch, (datas, labels) in enumerate(self.train_dataloader):
        for nbatch, databatch in enumerate(self.train_dataloader):

            datas, labels = databatch.data[0], databatch.label[0]
            self.train_batch(datas, labels)

            # batch-end-callback
            if self.batch_end_callback is not None:
                batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch, eval_metric=self.metrics, locals=locals())
                for callback in self.batch_end_callback:
                    callback(batch_end_params)

    def train_batch(self, datas, labels):
        gpu_datas = mx.gluon.utils.split_and_load(datas, self.ctxs)
        gpu_labels = mx.gluon.utils.split_and_load(labels, self.ctxs)
        with mx.autograd.record():
            losses = []
            for datai, labeli in zip(gpu_datas, gpu_labels):
                predi = self.net(datai.astype(self.config.dtype))
                losses.append(self.compute_losses(preds=predi, labels=labeli))
                self.update_metrics(predi, labeli)
        for loss in losses:
            loss.backward()
        self.trainer.step(self.config.batch_size)
        mx.nd.waitall()

    def compute_losses(self, preds, labels):
        losses = [loss(preds[idx], labels[idx]) for idx, loss in enumerate(self.losses)]
        return losses

    def update_metrics(self, preds, labels):
        if len(self.metrics) > 0:
            for idx, metric in enumerate(self.metrics):
                metric.update(labels[idx], preds[idx])
