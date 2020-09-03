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
from logger.logger_v4 import Log
logging = Log()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset.image_folder_dataset import ImageFolderDataIter, ImageFolderDataset, ImageFolderDataLoader
from utils.file_func import get_class_name, get_function_name
from model_zoo.mobilenet_v2 import MobileNetV20Transform
from config.config_v1 import Config, Config1
from config.config_v3 import Config, TrainConfig, EvalConfig


class Trainer(object):
    def __init__(self, config):
        self.config = config

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
        raise NotImplementedError


class TrainerWithFit(Trainer):
    '''
    tips: for one thing is training
    '''
    def __init__(self, config):
        assert isinstance(config, Config), f'config file is not correct!!!'
        self.config = config
        self.model = None

        if len(self.config.gpus) <= 0:
            self.ctxs = [mx.cpu()]
        else:
            self.ctxs = [mx.gpu(int(i)) for i in config.gpus.split(',')]

        self.init_fit_data()
        self.init_trainer()
        self.init_loss()
        self.init_metric()
        self.init_optimizer(optimizer='sgd', optimizer_params=None)

    def init_metric(self):
        self.train_acc = mx.metric.Accuracy()
        self.eval_acc = mx.metric.Accuracy()

    def init_dataset(self):
        self.train_iter = ImageFolderDataIter(
            root=self.config.image_dir,
            data_shape=self.config.data_shape,
            label_shape=(2,),
            data_names=['data'],
            label_names=['softmax_label'],
            flag=1,
            transform=lambda data, label: (data.astype(np.float32) / 255, label),
            batch_size=self.config.batch_size
        )
        # data_iter = iter(self.train_iter)
        # batch = next(data_iter)
        # logging.info(f'[] label: {batch.label}')
        # # sys.exit(0)
        # self.train_dataset = ImageFolderDataset(
        #     root=config.image_dir,
        #     flag=1,
        #     transform=lambda data, label: (data.astype(np.float32)/255, label)
        # )
        # self.train_dataloader = mx.gluon.data.DataLoader(
        #     dataset=self.train_dataset,
        #     batch_size=self.config.batch_size,
        #     shuffle=True
        # )
        # self.eval_dataset = ImageFolderDataset(
        #     root=config.image_dir,
        #     flag=1,
        #     transform=lambda data, label: (data.astype(np.float32)/255, label)
        # )
        # self.eval_dataloader = mx.gluon.data.DataLoader(
        #     dataset=self.eval_dataset,
        #     batch_size=self.config.batch_size,
        #     shuffle=True
        # )
        # self.train_iter = mx.contrib.io.DataLoaderIter(self.train_dataloader)

    def init_optimizer(
        self,
        optimizer='sgd',
        optimizer_params=None
    ):
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params if optimizer_params is not None else {'learning_rate': self.config.learning_rate, 'wd': self.config.weight_decay}

    def load_model(self, model_prefix, model_epoch=0, state=False):
        # sym, arg_params, aux_params = self.load_checkpoint()
        return mx.model.load_checkpoint(
            prefix=self.config.pretrained_prefix,
            epoch=self.config.pretrained_epoch
        )

    def train(self):
        sym, arg_params, aux_params = self.load_model(self.config.pretrained_prefix, self.config.pretrained_epoch, state=False)
        model = mx.mod.Module(
            symbol=sym,
            context=self.ctxs,
        )
        # logging.info(f'[0] {model.output_names}')
        # logging.info(f'[0] {model.label_names}')
        # logging.info(f'[0] {model.data_names}')
        # logging.info(f'[0] {self.train_iter.provide_data}')
        # logging.info(f'[0] {self.train_iter.provide_label}')
        # logging.info(f'[0] model.data_names: {model.data_names}')
        # logging.info(f'[0] model.label_names: {model.label_names}')
        data_shapes = [('data', (self.config.batch_size, 3, 128, 128))]
        label_shapes = [('softmax_label', (self.config.batch_size, ))]
        # logging.info(f'[0] data_shapes: {data_shapes}')
        # logging.info(f'[0] label_shapes: {label_shapes}')
        model.bind(
            # data_shapes=self.train_iter.provide_data,  # [('data', (self.config.batch_size, 3, 128, 128))],
            # label_shapes=self.train_iter.provide_label,  # [('softmax_label', (self.config.batch_size, self.config.num_class))],
            # data_shapes=self.train_iter.provide_data,  # [('data', (self.config.batch_size, 3, 128, 128))],
            # label_shapes=self.train_iter.provide_label  # [('softmax_label', (self.config.batch_size, self.config.num_class))],
            data_shapes=data_shapes,
            label_shapes=label_shapes,
        )
        # logging.info(f'[1] {model.output_names}')
        # logging.info(f'[1] {model.label_names}')
        # logging.info(f'[1] {model.label_names}  {model.label_shapes}')
        # logging.info(f'[1] {model.data_names}  {model.data_shapes}')
        model.set_params(arg_params=arg_params, aux_params=aux_params, allow_missing=False)
        model.fit(
            train_data=self.train_iter,
            optimizer=self.optimizer,
            optimizer_params=self.optimizer_params,
            num_epoch=self.config.max_epoch,
            batch_end_callback=[
                mx.callback.Speedometer(self.config.batch_size, self.config.log_interval)
            ],
            # epoch_end_callback=mx.callback.do_checkpoint(self.config.pretrained_name, 1),
            epoch_end_callback=mx.callback.module_checkpoint(model, self.config.pretrained_prefix, 1, save_optimizer_states=True),
            eval_metric=self.train_acc
        )


def TEST_TrainerWithFit():
    config = TrainConfig()

    MobileNetV20Transform().transform(
        num_class=config.num_class,
        prefix=config.transform_prefix,
        epoch=config.transform_epoch
    )

    trainer = TrainerWithFit(config)

    trainer.train()


class TrainerWithEpoch(Trainer):

    def __init__(self, config):
        super(TrainerWithEpoch, self).__init__(config=config)
        assert isinstance(config, Config), f'config file is not correct!!!'
        self.config = config
        self.model = None

        if len(self.config.gpus) <= 0:
            self.ctxs = [mx.cpu()]
        else:
            self.ctxs = [mx.gpu(int(i)) for i in config.gpus.split(',')]

        self.load_model()
        self.init_dataset()
        self.init_optimizer()
        self.losses = []
        self.init_loss()
        self.init_metric()

    def init_optimizer(self, optimizer='sgd', optimizer_params=None):
        self.trainer = mx.gluon.Trainer(
            self.net.collect_params(),
            optimizer,
            optimizer_params=optimizer_params  # {'learning_rate': self.config.learning_rate, 'wd': self.config.weight_decay}
        )

    def load_model(self):
        pretrained_json = self.config.pretrained_prefix+'-symbol.json'
        pretrained_params = self.config.pretrained_prefix+f'-{self.config.pretrained_epoch:04d}.params'
        logging.info(f'loading pretraned model ... \n{pretrained_json}\n{pretrained_params}')
        self.net = mx.gluon.SymbolBlock.imports(pretrained_json, ['data'], pretrained_params, ctx=self.ctxs)
        logging.info(f'loading pretraned model complete.')

    def init_loss(self):
        # self.sec_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
        self.losses.append(mx.gluon.loss.L2Loss(self.config.loss_weights[0]))
        # self.losses.append(mx.gluon.loss.L2Loss(self.config.loss_weights[0]))

    def init_metric(self):
        self.train_acc = mx.metric.Accuracy()
        self.eval_acc = mx.metric.Accuracy()

    def init_dataset(self):
        self.train_dataset = ImageFolderDataset(
            root=self.config.image_dir,
            flag=1,
            transform=lambda data, label: (data.astype(np.float32) / 255, label)
        )
        self.train_dataloader = mx.gluon.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        self.eval_dataset = ImageFolderDataset(
            root=self.config.image_dir,
            flag=1,
            transform=lambda data, label: (data.astype(np.float32) / 255, label)
        )
        self.eval_dataloader = mx.gluon.data.DataLoader(
            dataset=self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

    def save_model(self, model_prefix, model_epoch=0):
        logging.info(f'saving model ... {model_prefix} -- {model_epoch}')
        self.net.export(model_prefix, model_epoch)
        logging.info(f'save model complete.')

    def train(self):
        for epoch in range(self.config.min_epoch, self.config.max_epoch):
            self.train_epoch(epoch=epoch)
            ## TODO
            # check loss wethere save model-parameter
            # eval
    def train_epoch(self, epoch):

            total_loss = 0
            for idx, (feature, label) in enumerate(self.train_dataloader):
                gpu_datas = mx.gluon.utils.split_and_load(feature, self.ctxs)
                gpu_labels = mx.gluon.utils.split_and_load(label, self.ctxs)
                with mx.autograd.record():
                    # losses = [self.sec_loss(self.net(x), y) for x, y in zip(gpu_datas, gpu_labels)]
                    losses = []
                    for x, y in zip(gpu_datas, gpu_labels):
                        # logging.info(f'[train] x: {x.shape}  y: {y.shape}  {x.context}')
                        preds = self.net(x)
                        loss = self.sec_loss(preds, y)
                        # logging.info(f'loss: {loss.context}')
                        losses.append(loss)
                    cur_loss = sum([loss.sum().asscalar() for loss in losses])
                    total_loss += cur_loss
                for loss in losses:
                    loss.backward()
                self.trainer.step(self.config.batch_size)
                mx.nd.waitall()
                if idx % self.config.log_interval == 0:
                    logging.info(f'[TRAIN] [{epoch}/{idx}]  loss: {total_loss:8f} / {cur_loss:8f}')
            self.save_model(self.config.pretrained_name, epoch+1)

            eval_loss = 0
            for idx, (feature, label) in enumerate(self.eval_dataloader):
                gpu_datas = mx.gluon.utils.split_and_load(feature, self.ctxs)
                gpu_labels = mx.gluon.utils.split_and_load(label, self.ctxs)
                # losses = [self.sec_loss(self.net(x), y) for x, y in zip(gpu_datas, gpu_labels)]
                for x, y in zip(gpu_datas, gpu_labels):
                    preds = self.net(x)
                    losses.append(self.sec_loss(preds, y))
                    self.eval_acc.update(y, preds)
                cur_loss = sum([loss.sum().asscalar() for loss in losses])
                eval_loss += cur_loss
                if idx % self.config.log_interval == 0:
                    logging.info(f'[EVAL] [{epoch}/{idx}]  {self.eval_acc.get()[0]}:{self.eval_acc.get()[1]:8f}  {self.eval_acc.get()[0]}:{self.eval_acc.get()[1]:8f}  loss: {eval_loss:8f} / {cur_loss:8f}')


def TEST_TrainerWithEpoch():
    config = TrainConfig()

    MobileNetV20Transform().transform(
        num_class=config.num_class,
        prefix=config.transform_prefix,
        epoch=config.transform_epoch
    )

    trainer = TrainerWithFit(config)

    trainer.train()



if __name__ == '__main__':


    TEST_TrainerWithFit()

    TEST_TrainerWithEpoch()