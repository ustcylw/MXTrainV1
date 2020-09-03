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




class Optimizer(object):
    def __init__(
        self,
        trainer_type='sgd',
    ):
        self.trainer_type = trainer_type,

    def get_optimizer(self, params):
        if self.trainer_type == 'sgd':
            return mx.gluon.Trainer(
                params,
                'sgd',
                {'learning_rate': 1e-2, 'wd': 1e-5}
            )


class Trainer():

    def __init__(self, config):
        assert not isinstance(config, Config), f'config file is not correct!!!'
        self.config = config

        self.ctxs = [mx.gpu(int(i)) for i in config.gpus.split(',')]

        self.net = self.load_model_zoo(pretrained=self.config.pretrained_name, epoch=self.config.pretrained_epoch)
        self.net.collect_params().reset_ctx(self.ctxs)

        self.trainer = mx.gluon.Trainer(
            self.net.collect_params(),
            'sgd',
            {'learning_rate': config.learning_rate, 'wd': config.weight_decay}
        )

        if self.config.with_fit:
            self.train_iter = ImageFolderDataIter(
                root=self.config.image_dir,
                data_shape=(3, 128, 128),
                label_shape=(2,),
                data_names=['data'],
                label_names=['softmax_label'],
                flag=1,
                transform=lambda data, label: (data.astype(np.float32)/255, label),
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
        else:
            self.train_dataset = ImageFolderDataset(
                root=config.image_dir,
                flag=1,
                transform=lambda data, label: (data.astype(np.float32)/255, label)
            )
            self.train_dataloader = mx.gluon.data.DataLoader(
                dataset=self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )
            self.eval_dataset = ImageFolderDataset(
                root=config.image_dir,
                flag=1,
                transform=lambda data, label: (data.astype(np.float32)/255, label)
            )
            self.eval_dataloader = mx.gluon.data.DataLoader(
                dataset=self.eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )

        self.sec_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()

        self.train_acc = mx.metric.Accuracy()
        self.eval_acc = mx.metric.Accuracy()

    def load_model_zoo(self, pretrained:str=None, epoch:int=1, model_zoo:bool=True, hybridize:bool=True):
        if model_zoo:
            net = mx.gluon.model_zoo.vision.mobilenet_v2_1_0(classes=1000, pretrained=True, ctx=mx.cpu(), root=r'./model')
            net.collect_params().initialize(init=mx.init.Xavier(), ctx=[mx.cpu()], force_reinit=True)  # 多卡同时初始化
            net.output = mx.gluon.nn.HybridSequential()
            net.output.add(
                mx.gluon.nn.Conv2D(kernel_size=1, channels=self.config.num_class, strides=1),
                mx.gluon.nn.Flatten()
            )
            net.output.initialize(ctx=[mx.cpu()])

            if hybridize:
                net.hybridize()
            x = mx.nd.uniform(shape=(1, 3, 128, 128), ctx=(mx.cpu()))
            y = net(x)
            logging.info(f'[{get_class_name(self)}:{get_function_name()}]  x: {x.shape}  y: {y.shape}')
            logging.info(f'[{get_class_name(self)}:{get_function_name()}]  export model-zoo model ...')
            net.export(pretrained, 0)
            logging.info(f'[{get_class_name(self)}:{get_function_name()}]  export model-zoo model complete.')

        net = mx.gluon.SymbolBlock.imports(pretrained+'-symbol.json', ['data'], pretrained+f'-{epoch:04d}.params', ctx=mx.cpu())
        logging.info(f'[{get_class_name(self)}:{get_function_name()}]  net: {net}')
        logging.info(f'[{get_class_name(self)}:{get_function_name()}]  {net.summary(mx.nd.uniform(shape=(1,3,128,128)))}')
        if hybridize:
            net.hybridize()
        x = mx.nd.uniform(shape=(1, 3, 128, 128), ctx=(mx.cpu()))
        y = net(x)
        logging.info(f'[{get_class_name(self)}:{get_function_name()}]  x: {x.shape}  y: {y.shape}')
        return net

    def load_checkpoint(self):
        sym, arg_params, aux_params = mx.model.load_checkpoint(
            prefix=self.config.pretrained_name,
            epoch=self.config.pretrained_epoch
        )
        softmax_sym = mx.sym.Softmax(data=sym, name='softmax')
        logging.info(f'[train_with_fit] sym: {sym}')
        logging.info(f'[train_with_fit] softmax_sym: {softmax_sym}')
        return softmax_sym, arg_params, aux_params

    def save_model(self, model_prefix, model_epoch=0):
        self.net.export(model_prefix, model_epoch)

    def train_with_fit(self):
        sym, arg_params, aux_params = self.load_checkpoint()
        model = mx.mod.Module(
            symbol=sym,
            context=self.ctxs,
        )
        logging.info(f'[0] {model.output_names}')
        logging.info(f'[0] {model.label_names}')
        logging.info(f'[0] {model.data_names}')
        logging.info(f'[0] {self.train_iter.provide_data}')
        logging.info(f'[0] {self.train_iter.provide_label}')
        logging.info(f'[0] model.data_names: {model.data_names}')
        logging.info(f'[0] model.label_names: {model.label_names}')
        data_shapes = [('data', (self.config.batch_size, 3, 128, 128))]
        label_shapes = [('softmax_label', (self.config.batch_size, ))]
        logging.info(f'[0] data_shapes: {data_shapes}')
        logging.info(f'[0] label_shapes: {label_shapes}')
        model.bind(
            # data_shapes=self.train_iter.provide_data,  # [('data', (self.config.batch_size, 3, 128, 128))],
            # label_shapes=self.train_iter.provide_label,  # [('softmax_label', (self.config.batch_size, self.config.num_class))],
            # data_shapes=self.train_iter.provide_data,  # [('data', (self.config.batch_size, 3, 128, 128))],
            # label_shapes=self.train_iter.provide_label  # [('softmax_label', (self.config.batch_size, self.config.num_class))],
            data_shapes=data_shapes,
            label_shapes=label_shapes,
        )
        logging.info(f'[1] {model.output_names}')
        logging.info(f'[1] {model.label_names}')
        logging.info(f'[1] {model.label_names}  {model.label_shapes}')
        logging.info(f'[1] {model.data_names}  {model.data_shapes}')
        model.set_params(arg_params=arg_params, aux_params=aux_params, allow_missing=False)
        model.fit(
            train_data=self.train_iter,
            optimizer='sgd',
            optimizer_params={'learning_rate': self.config.learning_rate, 'wd': self.config.weight_decay},
            num_epoch=self.config.max_epoch,
            batch_end_callback=[
                mx.callback.Speedometer(self.config.batch_size, 1),
            ],
            epoch_end_callback=mx.callback.do_checkpoint(self.config.pretrained_name, 1),
            eval_metric='acc'
        )

    def train(self):
        if self.config.with_fit:
            self.train_with_fit()
        else:
            self.train_epoch()

    def train_epoch(self):

        for epoch in range(self.config.min_epoch, self.config.max_epoch):
            total_loss = 0
            for idx, (feature, label) in enumerate(self.train_dataloader):
                gpu_datas = mx.gluon.utils.split_and_load(feature, self.ctxs)
                gpu_labels = mx.gluon.utils.split_and_load(label, self.ctxs)
                with mx.autograd.record():
                    # losses = [self.sec_loss(self.net(x), y) for x, y in zip(gpu_datas, gpu_labels)]
                    losses = []
                    for x, y in zip(gpu_datas, gpu_labels):
                        logging.info(f'[train] x: {x.shape}  y: {y.shape}  {x.context}')
                        preds = self.net(x)
                        loss = self.sec_loss(preds, y)
                        logging.info(f'loss: {loss.context}')
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

    def eval(self):
        pass


class Trainer1(object):

    def __init__(self, config):
        assert isinstance(config, Config), f'config file is not correct!!!'
        self.config = config
        self.model = None

        if len(self.config.gpus) <= 0:
            self.ctxs = [mx.cpu()]
        else:
            self.ctxs = [mx.gpu(int(i)) for i in config.gpus.split(',')]

        if self.config.with_fit:
            self.init_fit_data()
        else:
            self.init_model()
            self.init_epoch_data()
        self.init_trainer()
        self.init_loss()
        self.init_metric()
        self.init_optimizer(optimizer='sgd', optimizer_params=None)

    def init_trainer(self):
        if not self.config.with_fit:
            self.trainer = mx.gluon.Trainer(
                self.net.collect_params(),
                'sgd',
                {'learning_rate': self.config.learning_rate, 'wd': self.config.weight_decay}
            )

    def init_model(self):
        if not self.config.with_fit:
            pretrained_json = self.config.pretrained_prefix+'-symbol.json'
            pretrained_params = self.config.pretrained_prefix+f'-{self.config.pretrained_epoch:04d}.params'
            logging.info(f'loading pretraned model ... \n{pretrained_json}\n{pretrained_params}')
            self.net = mx.gluon.SymbolBlock.imports(pretrained_json, ['data'], pretrained_params, ctx=self.ctxs)
            logging.info(f'loading pretraned model complete.')

    def init_loss(self):
        self.sec_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()

    def init_metric(self):
        self.train_acc = mx.metric.Accuracy()
        self.eval_acc = mx.metric.Accuracy()

    def init_epoch_data(self):
        self.train_dataset = ImageFolderDataset(
            root=config.image_dir,
            flag=1,
            transform=lambda data, label: (data.astype(np.float32) / 255, label)
        )
        self.train_dataloader = mx.gluon.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        self.eval_dataset = ImageFolderDataset(
            root=config.image_dir,
            flag=1,
            transform=lambda data, label: (data.astype(np.float32) / 255, label)
        )
        self.eval_dataloader = mx.gluon.data.DataLoader(
            dataset=self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

    def init_fit_data(self):
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
        if self.config.with_fit:
            self.optimizer = optimizer
            self.optimizer_params = optimizer_params if optimizer_params is not None else {'learning_rate': self.config.learning_rate, 'wd': self.config.weight_decay}

    def save_model(self, model_prefix, model_epoch=0):
        logging.info(f'saving model ... {model_prefix} -- {model_epoch}')
        self.net.export(model_prefix, model_epoch)
        logging.info(f'save model complete.')

    def train_with_fit(self):
        # sym, arg_params, aux_params = self.load_checkpoint()
        sym, arg_params, aux_params = mx.model.load_checkpoint(
            prefix=self.config.pretrained_prefix,
            epoch=self.config.pretrained_epoch
        )
        model = mx.mod.Module(
            symbol=sym,
            context=self.ctxs,
        )
        logging.info(f'[0] {model.output_names}')
        logging.info(f'[0] {model.label_names}')
        logging.info(f'[0] {model.data_names}')
        logging.info(f'[0] {self.train_iter.provide_data}')
        logging.info(f'[0] {self.train_iter.provide_label}')
        logging.info(f'[0] model.data_names: {model.data_names}')
        logging.info(f'[0] model.label_names: {model.label_names}')
        data_shapes = [('data', (self.config.batch_size, 3, 128, 128))]
        label_shapes = [('softmax_label', (self.config.batch_size, ))]
        logging.info(f'[0] data_shapes: {data_shapes}')
        logging.info(f'[0] label_shapes: {label_shapes}')
        model.bind(
            # data_shapes=self.train_iter.provide_data,  # [('data', (self.config.batch_size, 3, 128, 128))],
            # label_shapes=self.train_iter.provide_label,  # [('softmax_label', (self.config.batch_size, self.config.num_class))],
            # data_shapes=self.train_iter.provide_data,  # [('data', (self.config.batch_size, 3, 128, 128))],
            # label_shapes=self.train_iter.provide_label  # [('softmax_label', (self.config.batch_size, self.config.num_class))],
            data_shapes=data_shapes,
            label_shapes=label_shapes,
        )
        logging.info(f'[1] {model.output_names}')
        logging.info(f'[1] {model.label_names}')
        logging.info(f'[1] {model.label_names}  {model.label_shapes}')
        logging.info(f'[1] {model.data_names}  {model.data_shapes}')
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

    def train(self):
        if self.config.with_fit:
            self.train_with_fit()
        else:
            self.train_epoch()

    def train_epoch(self):

        for epoch in range(self.config.min_epoch, self.config.max_epoch):
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

    def train_with_excutor(self):
        pass

    def eval(self):
        pass











if __name__ == '__main__':


    config = Config1()

    MobileNetV20Transform().transform(
        num_class=config.num_class,
        prefix=config.transform_prefix,
        epoch=config.transform_epoch
    )



    trainer = Trainer1(config)

    trainer.train()