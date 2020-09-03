#! /usr/bin/env python
# coding: utf-8
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from dataset.FM_dataset_v001 import FMDataIterV1
from utils.file_func import get_class_name, get_function_name
from config.config_v1 import Config
from config.FM_config_v3 import Config1
from train.trainer_v2 import Trainer1
from dataset.FM_dataset_v001 import FMDataIterV1, FMDataset, FMDataLoader, FMDataIterV1_300VW
from model.FM_net_v3 import IQ



class MultiMetric(mx.metric.EvalMetric):
    """Calculate accuracies of multi label"""

    def __init__(self, num=None):
        super(MultiMetric, self).__init__(
            name='multi-metric',
            output_names=None,  # ('kpts68-regression', 'kptsn-regression'),
            label_names=('kpts68-regression', 'kptsn-regression'),
        )
        self.num = num

    def update(self, labels, preds):
        # print(f'[MultiMetric][update] [************]  labels: {type(labels)}  preds: {type(preds)}')
        # for i in range(len(labels)):
        #     print(f'[MultiMetric][update] [************]  labels-{i}: {labels[i].shape} / {type(labels[i])}  preds-{i}: {preds[i].shape} / {type(preds[i])}')
        mx.metric.check_label_shapes(labels, preds)

        if self.num is not None:
            assert len(labels) == self.num

        for idx, (label, pred) in enumerate(zip(labels, preds)):
            print(f'[MultiMetric][update] [======]  label-{idx}: {type(label)} / {label.shape} / {label.context}')
            print(f'[MultiMetric][update] [======]  pred-{idx}: {type(pred)} / {pred.shape} / {pred.context}')
            print(f'[MultiMetric][update] [======]  pred-{idx}: {pred}')
            pred = pred.as_in_context(mx.cpu()).asnumpy()
            label = label.asnumpy()
            # print(f'[MultiMetric][update] [======]  label-{idx}: {type(label)} / {label.shape}')
            # print(f'[MultiMetric][update] [======]  pred-{idx}: {type(pred)} / {pred.shape}')

            if len(label.shape) == 1:
                label = label.reshape(label.shape[0], 1)
            if len(pred.shape) == 1:
                pred = pred.reshape(pred.shape[0], 1)

            mae = np.abs(label - pred).mean()
            self.sum_metric += mae
            self.global_sum_metric += mae
            self.num_inst += 1  # numpy.prod(label.shape)
            self.global_num_inst += 1  # numpy.prod(label.shape)

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


class LandmarkTrainerV1(Trainer1):

    def __init__(self, config):
        super(LandmarkTrainerV1, self).__init__(config=config)
        assert isinstance(config, Config), f'config file is not correct!!!'
        self.config = config
        self.model = None

        if self.config.with_fit:
            self.init_fit_data()
        else:
            self.init_model()
            self.init_epoch_data()
        self.init_trainer()
        self.init_loss()
        self.init_metric()
        self.init_optimizer(optimizer='sgd', optimizer_params=None)

    # def init_trainer(self):
    #     if not self.config.with_fit:
    #         self.trainer = mx.gluon.Trainer(
    #             self.net.collect_params(),
    #             'sgd',
    #             {'learning_rate': config.learning_rate, 'wd': config.weight_decay}
    #         )

    # def init_model(self):
    #     if not self.config.with_fit:
    #         pretrained_json = self.config.pretrained_prefix+'-symbol.json'
    #         pretrained_params = self.config.pretrained_prefix+f'-{self.config.pretrained_epoch:04d}.params'
    #         logging.info(f'loading pretraned model ... \n{pretrained_json}\n{pretrained_params}')
    #         self.net = mx.gluon.SymbolBlock.imports(pretrained_json, ['data'], pretrained_params, ctx=self.ctxs)
    #         logging.info(f'loading pretraned model complete.')

    def init_loss(self):
        if not self.config.with_fit:
            self.sec_loss = mx.gluon.loss.L2Loss()

    def init_metric(self):
        # self.train_acc = mx.metric.MAE()
        # self.eval_acc = mx.metric.MAE()
        self.train_acc = MultiMetric(2)
        self.eval_acc = MultiMetric(2)

    def init_epoch_data(self):
        self.train_dataset = FMDDataset(
            root=config.image_dir,
            flag=1,
            transform=[lambda data, label: (data.astype(np.float32) / 255, label)],
            data_shape=self.config.data_shapes,
            label_shape=self.config.label_shapes,
            data_names=self.config.data_names,
            label_names=self.config.label_names,
            batch_size=self.config.batch_size,
            ctxs=self.ctxs
        )

        self.train_dataloader = mx.gluon.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        self.eval_dataset = FMDDataset(
            root=config.image_dir,
            flag=1,
            transform=[lambda data, label: (data.astype(np.float32) / 255, label)],
            data_shape=self.config.data_shapes,
            label_shape=self.config.label_shapes,
            data_names=self.config.data_names,
            label_names=self.config.label_names,
            batch_size=self.config.batch_size
        )
        self.eval_dataloader = mx.gluon.data.DataLoader(
            dataset=self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

    def init_fit_data(self):
        transform = lambda data, label: (data.astype(np.float32) / 255.0, label)
        self.train_iter = FMDataIterV1_300VW(
            root=self.config.image_dir,
            data_shape=self.config.data_shapes,
            label_shape=self.config.label_shapes,
            data_names=self.config.data_names,
            label_names=self.config.label_names,
            flag=1,
            transform=transform,
            batch_size=self.config.batch_size
        )
        print(f'shapes:  {self.config.data_names} {self.config.data_shapes}  {self.config.label_names}  {self.config.label_shapes}')

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

    # def save_model(self, model_prefix, model_epoch=0):
    #     self.net.export(model_prefix, model_epoch)

    def train_with_fit(self):
        # sym, arg_params, aux_params = self.load_checkpoint()
        sym, arg_params, aux_params = mx.model.load_checkpoint(
            prefix=self.config.pretrained_prefix,
            epoch=self.config.pretrained_epoch
        )
        model = mx.mod.Module(
            symbol=sym,
            context=self.ctxs,
            data_names=self.config.data_names,  # ('data',),
            label_names=self.config.label_names  # ('kp68-regression', 'kpn-regression')
        )
        # logging.info(f'[0] {model.output_names}')
        # logging.info(f'[0] {model.label_names}')
        # logging.info(f'[0] {model.data_names}')
        # logging.info(f'[0] {self.train_iter.provide_data}')
        # logging.info(f'[0] {self.train_iter.provide_label}')
        # logging.info(f'[0] model.data_names: {model.data_names}')
        # logging.info(f'[0] model.label_names: {model.label_names}')
        data_shapes = self.train_iter.provide_data  # [('data', (self.config.batch_size, 3, 128, 128))]
        label_shapes = self.train_iter.provide_label  # [('softmax_label', (self.config.batch_size, ))]
        # logging.info(f'[0] data_shapes: {data_shapes}')
        # logging.info(f'[0] label_shapes: {label_shapes}')
        model.bind(
            # data_shapes=self.train_iter.provide_data,  # [('data', (self.config.batch_size, 3, 128, 128))],
            # label_shapes=self.train_iter.provide_label,  # [('softmax_label', (self.config.batch_size, self.config.num_class))],
            # data_shapes=self.train_iter.provide_data,  # [('data', (self.config.batch_size, 3, 128, 128))],
            # label_shapes=self.train_iter.provide_label  # [('softmax_label', (self.config.batch_size, self.config.num_class))],
            data_shapes=self.train_iter.provide_data,  # data_shapes,
            label_shapes=self.train_iter.provide_label  # label_shapes,
        )
        # logging.info(f'[1] {model.output_names}')
        # logging.info(f'[1] {model.label_names}')
        # logging.info(f'[1] {model.label_names}  {model.label_shapes}')
        # logging.info(f'[1] {model.data_names}  {model.data_shapes}')
        model.set_params(arg_params=arg_params, aux_params=aux_params, allow_missing=False)
        print(f'[train_with_fit] [******]  {type(self.train_acc)}')
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
            for idx, (feature, keypoints1, keypoints2) in enumerate(self.train_dataloader):
                gpu_datas = mx.gluon.utils.split_and_load(feature, self.ctxs)
                gpu_labels1 = mx.gluon.utils.split_and_load(keypoints1, self.ctxs)
                gpu_labels2 = mx.gluon.utils.split_and_load(keypoints2, self.ctxs)
                with mx.autograd.record():
                    # losses = [self.sec_loss(self.net(x), y) for x, y in zip(gpu_datas, gpu_labels)]
                    losses = []
                    for x, y1, y2 in zip(gpu_datas, gpu_labels1, gpu_labels2):
                        # logging.info(f'[train] x: {x.shape}  y: {y.shape}  {x.context}')
                        preds = self.net(x)
                        # logging.info(f'[train] x: {x.shape}  {x.context}  {x.dtype}')
                        # logging.info(f'[train] preds-1: {preds[0].shape}  {preds[0].context}  {preds[0].dtype}')
                        # logging.info(f'[train] preds-2: {preds[1].shape}  {preds[1].context}  {preds[1].dtype}')
                        # logging.info(f'[train] y1: {y1.shape}  {y1.context}  {y1.dtype}')
                        # logging.info(f'[train] y2: {y2.shape}  {y2.context}  {y2.dtype}')
                        loss1 = self.sec_loss(preds[0], y1.astype(np.float32))
                        loss2 = self.sec_loss(preds[1], y2.astype(np.float32))
                        # logging.info(f'loss: {loss.context}')
                        losses.append(loss1 + loss2)
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

    step = 3
    if step == 1:
        pass
    elif step == 2:
        pass
    else:
        trainer = LandmarkTrainerV1(config)
        trainer.train()
