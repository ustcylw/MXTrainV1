#! /usr/bin/env python
# coding: utf-8
import os, sys
import numpy as np
import cv2 as cv
import  mxnet as mx
from mxnet.model import BatchEndParam
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from logger.logger_v1 import LogHandler
logging = LogHandler()

class EvalParams(object):
    def __init__(
            self,
            image_list,
            data_process_func,
            label_list=None,
            log_interval=1,
            metrics=None,
            show=True,
            show_func=None,
            save=True,
            save_func=None,
            save_dir='../results',
            dataloader=None,
            batch_size=1,
            batch_end_callback_list=None,
            ctxs=[mx.cpu()],
            dtype='float32'
    ):
        self.image_list=image_list
        self.data_process_func = data_process_func
        self.label_list = label_list
        self.log_interval = log_interval
        self.metrics = metrics
        self.show = show
        self.show_func = show_func
        self.save = save
        self.save_func = save_func
        self.save_dir = save_dir
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.batch_end_callback_list = batch_end_callback_list
        self.ctxs = ctxs
        self.dtype = dtype

    def reset_metrics(self):
        if self.metrics is not None:
            for metric in self.metrics:
                metric.reset()


class Evaluator(object):
    def __init__(self):
        self.net = None

    def set_model(self, net):
        self.net = net

    def eval(self, eval_params):
        # self.eval_with_dataloader(eval_params)
        raise NotImplementedError('not implemented!!!')

    def eval_with_image_folder(self, eval_params):
        image_list = eval_params.image_list
        data_process_func = eval_params.data_process_func
        label_list = eval_params.lable_list
        log_interval = eval_params.log_interval
        metrics = eval_params.metrics
        show = eval_params.show
        show_func = eval_params.show_func
        save = eval_params.save
        save_func = eval_params.save_func

        callback = mx.callback.Speedometer(1, log_interval)
        num_samples = len(image_list)
        for idx, image_file in enumerate(image_list):
            if label_list is not None:
                label_file = label_list[idx]
            else:
                label_file = None
            datas, labels = data_process_func(image=image_file, label=label_file)
            preds = self.net(datas)

            self.update_metrics(preds, labels, metrics)

            callback_params = BatchEndParam(
                epoch=1,
                nbatch=1,
                eval_metric=metrics,
                locals=locals()
            )
            callback(callback_params)

            if show:
                show_func(image_file, datas, preds, labels)
            if save:
                save_func(image_file, datas, preds, labels)

    def eval_with_dataloader(self, eval_params):
        data_process_func = eval_params.data_process_func
        dataloader = eval_params.dataloader
        metrics = eval_params.metrics
        log_interval = eval_params.log_interval
        batch_size = eval_params.batch_size
        batch_end_callback_list = eval_params.batch_end_callback_list
        show = eval_params.show
        show_func = eval_params.show_func
        save = eval_params.save
        save_func = eval_params.save_func
        save_dir = eval_params.save_dir
        ctxs = eval_params.ctxs
        dtype = eval_params.dtype

        dataloader.reset()

        if batch_end_callback_list is None:
            batch_end_callback_list = [mx.callback.Speedometer(batch_size, log_interval)]

        # for nbatch, (datas, labels) in enumerate(dataloader):
        for nbatch, databatch in enumerate(dataloader):
            datas, labels = databatch.data[0], databatch.label[0]

            self.net.cast(dtype)
            gpu_datas = mx.gluon.utils.split_and_load(datas, ctxs)
            gpu_labels = mx.gluon.utils.split_and_load(labels, ctxs)

            for datai, labeli in zip(gpu_datas, gpu_labels):
                datai, labeli = data_process_func(datai, labeli)
                datai = datai.astype(dtype)
                predi = self.forward(datai)

                self.update_metrics(predi, labeli, metrics)

                if show:
                    show_func(datas, predi, labeli)
                if save:
                    save_func(
                        datas,
                        predi,
                        labeli,
                        os.path.join(save_dir, 'eval'),
                        num_count=nbatch * datas.shape[0]
                    )

            for batch_end_callback in batch_end_callback_list:
                batch_end_params = BatchEndParam(
                    epoch=1,
                    nbatch=nbatch,
                    eval_metric=metrics,
                    locals=locals()
                )
                batch_end_callback(batch_end_params)

    def forward(self, datas):
        return self.net(datas)

    def compute_loss(self, preds, labels, loss_funcs):
        return [loss_func(labels, preds) for loss_func in loss_funcs]

    def update_metrics(self, preds, labels, metrics):
        if metrics is not None:
            for metric in metrics:
                metric.update(labels, preds)
