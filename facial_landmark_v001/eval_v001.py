#! /usr/bin/env python
# coding: utf-8
import os, sys
import numpy as np
import cv2 as cv
import  mxnet as mx
from mxnet.model import BatchEndParam
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from eval.evaluator import EvalParams, Evaluator
from logger.logger_v1 import LogHandler
logging = LogHandler()


class FMEvalParams(EvalParams):
    def __init__(
            self,
            data_process_func,
            image_list=None,
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
        super(FMEvalParams, self).__init__(
            image_list=image_list,
            data_process_func=data_process_func,
            label_list=label_list,
            log_interval=log_interval,
            metrics=metrics,
            show=show,
            show_func=show_func,
            save=save,
            save_func=save_func,
            save_dir=save_dir,
            dataloader=dataloader,
            batch_size=batch_size,
            batch_end_callback_list=batch_end_callback_list,
            ctxs=ctxs,
            dtype=dtype
        )


class FMEvaluator(Evaluator):
    def __init__(self):
        self.net = None

    def set_model(self, net):
        self.net = net

    def eval(self, eval_params):
        self.eval_with_dataloader(eval_params)

    # def eval_with_image_folder(self, eval_params):
    #     image_list = eval_params.image_list
    #     data_process_func = eval_params.data_process_func
    #     label_list = eval_params.lable_list
    #     log_interval = eval_params.log_interval
    #     metrics = eval_params.metrics
    #     show = eval_params.show
    #     show_func = eval_params.show_func
    #     save = eval_params.save
    #     save_func = eval_params.save_func
    #
    #     callback = mx.callback.Speedometer(1, log_interval)
    #     num_samples = len(image_list)
    #     for idx, image_file in enumerate(image_list):
    #         if label_list is not None:
    #             label_file = label_list[idx]
    #         else:
    #             label_file = None
    #         datas, labels = data_process_func(image=image_file, label=label_file)
    #         preds = self.net(datas)
    #
    #         self.update_metrics(preds, labels, metrics)
    #
    #         callback_params = BatchEndParam(
    #             epoch=1,
    #             nbatch=1,
    #             eval_metric=metrics,
    #             locals=locals()
    #         )
    #         callback(callback_params)
    #
    #         if show:
    #             show_func(image_file, datas, preds, labels)
    #         if save:
    #             save_func(image_file, datas, preds, labels)
    #
    # def eval_with_dataloader(self, eval_params):
    #     dataloader = eval_params.dataloader
    #     metrics = eval_params.metrics
    #     log_interval = eval_params.log_interval
    #     batch_size = eval_params.batch_size
    #     batch_end_callback_list = eval_params.batch_end_callback_list
    #     show = eval_params.show
    #     show_func = eval_params.show_func
    #     save = eval_params.save
    #     save_func = eval_params.save_func
    #
    #     if batch_end_callback_list is None:
    #         batch_end_callback_list = [mx.callback.Speedometer(batch_size, log_interval)]
    #
    #     for nbatch, (datas, labels) in enumerate(dataloader):
    #
    #         preds = self.forward(datas)
    #
    #         self.update_metrics(preds, labels, metrics)
    #
    #         for batch_end_callback in batch_end_callback_list:
    #             batch_end_params = BatchEndParam(
    #                 epoch=1,
    #                 nbatch=nbatch,
    #                 eval_metric=metrics,
    #                 locals=locals()
    #             )
    #             batch_end_callback(batch_end_params)
    #
    #         if show:
    #             show_func(datas, preds, labels)
    #         if save:
    #             save_func(datas, preds, labels)
    #
    # def forward(self, datas):
    #     return self.net(datas)
    #
    # def compute_loss(self, preds, labels, loss_funcs):
    #     return [loss_func(labels, preds) for loss_func in loss_funcs]
    #
    # def update_metrics(self, preds, labels, metrics):
    #     if metrics is not None:
    #         for metric in metrics:
    #             metric.update(labels, preds)


if __name__ == '__main__':
    from metric.FM_metric import MultiMetric
    from utils.cv_show import cv_show_lm_rets
    from utils.cv_save import cv_save_lm_rets
    from dataset_v001 import RecDataIterV1
    from callback.callback import Speedometer

    show = False
    save = True
    H, W = 128, 128
    data_process_func = lambda x, y: (x, y)
    eval_rec_dir = r'/data1/dataset/300/rec/002'
    eval_rec_prefix = f'300w_{H}X{W}'
    batch_size = 64
    shuffle = False
    ctxs = [mx.cpu()]
    dtype = 'float32'
    log_interval = 1
    epoch = 75
    sym = f'/data2/personal/MXTrainV1/checkpoints/FM/001/epochs/init-128-128-symbol.json'
    params = f'/data2/personal/MXTrainV1/checkpoints/FM/001/epochs/init-128-128-{epoch:04d}.params'

    eval_dataloader = iter(RecDataIterV1(
        rec_dir=eval_rec_dir,
        rec_prefix=eval_rec_prefix,
        batch_size=batch_size,
        shuffle=shuffle
    ))

    eval_params = FMEvalParams(
        data_process_func=data_process_func,
        log_interval=1,
        metrics=[MultiMetric(name='eval-L2-68')],
        show=show,
        show_func=cv_show_lm_rets,
        save=save,
        save_func=cv_save_lm_rets,
        save_dir='../results/001',
        dataloader=eval_dataloader,
        batch_size=batch_size,
        batch_end_callback_list=[Speedometer(batch_size, log_interval)],
        ctxs=ctxs,
        dtype=dtype
    )

    model = mx.gluon.SymbolBlock.imports(
        sym,
        ['data'],
        params
    )
    evaluator = FMEvaluator()
    evaluator.set_model(model)
    evaluator.eval(eval_params)
