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
import utils.cv_show as CVShow


def cv_save_lm_rets(datas, predi, labeli, prefix, num_count=0):
    if isinstance(datas, mx.nd.NDArray):
        datas = datas.as_in_context(mx.cpu()).asnumpy()
    if isinstance(predi, mx.nd.NDArray):
        predi = predi.as_in_context(mx.cpu()).asnumpy()
    if isinstance(labeli, mx.nd.NDArray):
        labeli = labeli.as_in_context(mx.cpu()).asnumpy()
    images = CVShow.cv_draw_batch_points(datas, predi * 64, color=(255, 0, 0))
    images = np.stack([image.get().transpose((2, 0, 1)) for image in images], axis=0)
    images = CVShow.cv_draw_batch_points(images, labeli, normalized=False, color=(0, 0, 255))
    for image in images:
        image_file = prefix + f"{num_count:06d}.jpg"
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        image = cv.resize(image, (512, 512))
        cv.imwrite(image_file, image)
        num_count += 1


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
            save_dir='../results/002',
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



if __name__ == '__main__':
    from metric.FM_metric import MultiMetric
    from dataset_v002 import RecDataIterV1
    from callback.callback import Speedometer

    show = False
    save = True
    save_dir = '../results/002'
    H, W = 64, 64
    data_process_func = lambda x, y: (x, y)
    eval_rec_dir = r'/data1/dataset/300/rec/002'
    eval_rec_prefix = f'300w_{H}X{W}'
    batch_size = 64
    shuffle = False
    ctxs = [mx.cpu()]
    dtype = 'float32'
    log_interval = 1
    epoch = 106
    sym = f'/data2/personal/MXTrainV1/checkpoints/FM/002/epochs/init-64-64-symbol.json'
    params = f'/data2/personal/MXTrainV1/checkpoints/FM/002/epochs/init-64-64-{epoch:04d}.params'

    eval_dataloader = iter(RecDataIterV1(
        rec_dir=eval_rec_dir,
        rec_prefix=eval_rec_prefix,
        batch_size=batch_size,
        shuffle=shuffle
    ))

    eval_params = FMEvalParams(
        data_process_func=data_process_func,
        log_interval=log_interval,
        metrics=[MultiMetric(name='eval-L2-68')],
        show=show,
        show_func=CVShow.cv_show_lm_rets,
        save=save,
        save_func=cv_save_lm_rets,
        save_dir=save_dir,
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
