# -*- coding: utf-8 -*-
import os
import sys
import time
root_dir = os.path.dirname(os.path.dirname(__file__))
print(f'root_dir: {root_dir}')
sys.path.insert(0, root_dir)
sys.path.insert(0, r'D:\personal\code\tf2.0\FacialLandmark')
from logger.logger_v1 import LogHandler
logging = LogHandler()


class CallBack(object):

    def __init__(self, name='defaul-callback'):
        self.name = name

    def __call__(self, cfg, dataset, train_brick, *args, **kwargs):
        # Logger.debug('should not reach here!!!', show_type=1, forground=31, background=0)
        logging.error('should not reach here!!!')
        raise NotImplementedError


class CheckpointCallback(CallBack):
    """Logs training speed and evaluation metrics periodically.

    Parameters
    ----------
    batch_size: int
        Batch size of data.
    frequent: int
        Specifies how frequently training speed and evaluation metrics
        must be logged. Default behavior is to log once every 50 batches.
    auto_reset : bool
        Reset the evaluation metrics after each log.

    Example
    -------
    >>> # Print training speed and evaluation metrics every ten batches. Batch size is one.
    >>> module.fit(iterator, num_epoch=n_epoch,
    ... batch_end_callback=mx.callback.Speedometer(1, 10))
    Epoch[0] Batch [10] Speed: 1910.41 samples/sec  Train-accuracy=0.200000
    Epoch[0] Batch [20] Speed: 1764.83 samples/sec  Train-accuracy=0.400000
    Epoch[0] Batch [30] Speed: 1740.59 samples/sec  Train-accuracy=0.500000
    """
    def __init__(self, name='CheckpointCallback'):
        super(CheckpointCallback, self).__init__(name=name)
        self.tic = 0

    def __call__(self, model, prefix, epoch, *args, **kwargs):
        logging.info(f'saving model ... {prefix} -- {epoch}')
        model.export(prefix, epoch)
        logging.info(f'save model complete.')


class Speedometer(CallBack):
    """Logs training speed and evaluation metrics periodically.

    Parameters
    ----------
    batch_size: int
        Batch size of data.
    frequent: int
        Specifies how frequently training speed and evaluation metrics
        must be logged. Default behavior is to log once every 50 batches.
    auto_reset : bool
        Reset the evaluation metrics after each log.

    Example
    -------
    >>> # Print training speed and evaluation metrics every ten batches. Batch size is one.
    >>> module.fit(iterator, num_epoch=n_epoch,
    ... batch_end_callback=mx.callback.Speedometer(1, 10))
    Epoch[0] Batch [10] Speed: 1910.41 samples/sec  Train-accuracy=0.200000
    Epoch[0] Batch [20] Speed: 1764.83 samples/sec  Train-accuracy=0.400000
    Epoch[0] Batch [30] Speed: 1740.59 samples/sec  Train-accuracy=0.500000
    """
    def __init__(self, batch_size, frequent=50, auto_reset=True):
        super(Speedometer, self).__init__(name='speedometer')
        self.batch_size = batch_size
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0
        self.auto_reset = auto_reset

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                # #11504
                try:
                    speed = self.frequent * self.batch_size / (time.time() - self.tic)
                except ZeroDivisionError:
                    speed = float('inf')
                # print(f'[Speedometer]  name-value:  {param.eval_metric}  {type(param.eval_metric)}')
                if param.eval_metric is not None:
                    name_value = []
                    for metric in param.eval_metric:
                        name_value.extend(metric.get_name_value())
                    if self.auto_reset:
                        for metric in param.eval_metric:
                            metric.reset_local()
                        msg = 'Epoch[%d] Batch [%d-%d]\tSpeed: %.2f samples/sec'
                        msg += '\t%s=%f'*len(name_value)
                        logging.info(msg, param.epoch, count-self.frequent, count, speed, *sum(name_value, ()))
                    else:
                        msg = 'Epoch[%d] Batch [0-%d]\tSpeed: %.2f samples/sec'
                        msg += '\t%s=%f'*len(name_value)
                        logging.info(msg, param.epoch, count, speed, *sum(name_value, ()))
                else:
                    logging.info("Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec", param.epoch, count, speed)
                self.tic = time.time()
        else:
            # print(f'[3]  init: {self.init}  {self.last_count}  {count}  {self.frequent}')
            self.init = True
            self.tic = time.time()


if __name__ == '__main__':

    import mxnet as mx
    from mxnet.model import BatchEndParam

    acc = mx.metric.Accuracy()
    metric = mx.metric.MSE()
    params = BatchEndParam(
        epoch=0,
        nbatch=1,
        eval_metric=acc,
        locals=locals()
    )

    batch_size = 2
    acc.update(labels=mx.nd.uniform(shape=(batch_size, 512)), preds=mx.nd.uniform(shape=(batch_size, 512)))

    cb = Speedometer(batch_size=batch_size, frequent=1)
    cb(params)
    cb(params)
    cb(params)
