#! /usr/bin/env python
# coding: utf-8
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
# print('system path: {}  {}'.format(sys.path, os.path.dirname(__file__)))
import numpy as np
import cv2 as cv
import mxnet as mx
from mxnet import autograd
import mxnet.ndarray as nd
from config.config_v011 import Config
# import log.logger.Logger as logger
import device_control.device_info as DeviceInfo
from dataset.mx_data_iter_v010 import RecDataIterV1
import time
from metric.mx_rmse_metric import RMSE
from metric.metrics import BaseMetric
from log.logger import Logger as logger
from loss.L2Loss import L2Loss
from network.ring_chain_net import get_mobilenet_v2


def acc(preds, targets):
    preds = preds.as_in_context(mx.cpu()).asnumpy()
    exp_ret = np.exp(preds)
    sum_ret = np.sum(exp_ret, axis=1, keepdims=True)
    softmax_predict = exp_ret /sum_ret
    preds_idx = softmax_predict.argmax(axis=1)
    correct = preds_idx == targets.as_in_context(mx.cpu()).asnumpy()
    correct_count = correct.sum()
    return correct_count, preds.shape(0)

def learning_rate_schedule(epoch, start_epoch, learning_rate):
    if learning_rate < 1e-6:
        return learning_rate
    if (epoch - start_epoch) < 3:
        return learning_rate
    if (epoch - start_epoch) % 2 == 0:
        return learning_rate * 0.1
    return learning_rate

def _reshape_like(F, x, y):
    return x.reshape(y.shape) if F is mx.ndarray else F.reshape_like(x, y)

numeric_types = (float, int, np.long, np.generic)
def _apply_weighting(F, loss, weight=None, sample_weight=None):
    if sample_weight is not None:
        loss = F.broadcast_mul(loss, sample_weight)
    if weight is not None:
        assert isinstance(weight, numeric_types), "weight must be a number"
        loss = loss * weight
    return loss

class L1Loss(mx.gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(L1Loss, self).__init__(weight, batch_axis, **kwargs)
    def hybrid_forward(self, F, pred, label, sample_weight=None):
        label = _reshape_like(F, label, pred)
        loss = F.abs(label - pred)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.sum(loss, axis=self._batch_axis, exclude=True)

def train_v1(opts):

    ## log

    ## devices
    devices = DeviceInfo.get_devices([int(i) for i in opts.processors.split(',')])
    logger.info('devices: {}'.format(devices))

    ## metric
    rmse = RMSE()
    loss_metric = [BaseMetric(name='loss')]
    loss_metric.append(BaseMetric('nose'))
    loss_metric.append(BaseMetric('delta'))
    loss_metric.append(BaseMetric('merge'))

    ## network
    # net = resnet18_v2(classes=136)
    net = get_mobilenet_v2(multiplier=1.0, classes=opts.num_classes)
    if opts.pretrained_name is not None and opts.pretrained_dir is not None:
        logger.info('loading pre-trained {} ...'.format(os.path.join(opts.pretrained_dir, opts.pretrained_name)))
        net.load_params(
            os.path.join(opts.pretrained_dir, opts.pretrained_name),
            ctx=devices,
            allow_missing=True
        )
        logger.info('load pre-trained model complete.')
    else:
        # net.hybridize()
        net.initialize(init=mx.init.Normal(sigma=0.01), force_reinit=True, ctx=devices)
        logger.info('initial params with random data.')
    net.collect_params().reset_ctx(devices)
    logger.info('net: {}'.format(net))

    ## loss
    loss_funcs = []
    # loss_funcs.append(mx.gluon.loss.SoftmaxCrossEntropyLoss())
    loss_funcs.append(mx.gluon.loss.L1Loss())
    # loss_funcs.append(L1Loss())
    # loss_funcs.append(mx.gluon.loss.L2Loss())
    # loss_funcs.append(L2Loss())

    ## optimizer
    lr_scheduler = mx.lr_scheduler.FactorScheduler(step=10, factor=0.1, base_lr=opts.learning_rate, stop_factor_lr=1e-6)

    ## trainer
    trainer = mx.gluon.Trainer(
        net.collect_params(),
        opts.optimizer,
        {
            'learning_rate': opts.learning_rate,
            # 'momentum': opts,
            'wd': opts.weight_decay
            # 'lr_scheduler': lr_scheduler
        }
    )


    ## datsets
    logger.info('loading datasets ...')
    train_iter = RecDataIterV1(
        rec_dir=opts.rec_dir,
        rec_prefix=opts.rec_prefix,
        batch_size=opts.batch_size,
        shuffle=opts.shuffle
    )
    logger.info('load datasets complete.')

    params = net.collect_params()
    param_names = params.keys()

    logger.info(
        'starting training ...',
        show_type=logger.LOG_STYLE.DEFAULT,
        forground=logger.LOG_FRONT_COLOR.RED,
        background=logger.LOG_BACK_COLOR.DEFAULT
    )
    start_time = time.time()
    global_step = 0
    for epoch in range(opts.min_epoch, opts.max_epoch):
        for idx_step, batch_data in enumerate(iter(train_iter)):
            datas = mx.gluon.utils.split_and_load(
                batch_data.data[0],
                ctx_list=devices,
                batch_axis=0
            )
            targets = mx.gluon.utils.split_and_load(
                batch_data.label[0],
                ctx_list=devices,
                batch_axis=0
            )
            losses = []
            with mx.autograd.record():
                for datai, targeti in zip(datas, targets):
                    # print('[===]  datai: {}  {}  {}'.format(datai.shape, datai.dtype, type(datai)))
                    # print('[===]  targeti: {}  {}  {}'.format(targeti.shape, targeti.dtype, type(targeti)))
                    datai = (datai - 127.5) / 128.0
                    # predicts = net(datai)
                    predicts = net(datai)
                    # logger.info('predicts: {}  {}, {}  {}'.format(
                    #     type(predicts), predicts.shape, mx.nd.max(predicts), mx.nd.min(predicts)
                    # ))
                    # print('predicts1: ', predicts[0])
                    # print('predicts2: ', predicts[1])

                    targeti = targeti / 128.0
                    # print('[===] targeti min-max: {}  {}'.format(mx.nd.max(targeti[0, :]).asnumpy()[0], mx.nd.min(targeti[0, :]).asnumpy()[0]))
                    # print('[===] predicts min-max: {}  {}'.format(mx.nd.max(predicts[0, :]).asnumpy()[0], mx.nd.min(predicts[0, :]).asnumpy()[0]))
                    # mse_loss1 = loss_funcs[0](targeti[:, 0:2], predicts[0])
                    # mse_loss2 = loss_funcs[0](targeti[:, 2:], predicts[1])
                    # print('[===] targeti: {}'.format(targeti.shape))
                    nose_points = targeti.reshape((-1, 69, 2))[:, 0, :]
                    # print('nose point: ', nose_points.shape)
                    points = targeti.reshape((-1, 69, 2))[:, 1:, :]
                    # print('point: ', points.shape)
                    mse_loss1 = loss_funcs[0](nose_points, predicts[0])
                    mse_loss2 = loss_funcs[0](points, predicts[1])

                    tar_points = targeti[:, 2:].reshape((-1, 68, 2)) + targeti[:, 0:2].reshape((-1, 1, 2))
                    tar_points = tar_points.reshape((-1, 136))
                    pred_points = predicts[0].reshape((-1, 1, 2)) + predicts[1].reshape((-1, 68, 2))
                    mse_loss3 = loss_funcs[0](tar_points, pred_points)

                    mse_loss = mse_loss1 + mse_loss2  #  + 0.5 * mse_loss3


                    if False:
                        points = targeti[:, 2:].reshape((-1, 68, 2)) + targeti[:, 0:2].reshape((-1, 1, 2))
                        points = points.asnumpy() * 128.0
                        image = datai[0, :, :, :].asnumpy() * 255.0
                        image = np.transpose(image, (1, 2, 0))
                        image = image.astype(np.uint8)
                        print('image: {}  {}'.format(image.shape, image.dtype))
                        import cv2
                        points = points[0, :, :]
                        for j in range(68):
                            point = points[j, :]
                            image = cv2.circle(
                                img=image,
                                center=(int(round(point[0])), int(round(point[1]))),
                                radius=1,
                                color=(255, 0, 0),
                                thickness=-1
                            )
                        import matplotlib.pyplot as plt
                        plt.imshow(image.get())  ## cv2.UMat --> np.array
                        # plt.imshow(image)  ## cv2.UMat --> np.array
                        plt.show()

                        sys.exit(0)

                    losses.extend(mse_loss)
                    # print('target: {}'.format(targeti.shape))
                    # print('predicts: {}'.format(predicts.shape))
                    # print('loss: {}'.format(mse_loss.shape))
                    # print('[===] {}'.format(mse_loss))

                    # correct_count, num_samples = acc(preds=predicts, targets=targeti)

                    ## update metrics
                    loss_metric[0].update(mse_loss.shape[0], mse_loss.asnumpy().sum())
                    loss_metric[1].update(mse_loss1.shape[0], mse_loss1.asnumpy().sum())
                    loss_metric[2].update(mse_loss2.shape[0], mse_loss2.asnumpy().sum())
                    loss_metric[3].update(mse_loss3.shape[0], mse_loss3.asnumpy().sum())

            mx.autograd.backward(losses)
            # for loss in losses:
            #     loss.backward()
            #     mx.autograd.backward(loss)
            trainer.step(batch_data.data[0].shape[0])
            mx.nd.waitall()

            ## log
            elapse_time = time.time() - start_time
            samples_per_second = 0
            if global_step > 30:
                start_time = time.time()
                samples_per_second = int(opts.batch_size * opts.checkpoint_interval / elapse_time)
            if (global_step + 1) % opts.log_interval == 0:
                logger.info('[{}/{}][{}/{}] [loss: {:.6f} / {:.6f}  {:.6f} / {:.6f}  {:.6f} / {:.6f}  {:.6f} / {:.6f}]  [{} samples/s]  [lr: {:.10f}]'.format(
                    epoch,
                    opts.max_epoch-opts.min_epoch,
                    idx_step,
                    int(train_iter.max_index/opts.batch_size),
                    loss_metric[0].get_avg()[0],
                    loss_metric[0].get_avg()[1],
                    loss_metric[1].get_avg()[0],
                    loss_metric[1].get_avg()[1],
                    loss_metric[2].get_avg()[0],
                    loss_metric[2].get_avg()[1],
                    loss_metric[3].get_avg()[0],
                    loss_metric[3].get_avg()[1],
                    samples_per_second,
                    lr_scheduler.base_lr
                ))

            ## update metrics

            ## update global step
            global_step += 1

        ## update trainer
        train_iter.reset()

        ## update learning rate
        lr_scheduler(epoch - opts.min_epoch)
        trainer.set_learning_rate(lr_scheduler.base_lr)
        print('lr: ', lr_scheduler.base_lr, epoch - opts.min_epoch)

        ## save checkpoint
        if (epoch + 1) % opts.checkpoint_interval == 0:
            checkpoint_path = os.path.join(
                opts.checkpoint_dir,
                opts.checkpoint_prefix+'{}'.format(epoch)
            )
            logger.info(
                'begin save checkpoints {} ...'.format(checkpoint_path),
                show_type=logger.LOG_STYLE.DEFAULT,
                forground=logger.LOG_FRONT_COLOR.RED,
                background=logger.LOG_BACK_COLOR.DEFAULT
            )
            net.save_params(filename=checkpoint_path)
            logger.info(
                'complete save checkpoints.',
                show_type=logger.LOG_STYLE.DEFAULT,
                forground=logger.LOG_FRONT_COLOR.RED,
                background=logger.LOG_BACK_COLOR.DEFAULT
            )

            ## eval
            # if (epoch + 1) % opts.eval_interval == 0:
            #     logger.info('begin evaluating ...')
            #     logger.info('complete evaluating.')














if __name__ == '__main__':

    opts = Config()
    train_v1(opts=opts)