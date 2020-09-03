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


class MultiMetric(mx.metric.EvalMetric):
    def __init__(self, num=None, name='multi-metric'):
        super(MultiMetric, self).__init__(
            name=name,
            output_names=None,  # ('kpts68-regression', 'kptsn-regression'),
            label_names=('kpts68-regression', 'kptsn-regression'),
        )
        self.num = num

    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)

        if self.num is not None:
            assert len(labels) == self.num

        for idx, (label, pred) in enumerate(zip(labels, preds)):
            pred = pred.as_in_context(mx.cpu()).asnumpy()
            label = label.asnumpy()

            if len(label.shape) == 1:
                label = label.reshape(label.shape[0], 1)
            if len(pred.shape) == 1:
                pred = pred.reshape(pred.shape[0], 1)

            mae = np.abs(label / 128 - pred).mean()
            self.sum_metric += mae
            self.global_sum_metric += mae
            self.num_inst += 1  # numpy.prod(label.shape)
            self.global_num_inst += 1  # numpy.prod(label.shape)


class KeypointsMultiMetric(mx.metric.EvalMetric):
    def __init__(self, num=None, name='multi-metric'):
        super(KeypointsMultiMetric, self).__init__(
            name=name,
            output_names=None,  # ('kpts68-regression', 'kptsn-regression'),
            label_names=('kpts68-regression', 'kptsn-regression'),
        )
        self.num = num

    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)

        if self.num is not None:
            assert len(labels) == self.num

        for idx, (label, pred) in enumerate(zip(labels, preds)):
            pred = pred.as_in_context(mx.cpu()).asnumpy()
            label = label.asnumpy()
            pred = pred.reshape((-1, 68, 2))
            label = label.reshape((-1, 5, 2))

            if len(label.shape) == 1:
                label = label.reshape(label.shape[0], 1)
            if len(pred.shape) == 1:
                pred = pred.reshape(pred.shape[0], 1)

            mae = np.abs(label / 128 - pred).mean()
            self.sum_metric += mae
            self.global_sum_metric += mae
            self.num_inst += 1  # numpy.prod(label.shape)
            self.global_num_inst += 1  # numpy.prod(label.shape)
