#! /usr/bin/env python
# coding: utf-8
import os, sys
import mxnet as mx



class L2Loss(mx.gluon.loss.Loss):
    def __init__(self, weight=1., batch_axis=0, **kwargs):
        super(L2Loss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        # label = _reshape_like(F, label, pred)
        loss = F.square(label - pred)
        # loss = _apply_weighting(F, loss, self._weight / 2, sample_weight)
        # return F.mean(loss, axis=self._batch_axis, exclude=True)
        return F.sum(loss, axis=self._batch_axis, exclude=True)

