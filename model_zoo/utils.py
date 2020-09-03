#! /usr/bin/env python
# coding: utf-8
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mxnet as mx
import symbol

from model_zoo.base_model_zoo import BaseModelZooTransform
from utils.file_func import get_class_name, get_function_name
from logger.logger_v4 import Log
from logger.logger_v1 import LogHandler
logging = LogHandler()


__all__ = [
    'TransformGluon2Sym'
]


def TransformGluon2Sym(prefix, epoch, weights=[1.0, 1.0], grad_scale=1.0, data_shape=(3, 128, 128), num_keypoints=1, name='regression', **kwargs):
    sym, arg_params, aux_params = mx.model.load_checkpoint(
        prefix=prefix,
        epoch=epoch
    )
    # weight_variables = [mx.sym.Variable(name=f'loss{i}_weight') for i, val in enumerate(weights)]
    keypoints_label_68 = mx.sym.Variable(name='kpts68-'+name)# * weight_variables[0]
    keypoints_label_n = mx.sym.Variable(name='kptsn-'+name)# * weight_variables[1]
    # loss1 = mx.sym.LinearRegressionOutput(data=sym[0], label=keypoints_label_68, grad_scale=grad_scale, name='kpts68-regression')
    # loss2 = mx.sym.LinearRegressionOutput(data=sym[1], label=keypoints_label_n, grad_scale=grad_scale, name='kptsn-regression')
    # loss1 = mx.sym.LogisticRegressionOutput(data=sym[0], label=keypoints_label_68, grad_scale=grad_scale, name='kpts68-regression')
    # loss2 = mx.sym.LogisticRegressionOutput(data=sym[1], label=keypoints_label_n, grad_scale=grad_scale, name='kptsn-regression')
    loss1 = mx.sym.MAERegressionOutput(data=sym[0], label=keypoints_label_68, grad_scale=grad_scale, name='kpts68-regression-loss')
    loss2 = mx.sym.MAERegressionOutput(data=sym[1], label=keypoints_label_n, grad_scale=grad_scale, name='kptsn-regression-loss')
    loss = mx.sym.Group([loss1, loss2])
    logging.info(f'[TransformGluon2Sym] sym: {sym}')
    logging.info(f'[TransformGluon2Sym] loss: {loss}  {loss1}  {loss2}')

    # mx.model.save_checkpoint(prefix + '-sym', epoch, loss, arg_params, aux_params)

    inputs = ['data']
    inputs = mx.sym.var('data')
    net = mx.gluon.SymbolBlock(loss, inputs=inputs)
    net.initialize()
    net.hybridize()
    y = net.forward(mx.nd.uniform(shape=(2, 3, data_shape[1], data_shape[2])))
    print(f'[******] y: {len(y)}')
    for i in range(len(y)):
        print(f'[******] y-{i}: {y[i].shape}')
    net.export(prefix + '-sym', 0)

    sym, _, _ = mx.model.load_checkpoint(prefix=prefix + '-sym', epoch=epoch)
    digraph = mx.viz.plot_network(sym, title=prefix+'-sym-v3', save_format='pdf', shape={'data':(1, 3, data_shape[1], data_shape[2])})
    digraph.view()
