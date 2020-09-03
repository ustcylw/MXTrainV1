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
    'MobileNetV20Transform'
]


class MobileNetV20Transform(BaseModelZooTransform):
    def __init__(self, name='MobileNetV20Transform', logger=LogHandler()):
        super(MobileNetV20Transform, self).__init__(name)
        self.name = name
        self.logging = logger

    def transform(self, num_class, prefix, epoch=0):
        self.transform2gluon(num_class, prefix, epoch)
        self.transform2sym(prefix, epoch)

    def transform2gluon(self, num_class, prefix, epoch):
        net = mx.gluon.model_zoo.vision.mobilenet_v2_1_0(classes=1000, pretrained=True, ctx=mx.cpu(), root=r'./model')
        net.collect_params().initialize(init=mx.init.Xavier(), ctx=[mx.cpu()], force_reinit=True)  # 多卡同时初始化
        net.output = mx.gluon.nn.HybridSequential()
        net.output.add(
            mx.gluon.nn.Conv2D(kernel_size=1, channels=num_class, strides=1),
            mx.gluon.nn.Flatten()
        )
        net.output.initialize(ctx=[mx.cpu()])

        net.hybridize()
        x = mx.nd.uniform(shape=(1, 3, 128, 128), ctx=(mx.cpu()))
        y = net(x)
        self.logging.info(f'[{get_class_name(self)}:{get_function_name()}]  x: {x.shape}  y: {y.shape}')
        self.logging.info(f'[{get_class_name(self)}:{get_function_name()}]  export model-zoo model ...')
        net.export(prefix, 0)
        self.logging.info(f'[{get_class_name(self)}:{get_function_name()}]  export model-zoo model complete.')

        net = MobileNetV20Transform().imports(prefix=prefix, epoch=epoch, input_names=['data'], ctx=mx.cpu())
        self.logging.info(f'[{get_class_name(self)}:{get_function_name()}]  net: {net}')
        self.logging.info(f'[{get_class_name(self)}:{get_function_name()}]  {net.summary(mx.nd.uniform(shape=(1, 3, 128, 128)))}')
        net.hybridize()
        x = mx.nd.uniform(shape=(1, 3, 128, 128), ctx=(mx.cpu()))
        y = net(x)
        self.logging.info(f'[{get_class_name(self)}:{get_function_name()}]  x: {x.shape}  y: {y.shape}')
        return net

    def transform2sym(self, prefix, epoch, remove_amp_cast=True):
        sym, arg_params, aux_params = mx.model.load_checkpoint(
            prefix=prefix,
            epoch=epoch
        )
        softmax_sym = mx.sym.Softmax(data=sym, name='softmax')
        self.logging.info(f'[train_with_fit] sym: {sym}')
        self.logging.info(f'[train_with_fit] softmax_sym: {softmax_sym}')

        mx.model.save_checkpoint(prefix+'-sym', epoch, softmax_sym, arg_params, aux_params, remove_amp_cast=remove_amp_cast)

        # softmax_sym.save('%s-symbol.json' % prefix, remove_amp_cast=remove_amp_cast)
        #
        # save_dict = {('arg:%s' % k): v.as_in_context(mx.cpu()) for k, v in arg_params.items()}
        # save_dict.update({('aux:%s' % k): v.as_in_context(mx.cpu()) for k, v in aux_params.items()})
        # param_name = '%s-%04d.params' % (prefix, epoch)
        # mx.nd.save(param_name, save_dict)
        # self.logging.info('Saved checkpoint to \"%s\"', param_name)
