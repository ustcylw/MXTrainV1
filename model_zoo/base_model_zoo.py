#! /usr/bin/env python
# coding: utf-8
import os, sys
import numpy as np
import mxnet as mx
from logger.logger_v4 import Log




class BaseModelZooTransform(object):
    def __init__(self, name='default', logger=Log()):
        self.name = name
        self.logging = logger

    def transform(self, prefix, epoch):
        raise NotImplementedError

    def imports(self, prefix, epoch, input_names, ctx=None):
        return mx.gluon.SymbolBlock.imports(
            prefix + '-symbol.json',
            input_names,
            prefix + f'-{epoch:04d}.params',
            ctx=mx.cpu() if ctx is None else ctx
        )

    def export(self, net, prefix, epoch):
        net.export(prefix, epoch)

    def load_checkpoints(self, prefix, epoch):
        return mx.model.load_checkpoint(
            prefix=prefix,
            epoch=epoch
        )