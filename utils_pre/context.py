#! /usr/bin/env python
# coding: utf-8
import os, sys
import numpy as np
import mxnet as mx


__all__ = [
    'get_context', 'try_gpu',
    'TEST'
]


def get_context(gpus=[]):
    gpu_count = mx.util.get_gpu_count()
    aval_gpu_count = []
    if len(gpus) > 0:
        aval_gpu_count = [i for i in gpus if i in range(gpu_count)]
    if len(aval_gpu_count) > 0:
        gpu_memorys = [mx.util.get_gpu_memory(gpu_dev_id=gpu_id) for gpu_id in aval_gpu_count]
        contexts = [mx.gpu(device_id) for device_id in aval_gpu_count]
    else:
        gpu_memorys = [-1]
        contexts = [mx.cpu()]
    return contexts, gpu_memorys

def TEST_get_context():
    gpus = []
    ctxs, mems = get_context(gpus)
    print(f'[TEST_get_context] ctxs: {ctxs}  mems: {mems}')
    gpus = [0, 1]
    ctxs, mems = get_context(gpus)
    print(f'[TEST_get_context] ctxs: {ctxs}  mems: {mems}')


def try_gpu(gpu_id=0):
    try:
        ctx = mx.gpu(gpu_id)
        _ = mx.nd.zeros((1,), ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx

def TEST_try_gpu():
    ctx = try_gpu(0)
    print(f'[] ctx: {ctx}')

    ctx = try_gpu(3)
    print(f'[] ctx: {ctx}')









def TEST():
    TEST_get_context()
    TEST_try_gpu()



if __name__ == '__main__':

    TEST()