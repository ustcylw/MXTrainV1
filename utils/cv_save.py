#! /usr/bin/env python
# coding: utf-8
import os, sys
import numpy as np
import cv2 as cv
import utils.utils as Utils
import mxnet as mx
import utils.cv_show as CVShow


def cv_save_lm_rets(datas, predi, labeli, prefix, preds_scale=128, labels_scale=1.0, num_count=0, resize=None):
    if isinstance(datas, mx.nd.NDArray):
        datas = datas.as_in_context(mx.cpu()).asnumpy()
    if isinstance(predi, mx.nd.NDArray):
        predi = predi.as_in_context(mx.cpu()).asnumpy()
    if isinstance(labeli, mx.nd.NDArray):
        labeli = labeli.as_in_context(mx.cpu()).asnumpy()
    images = CVShow.cv_draw_batch_points(datas, predi * preds_scale, color=(255, 0, 0))
    images = np.stack([image.get().transpose((2, 0, 1)) for image in images], axis=0)
    images = CVShow.cv_draw_batch_points(images, labeli * labels_scale, normalized=False, color=(0, 0, 255))
    for image in images:
        image_file = prefix + f"{num_count:06d}.jpg"
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        if resize is not None:
            image = cv.resize(image, resize)
        cv.imwrite(image_file, image)
        num_count += 1
