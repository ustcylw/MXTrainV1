#! /usr/bin/env python
# coding: utf-8
import os, sys
import numpy as np
import cv2 as cv
import utils.utils as Utils
import mxnet as mx



def cv_show_image(image, wait_time=0, RGB2BGR=True, name='image'):
    if RGB2BGR:
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    cv.imshow(name, cv.UMat(image))
    if cv.waitKey(wait_time) == ord('q'):
        sys.exit(0)


def cv_show_images(images, wait_time=0, RGB2BGR=True, name='images'):
    for image in images:
        cv_show_image(image, wait_time=wait_time, name=name)


def cv_show_batch_images(batch_images, wait_time=0, RGB2BGR=True, name='images'):
    for i in range(batch_images.shape[0]):
        image = batch_images[i, :, :, :]
        image = image.transpose((1, 2, 0))
        cv_show_image(image, wait_time=wait_time, name=name)


def cv_draw_bbox(image, bbox_xyxy, color=(255, 0, 0)):
    return cv.rectangle(cv.UMat(image), (bbox_xyxy[0][0], bbox_xyxy[0][1]), (bbox_xyxy[1][0], bbox_xyxy[1][1]), color)


def cv_draw_points(image, points, color=(0, 0, 255), radius=1):
    for point in points:
        image = cv.circle(cv.UMat(image), center=(Utils.ToInt(point[0]), Utils.ToInt(point[1])), color=color, radius=radius)
    return image


def cv_draw_batch_points(batch_images, batch_points, normalized=True, radius=1, color=(0, 0, 255)):
    '''
    :param batch_images: numpy.array, [N, C, H, W]
    :param batch_points: numpy.array, [N, (x1, y1, x2, y2, ...)]
    :param normalized: image transform
    :param radius:
    :param color:
    :return:
    '''
    images = []
    for i in range(batch_images.shape[0]):
        image = batch_images[i, :, :, :]
        image = image.transpose((1, 2, 0))
        keypoints = batch_points[i, :].reshape((-1, 2))
        if normalized:
            image = image * 128.0 + 127.5
        image = image.astype(np.uint8)
        image = cv_draw_points(image, keypoints, color=color, radius=radius)
        images.append(image)
    return images


def cv_show_lm_rets(datas, predi, labeli):
    if isinstance(datas, mx.nd.NDArray):
        datas = datas.as_in_context(mx.cpu()).asnumpy()
    if isinstance(predi, mx.nd.NDArray):
        predi = predi.as_in_context(mx.cpu()).asnumpy()
    if isinstance(labeli, mx.nd.NDArray):
        labeli = labeli.as_in_context(mx.cpu()).asnumpy()

    # cv_show_batch_images(datas, wait_time=300)

    images = cv_draw_batch_points(datas, predi * 128.0, color=(255, 0, 0))
    images = np.stack([image.get().transpose((2, 0, 1)) for image in images], axis=0)
    images = cv_draw_batch_points(images, labeli, normalized=False, color=(0, 0, 255))
    cv_show_images(images, wait_time=300)
