#! /usr/bin/env python
# coding: utf-8
import os, sys
import cv2 as cv
import mxnet as mx
import numpy as np

from collections import namedtuple

__all__ = [
    'Transform'
]


class Transform(object):

    def __init__(self, model_prefix, model_epoch, device, data_channels, data_height, data_width, scale):
        # self.device = device
        # self.data_chanenls = data_channels
        # self.data_height = data_height
        # self.data_width = data_width
        # self.scale = scale
        # self.model_prefix = model_prefix
        # self.model_epoch = model_epoch
        #
        # sym, arg_params, aux_params = mx.model.load_checkpoint(self.model_prefix, self.model_epoch)
        # self.model = mx.mod.Module(symbol=sym, context=self.device, label_names=None)
        # self.model.bind(
        #     data_shapes=[('data', (1, self.data_chanenls, self.data_height, self.data_width))],
        #     label_shapes=self.model._label_shapes,
        #     for_training=False
        # )
        # self.model.set_params(arg_params=arg_params, aux_params=aux_params, allow_missing=True)
        pass

    @staticmethod
    def rect_resize(ori_rect, new_rect, scale):
        '''
        resize face rect
        :param ori_rect:original face rect
        :param new_rect: new resized face rect
        :param scale: scale
        :return: the resized rect
        '''
        new_rect[2] = int(ori_rect[2] * scale + 0.5)
        new_rect[3] = int(ori_rect[3] * scale + 0.5)
        new_rect[0] = ori_rect[0] - (new_rect[2] - ori_rect[2]) / 2
        new_rect[1] = ori_rect[1] - (new_rect[3] - ori_rect[3]) / 2
        for i in range(len(new_rect)):
            new_rect[i] = int(new_rect[i])
        return new_rect

    @staticmethod
    def validate_rect(prect, width, height):
        '''
        validate face rect to make sure it will not exceed border of image
        :param prect: original face rect
        :param width: image width
        :param height: image height
        :return: the validate rect
        '''
        if (prect[0] < 0):
            prect[2] = prect[2] - abs(prect[0])
            prect[0] = 0
        if (prect[1] < 0):
            prect[3] = prect[3] - abs(prect[1])
            prect[1] = 0
        if (prect[0] > width -1):
            prect[0] = width - 1
            prect[2] = 0
        if (prect[1] > height -1):
            prect[1] = height - 1
            prect[3] = 0
        if (prect[2] < 0):
            prect[2] =0
        if (prect[2] > width - prect[0]):
            prect[2] = width - prect[0]
        if (prect[3] < 0):
            prect[3] = 0
        if (prect[3] > height - prect[1]):
            prect[3] = height - prect[1]
        for i in range(len(prect)):
            prect[i] = int(prect[i])
        return prect

    @staticmethod
    def gen_rect_refer(roi_rect, bg_rect, new_rect):
        '''
        generate face rect in refer coordination
        :param roi_rect: original face rect
        :param bg_rect: background rect
        :param new_rect: new face rect refer to background rect
        :return: rect in refer coordination
        '''
        new_rect[0] = roi_rect[0] - bg_rect[0]
        new_rect[1] = roi_rect[1] - bg_rect[1]
        new_rect[2] = roi_rect[2]
        new_rect[3] = roi_rect[3]
        for i in range(len(new_rect)):
            new_rect[i] = new_rect[i]
        return new_rect

    @staticmethod
    def std_image_by_rect(image, valid_rect, refer_rect, std_size):
        '''
        generate standard size image by valid rect and refered rect
        :param image: original image
        :param valid_rect: valid rect
        :param refer_rect: refer rect
        :param std_size: standard size
        :return: standard size image
        '''
        valid_rect = np.int32(valid_rect)
        roi_mat = image[valid_rect[1]:(valid_rect[1] + valid_rect[3]), valid_rect[0]:(valid_rect[0] + valid_rect[2])]
        roi_image = cv.resize(roi_mat, (std_size[0], std_size[1]), interpolation=cv.INTER_CUBIC)
        return roi_image

    @staticmethod
    def extract_std_image(image, rect, std_shape, scale):
        '''
        generate standard size image by face rect
        :param image: source image
        :param rect: detected face rectangle
        :return: standard size image
        '''
        std_size = [std_shape[0], std_shape[1]]
        rect_enlarge = np.zeros(4)
        rect_valid = np.zeros(4)
        rect_enlarge = Transform.rect_resize(rect, rect_enlarge, scale)
        rect_valid = rect_enlarge.copy()
        img_shape = image.shape
        rect_valid = Transform.validate_rect(rect_valid, img_shape[1], img_shape[0])
        rect_refer = np.zeros(4)
        rect_refer = Transform.gen_rect_refer(rect_valid, rect_enlarge, rect_refer)
        enlarge_img = np.zeros((int(rect_enlarge[3]), int(rect_enlarge[2]), 3), np.uint8)
        std_img = Transform.std_image_by_rect(image, enlarge_img, rect_valid, rect_refer, std_size)
        return std_img, rect_enlarge

    @staticmethod
    def trans2format_with_mxnet(image, rect, num_channel=3, normal=0):
        '''
        transfer opencv format image to mxnet format image
        :param image: original opencv format image
        :param rect: face rect
        :param num_channel: number of channel
        :param normal: normalizition
        :return: mxnet format image
        '''
        image, _ = Transform.extract_std_image(image, rect)
        if normal == 0:
            image = (image - 127.5) / 128.0
        elif normal == 1:
            image = image / 255.0
        if num_channel == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image = cv.transpose(image, (2, 0, 1))
            image = image[np.newaxis, :, :, :]
        elif num_channel == 1:
            image = image[np.newaxis, :]
        else:
            assert num_channel not in [1, 3], 'channel:{} is not correct !!!'.format(num_channel)
        return image

    @staticmethod
    def trans2fromat_with_cv(data, batch_dim=None, num_channels=3, normal=0):
        '''
        transform data to opencv format
        :param data: original data
        :param batch_dim: batch dimension
        :param num_channels: number of channels
        :param normal: normalization
        :return: opencv format image
        '''
        image = data
        if num_channels == 3:
            if batch_dim:
                image = np.transpose(data, (0, 2, 3, 1))
            else:
                image = np.transpose(data, (1, 2, 0))
        if normal == 0:
            image = image * 128.0 + 127.5
        elif normal == 1:
            image = image * 255.0
        else:
            assert normal not in [0, 1], 'normalization type: {} is not correct !!!'.format(normal)
        return image
