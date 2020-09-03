#! /usr/bin/env python
# coding: utf-8
import os, sys
import numpy as np
import cv2
import mxnet
import random
import glob
import mx_rec as MXRec


class AugType(object):
    def __init__(self):
        pass
    def process(self, image):
        pass

class Whiteening(AugType):
    def process(self, image):
        image_mean = np.mean(image)
        image_var = np.std(image)
        std_adj = np.maximum(image_var, 1.0/np.sqrt(image.size))
        return np.multiply(np.subtract(image, image_mean), 1.0/std_adj)


class DataIter(object):
    def __init__(
        self,
        devices,
        root_dir,
        file_list,
        data_shape,
        label_shape,
        data_name='data',
        label_name='label',
        batch_size=128,
        shuffle=True,
        aug_list=None
    ):
        self._devices = devices
        self._root_dir = root_dir
        self._file_list = file_list
        self._data_name = data_name
        self._label_name = label_name
        self._data_shape = data_shape
        self._label_shape = label_shape
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._aug_list = aug_list

        self._max_index = 0
        self._current_cursor = 0
        self._provide_data = zip(self.data_name, self.data_shape)
        self._provide_label = zip(self.label_name, self.label_shape)
        self._sample_list = []

    def prepare_dataset(self):
        if os.path.exists(self._file_list):
            print('data list file is exists.')
            lines = None
            with open(self._file_list, 'r') as lf:
                lines = lf.readlines()
            self._sample_list.clear()
            self._sample_list = [line.strip() for line in lines]
            random.shuffle(self._sample_list)
            self._max_index = len(self._sample_list)
            return
        print('create dataset file list...')
        sub_dirs = os.listdir(self._root_dir)
        to_string = ''
        self._sample_list.clear()
        for sub_dir in sub_dirs:
            label = 0
        with open(self.file_list, 'w') as df:
            df.write(to_string)
            df.flush()

    def provide_data(self):
        return self._provide_data

    def provide_label(self):
        return self._provide_label

    def __init__(self):
        return self

    def _shuffle_list(self):
        random.shuffle(self._sample_list)
        self._current_cursor = 0

    def reset(self):
        self._current_cursor = 0
        if self._shuffle:
            self._shuffle_list()

    def next(self):
        data, target = None, None
        if self.iter_next():
            batch_items = self._sample_list[self._current_cursor:self._current_cursor+self._batch_size]
            batch_paths, batch_targets = [], []
            for batch_item in batch_items:
                label, depth_file = batch_item.split(' ')
                batch_paths.append(depth_file)
                batch_targets.append(int(label))
            data_list, target_list = [], []
            for i in range(len(batch_paths)):
                depth = np.load(batch_paths[i], allow_pickle=True)
                if depth is None:
                    print('depth file is not a file!!!', os.path.join(self._root_dir, batch_paths[i]))
                depth = depth[np.newaxis, :, :, :]
                if self._aug_list:
                    for aug in self._aug_list:
                        depth = aug.process(depth)
                data_list.append(depth)
                target_list.append(depth)
            data_array = np.array(data_list)
            target_array = np.array(target_list)
            data = mx.nd.array(data_array, ctx=self._devices)
            target = mx.nd.array(target_array, ctx=self._devices)
        else:
            raise StopIteration
        self._current_cursor += self._batch_size
        return mx.io.DataBatch(data=[data], label=[target])

    def __next__(self):
        return self.next()

    def getpad(self):
        pass

    def getindex(self):
        return None

    def getlabel(self):
        pass

    def getdata(self):
        pass

    def iter_next(self):
        if self._current_cursor + self._batch_size > self._max_index:
            return False
        return True