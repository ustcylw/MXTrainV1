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
import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

__all__ = [
    'ImageFolderDataIter', 'ImageFolderDataset', 'default_batchify_fn', 'ImageFolderDataLoader'
]


def default_batchify_fn(data):
    """Collate data into batch."""
    if isinstance(data[0], mx.nd.NDArray):
        # return _mx_np.stack(data) if is_np_array() else nd.stack(*data)
        return mx.numpy.stack(data) if mx.util.is_np_array() else mx.nd.stack()
    elif isinstance(data[0], tuple):
        data = zip(*data)
        return [ImageFolderDataIter.default_batchify_fn(i) for i in data]
    else:
        data = np.asarray(data)
        # array_fn = _mx_np.array if is_np_array() else nd.array
        array_fn = mx.numpy.array if mx.util.is_np_array() else mx.nd.array
        return array_fn(data, dtype=data.dtype)


class ImageFolderDataIter(mx.io.DataIter):
    def __init__(
            self,
            root,
            data_shape=(3, 128, 128),
            label_shape=(2,),
            data_names=['data'],
            label_names=['softmax_label'],
            flag=1,
            transform=None,
            batch_size=0
    ):
        super(ImageFolderDataIter, self).__init__(batch_size=batch_size)
        self.batch_size = batch_size
        self._root = os.path.expanduser(root)
        self.data_shape = data_shape
        self.label_shape = label_shape
        self.data_names = data_names
        self.label_names = label_names
        self._flag = flag
        self._transform = transform
        self._exts = ['.jpg', '.jpeg', '.png']
        self._list_images(self._root)

        self.max_sample = len(self.items)
        self.cur = 0
        self.current_batch = [None, None]

    @property
    def provide_data(self):
        return mx.io.DataDesc(self.data_names[0], (self.batch_size, self.data_shape[0], self.data_shape[1], self.data_shape[2]))

    @property
    def provide_label(self):
        return mx.io.DataDesc(self.label_names[0], (self.batch_size, self.label_shape[0]))

    def __iter__(self):
        return self

    def reset(self):
        self.cur = 0

    def next(self):
        if self.iter_next():
            data = self.getdata()
            label = self.getlabel()
            return mx.io.DataBatch(data=[data], label=[label], pad=self.getpad(), index=self.getindex())
        else:
            raise StopIteration

    def _list_images(self, root):
        self.synsets = []
        self.items = []

        for folder in sorted(os.listdir(root)):
            path = os.path.join(root, folder)
            if not os.path.isdir(path):
                warnings.warn('Ignoring %s, which is not a directory.'%path, stacklevel=3)
                continue
            label = len(self.synsets)
            self.synsets.append(folder)
            for filename in sorted(os.listdir(path)):
                filename = os.path.join(path, filename)
                ext = os.path.splitext(filename)[1]
                if ext.lower() not in self._exts:
                    warnings.warn('Ignoring %s of type %s. Only support %s'%(
                        filename, ext, ', '.join(self._exts)))
                    continue
                self.items.append((filename, label))

    def __next__(self):
        return self.next()

    def read_batch(self):
        images, labels = [], []
        for i in range(self.batch_size):
            img = cv.imread(self.items[self.cur][0], self._flag)
            img = cv.resize(img, (128, 128))
            img = img.transpose((2, 0, 1))
            # img = img[np.newaxis, :, :, :]
            label = self.items[self.cur][1]
            if self._transform is not None:
                img, label = self._transform(img, label)
            images.append(img)
            labels.append(label)
            self.cur += 1
        data = default_batchify_fn(images)
        targets = default_batchify_fn(labels).astype(data.dtype)
        # print(f'[read_batch] data: {data.shape}  {data.dtype}')
        # print(f'[read_batch] targets: {targets.shape}  {targets.dtype}')
        self.current_batch[0] = data
        self.current_batch[1] = targets

    def iter_next(self):
        if (self.cur + self.batch_size) >= (self.max_sample - 1):
            return False
        try:
            self.read_batch()
            # self.current_batch[0] = self.data_iter.next()
        except StopIteration:
            # self.data_iter.reset()
            # self.current_batch = self.data_iter.next()
            self.reset()
            self.read_batch()

        return True

    def getdata(self):
        return self.current_batch[0]

    def getlabel(self):
        return self.current_batch[1]

    def getindex(self):
        return self.cur

    def getpad(self):
        """Get the number of padding examples in the current batch.

        Returns
        -------
        int
            Number of padding examples in the current batch.
        """
        pass



class ImageFolderDataset(mx.gluon.data.Dataset):
    def __init__(self, root, flag=1, transform=None):
        self._root = os.path.expanduser(root)
        self._flag = flag
        self._transform = transform
        self._exts = ['.jpg', '.jpeg', '.png']
        self._list_images(self._root)

    def _list_images(self, root):
        self.synsets = []
        self.items = []

        for folder in sorted(os.listdir(root)):
            path = os.path.join(root, folder)
            if not os.path.isdir(path):
                warnings.warn('Ignoring %s, which is not a directory.'%path, stacklevel=3)
                continue
            label = len(self.synsets)
            self.synsets.append(folder)
            for filename in sorted(os.listdir(path)):
                filename = os.path.join(path, filename)
                ext = os.path.splitext(filename)[1]
                if ext.lower() not in self._exts:
                    warnings.warn('Ignoring %s of type %s. Only support %s'%(
                        filename, ext, ', '.join(self._exts)))
                    continue
                self.items.append((filename, label))

    def __getitem__(self, idx):
        img = cv.imread(self.items[idx][0], self._flag)
        img = cv.resize(img, (128, 128))
        img = img.transpose((2, 0, 1))
        # img = image.imread(self.items[idx][0], self._flag)
        label = self.items[idx][1]
        if self._transform is not None:
            img, label = self._transform(img, label)
        return img, label

    def __len__(self):
        return len(self.items)


class ImageFolderDataLoader(mx.gluon.data.DataLoader):
    def __init__(
            self,
            dataset,
            batch_size,
            data_names=['data',],
            label_names=['softmax_label',],
            data_shape=(3, 128, 128),
            label_shape=(2,),
            shuffle=True,
            last_batch='discard'
    ):
        super(ImageFolderDataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            last_batch=last_batch
        )
        self._provide_data = [(data_names[0], (batch_size, data_shape[0], data_shape[1], data_shape[2]))]
        self._provide_label = [(label_names[0], (batch_size, label_shape[0]))]

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label
