#! /usr/bin/env python
# coding: utf-8
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import mxnet as mx
import numpy as np
import cv2 as cv
import pickle
from dataset.mx_rec import MXRec


class RecDataIterV1(object):
    def __init__(
            self,
            rec_dir,
            rec_prefix,
            batch_size=0,
            shuffle=True,
            aug_list=None,
            devices=None
    ):
        self.rec_dir = rec_dir
        self.rec_prefix = rec_prefix
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cursor = 0
        self.aug_list = aug_list
        self.devices = devices if devices is not None else mx.cpu()
        self.idx_list = []
        self.rec_handler = MXRec(rec_dir=self.rec_dir, prefix=self.rec_prefix)
        self.parse_idx_file()
        self.max_index = len(self.idx_list)

    def __iter__(self):
        return self

    def reset(self):
        self.cursor = 0

    def next(self):
        if self.iter_next():
            data = self.getdata()
            return mx.io.DataBatch(data=[data[0]], label=[data[1]], pad=self.getpad(), index=self.getindex())
        else:
            raise StopIteration

    def __next__(self):
        return self.next()

    def iter_next(self):
        if self.cursor + self.batch_size > self.max_index:
            return False
        return True

    def getdata(self):
        data, target1, target2 = None, None, None
        if self.iter_next():
            batch_items = self.idx_list[self.cursor:(self.cursor+self.batch_size)]
            batch_data = []
            for batch_item in batch_items:
                batch_data.append(self.rec_handler.read_rec(batch_item))
            data_list, target_list1, target_list2 = [], [], []
            for i in range(len(batch_data)):
                image = batch_data[i]['image']
                keypoints = batch_data[i]['keypoints']
                points = batch_data[i]['points']
                # image = image[np.newaxis, :, :, :]
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                image = np.transpose(image, (2, 0, 1))
                if self.aug_list:
                    for aug in self.aug_list:
                        image = aug.process(image)
                data_list.append(image)
                target_list1.append(points)
                target_list2.append(keypoints)
            data_array = np.array(data_list)
            target_array1 = np.array(target_list1)
            target_array2 = np.array(target_list2)
            data = mx.nd.array(data_array, ctx=self.devices)
            target1 = mx.nd.array(target_array1, ctx=self.devices)
            target2 = mx.nd.array(target_array2, ctx=self.devices)
        else:
            raise StopIteration
        # return mx.io.DataBatch(data=[data], label=[target])
        self.cursor += self.batch_size
        return data, (target1, target2)

    def getlabel(self):
        pass

    def getindex(self):
        return None

    def getpad(self):
        pass

    def parse_idx_file(self):
        if len(self.idx_list) > 0:
            self.idx_list.clear()
        idx_file = os.path.join(self.rec_dir, self.rec_prefix+'.idx')
        with open(idx_file, 'r') as lf:
            lines = lf.readlines()
        for line in lines:
            line = line.strip()
            idx, num_bytes = line.split('\t')
            idx, num_bytes = int(idx), int(num_bytes)
            self.idx_list.append(idx)



if __name__ == '__main__':

    import matplotlib.pyplot as plt

    rec_dir = '/data1/dataset/300/rec/002'
    rec_prefix = '300vw_1_1_'
    batch_size = 2
    shuffle = True
    loader = RecDataIterV1(
        rec_dir=rec_dir,
        rec_prefix=rec_prefix,
        batch_size=batch_size,
        shuffle=shuffle
    )

    for index, data in enumerate(iter(loader)):
        images = data.data[0]
        labels1 = data.label[0]
        labels2 = data.label[0]
        print(f'images: {images.shape}')
        print(f'labels1: {labels1.shape}')
        print(f'labels2: {labels2.shape}')
        shape = images.shape
        for i in range(shape[0]):
            image = images[i, :, :, :].asnumpy()
            label1 = labels1[i, :].reshape((68, 2)).asnumpy()
            label2 = labels2[i, :].reshape((-1, 2)).asnumpy()
            image = np.transpose(image, (1, 2, 0))
            image = image.astype(np.uint8)
            image_ = image
            for j in range(68):
                point = label1[j, :]
                image = cv.circle(
                    img=cv.UMat(image),
                    center=(int(round(point[0])), int(round(point[1]))),
                    radius=1,
                    color=(255, 0, 0),
                    thickness=-1
                )
            for j in range(5):
                point = label2[j, :]
                image = cv.circle(
                    img=cv.UMat(image),
                    center=(int(round(point[0])), int(round(point[1]))),
                    radius=2,
                    color=(0, 0, 255),
                    thickness=-1
                )
            print('image: {}  {}  {}'.format(image.get().shape, image.get().dtype, type(image)))
            # cv.imshow('image', image)
            # if cv.waitKey(1000) == ord('q'):
            #     sys.exit(0)
            # print('[{}/{}/{}]  image: {}  {}'.format(index, i, j, type(image.get()), image.get().shape))
            print('[{}/{}/{}]  image: {}  {}'.format(index, i, j, type(image), image.get().shape))
            plt.imshow(image.get())  ## cv.UMat --> np.array
            # plt.imshow(image)  ## cv.UMat --> np.array
            plt.show()
