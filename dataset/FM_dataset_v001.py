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
from utils.bbox import square_bbox, bbox_xywh2xyxy, bbox_xyxy2xywh, random_scale_bbox
import utils.cv_show as CVShow



__all__ = [
    'FMDataIterV1', 'default_batchify_fn', 'FMDataset', 'FMDataLoader', 'FMDataIterV1_300VW'
]


ToInt = lambda x: int(round(x))


def default_batchify_fn(data):
    """Collate data into batch."""
    if isinstance(data[0], mx.nd.NDArray):
        # return _mx_np.stack(data) if is_np_array() else nd.stack(*data)
        return mx.numpy.stack(data) if mx.util.is_np_array() else mx.nd.stack()
    elif isinstance(data[0], tuple):
        data = zip(*data)
        return [FMDataIterV1.default_batchify_fn(i) for i in data]
    else:
        data = np.asarray(data)
        # array_fn = _mx_np.array if is_np_array() else nd.array
        array_fn = mx.numpy.array if mx.util.is_np_array() else mx.nd.array
        return array_fn(data, dtype=data.dtype)


class FMDataIterV1(mx.io.DataIter):
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
        super(FMDataIterV1, self).__init__(batch_size=batch_size)
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
        self.items = []

        anno_dir = os.path.join(root, 'annos/**/*.txt')
        anno_files = glob.glob(anno_dir, recursive=True)
        image_dir = os.path.join(root, 'images')
        for anno_file in anno_files:
            if not anno_file.endswith('.txt'):
                warnings.warn('Ignoring %s, which is not a txt file.'%anno_file, stacklevel=3)
                continue
            if not os.path.exists(anno_file):
                warnings.warn('Ignoring %s, which is not a anno-file.'%anno_file, stacklevel=3)
                continue
            lines = None
            with open(anno_file, 'r') as lf:
                lines = lf.readlines()
            for line in lines:
                img_file, keypoints, visable = line.strip().split(' ')
                _, ext = os.path.splitext(img_file)
                if ext not in ['.jpg', '.png']:
                    warnings.warn(f'not image file!!! {ext}  {img_file}', stacklevel=3)
                    continue
                image_file = os.path.join(image_dir, img_file)
                keypoints = np.fromstring(keypoints, dtype=np.float32, sep=',')
                visable = np.fromstring(visable, dtype=np.int32, sep=',')
                self.items.append({'image_file': image_file, 'info':[keypoints, visable]})

    def _list_images_300vw(self, root):
        self.items = []

        id_names = os.listdir(root)
        for id_name in id_names:
            anno_path = os.path.join(os.path.join(root, id_name), 'annot')
            annot_names = os.listdir(anno_path)
            for annot_name in annot_names:
                annot_file = os.path.join(anno_path, annot_name)
                if not os.path.exists(annot_file):
                    warnings.warn(f'{annot_file} is not exist!!!')
                image_file = annot_file.replace('annot', 'images')
                if not os.path.exists(image_file):
                    warnings.warn(f'{image_file} is not exist!!!')
                self.items.append([annot_file, image_file])

    def __next__(self):
        return self.next()

    def read_batch(self):
        images, labels = [], []
        for i in range(self.batch_size):
            print(f'[============]  image file: {self.items[self.cur]["image_file"]}')
            img = cv.imread(self.items[self.cur]['image_file'], self._flag)
            img = cv.resize(img, (self.data_shape[1], self.data_shape[2]))
            img = img.transpose((2, 0, 1))
            # img = img[np.newaxis, :, :, :]
            label = self.items[self.cur]['info'][0]
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


class FMDataset(mx.gluon.data.Dataset):
    def __init__(self, root, ctxs=[mx.cpu()], data_shape = (3, 128, 128), label_shape = (2,), batch_size = 0, data_names = ['data'], label_names = ['softmax_label'], flag=1, transform=None):
        super(FMDataset, self).__init__()
        self._root = os.path.expanduser(root)
        self._flag = flag
        self._transform = transform
        self._exts = ['.jpg', '.jpeg', '.png']
        self._list_images(self._root)
        self.items = self.items[0:128]
        self.num_samples = len(self.items)

        self.data_shape = data_shape
        self.label_shape = label_shape
        self.data_names = data_names
        self.label_names = label_names
        self.batch_size = batch_size
        self.ctxs = ctxs

    def _list_images(self, root):
        self.items = []

        id_names = os.listdir(root)
        for id_name in id_names:
            # print(f'anno-path: {os.path.join(os.path.join(root, id_name), "annot")}')
            anno_path = os.path.join(os.path.join(root, id_name), 'annot')
            annot_names = os.listdir(anno_path)
            for annot_name in annot_names:
                annot_file = os.path.join(anno_path, annot_name)
                if not os.path.exists(annot_file):
                    warnings.warn(f'{annot_file} is not exist!!!')
                if not annot_file.endswith('.pts'):
                    continue
                image_file = annot_file.replace('annot', 'images').replace('.pts', '.jpg')
                if not os.path.exists(image_file):
                    # warnings.warn(f'{image_file} is not exist!!!')
                    image_file = annot_file.replace('annot', 'images').replace('.pts', '.png')
                    if not os.path.exists(image_file):
                        warnings.warn(f'{image_file} is not exist!!!')
                        continue
                self.items.append([image_file, annot_file])

    def read_annot(self, annot_file):
        lines = None
        # print(f'annot_file: {annot_file}')
        with open(annot_file, 'r') as lf:
            lines = lf.readlines()
        lines = lines[3:-1]
        if len(lines) != 68:
            warnings.warn(f'number of keypoints is not correct!!!')
        keypoints = []
        for line in lines:
            point = np.fromstring(line.strip(), dtype=np.float32, sep=' ')
            keypoints.append(point)
        keypoints = np.stack(keypoints, axis=0).reshape((-1, 2))
        # print(f'keypoints: {keypoints.shape}')
        return keypoints

    def __getitem__(self, idx):
        # print(f'[============]  image file: {self.items[self.cur][0]}')
        img = cv.imread(self.items[idx][0], self._flag)
        # print(f'[===], {img.shape}')
        ori_shape = [img.shape[1], img.shape[0]]

        annot_file = self.items[idx][1]
        keypoints = self.read_annot(annot_file)
        bbox = [np.min(keypoints, axis=0), np.max(keypoints, axis=0)]
        # print(f'[1]  bbox: {bbox}')
        bbox = random_scale_bbox(bbox, scale=np.random.randint(1, 10) / 10)
        # print(f'[2]  bbox: {bbox}  data-shape: {self.data_shape}')
        # bbox = square_bbox(bbox, (self.data_shape[0][1], self.data_shape[0][2])).astype(np.int32)
        bbox = square_bbox(bbox, ori_shape).astype(np.int32)
        # print(f'[3]  bbox: {bbox}')
        # CVShow.cv_show_bbox(image=img, bbox_xyxy=bbox, wait_time=0)

        img = img[ToInt(bbox[0][1]):ToInt(bbox[1][1]), ToInt(bbox[0][0]):ToInt(bbox[1][0]), :]
        # print(f'img: {img.shape}  bbox: {bbox}')
        # CVShow.cv_show_image(img)

        zero_point = bbox[0]
        keypoints = keypoints - zero_point
        # CVShow.cv_show_points(img, keypoints, wait_time=300)

        ori_shape = img.shape
        # print(f'zero point: {zero_point}  ori-shape: {ori_shape}  {self.items[idx][0]}')
        # print(f'images: {type(img.shape)}')
        img = cv.resize(img, (self.data_shape[0][1], self.data_shape[0][2]))
        # img_ = img.copy()
        img = img.transpose((2, 0, 1))
        # img = img[np.newaxis, :, :, :]

        # print(f'keypoints: {keypoints}  zeors-points: {zero_point}  ori-shape: {ori_shape}')
        # print(f'[2]  data-shape: {self.data_shape}  ori-shape: {ori_shape}')
        w_scale = self.data_shape[0][1] / ori_shape[0]
        h_scale = self.data_shape[0][2] / ori_shape[1]
        keypoints[:, 0] *= w_scale
        keypoints[:, 1] *= h_scale
        # CVShow.cv_show_points(img_, keypoints, wait_time=300)

        if self._transform is not None:
            for transform in self._transform:
                img, keypoints = transform(img, keypoints)
        # img = img.astype(np.float64)
        keypoints1 = keypoints#.astype(np.float64)
        keypoints2 = keypoints[30, :].reshape((-1))#.astype(np.float64)
        return img, (keypoints1.reshape((-1)), keypoints2.reshape((-1)))

    def __len__(self):
        return len(self.items)


class FMDataLoader(mx.gluon.data.DataLoader):
    def __init__(
            self,
            dataset,
            batch_size,
            data_names=['data',],
            label_names=['kps68-regression', 'kpsn-regression'],
            data_shape=((3, 128, 128),),
            label_shape=(68, 2),
            shuffle=True,
            last_batch='discard'
    ):
        super(FMDataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            last_batch=last_batch
        )
        # print(f'data: {data_names} / {data_shape}  label: {label_names} / {label_shape}')
        self._provide_data = [(data_names[i], (batch_size, data_shape[i][0], data_shape[i][1], data_shape[i][2])) for i in range(len(data_names))]
        self._provide_label = [(label_names[i], (batch_size, label_shape[i][0])) for i in range(len(label_names))]

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label


class FMDataIterV1_300VW(mx.io.DataIter):
    def __init__(
            self,
            root,
            data_shape=(3, 128, 128),
            label_shape=(2,),
            data_names=['data'],
            label_names=['softmax_label'],
            flag=1,
            transform=None,
            batch_size=0,
            ctxs=[mx.cpu()]
    ):
        super(FMDataIterV1_300VW, self).__init__(batch_size=batch_size)
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
        self.ctxs = ctxs

        self.max_sample = len(self.items)
        self.cur = 0
        self.current_batch = [None, None, None]

    @property
    def provide_data(self):
        print(f'provide data: {self.data_names}  {self.data_shape}  {len(self.data_names)}  {self.data_shape[0]}')
        # for i in range(len(self.data_names)):
            # print(f'>>>>>>>>  {i}  <<<<<<<<<<<<')
            # print(f'data-name-{i}: {self.data_names[i]}  {self.data_shape[i]}')
        return [mx.io.DataDesc(self.data_names[i], (self.batch_size, self.data_shape[i][0], self.data_shape[i][1], self.data_shape[i][2])) for i in range(len(self.data_names))]

    @property
    def provide_label(self):
        print(f'provide label: {self.label_names}  {self.label_shape}  {len(self.label_names)}  {self.label_shape[0]}')
        # for i in range(len(self.label_names)):
        #     print(f'>>>>>>>>  {i}  <<<<<<<<<<<<')
        #     print(f'label-name-{i}: {self.label_names[i]}  {(self.batch_size, self.label_shape[i])}')
        return [mx.io.DataDesc(self.label_names[i], (self.batch_size, self.label_shape[i])) for i in range(len(self.label_names))]

    def __iter__(self):
        return self

    def reset(self):
        self.cur = 0

    def next(self):
        if self.iter_next():
            data = self.getdata()
            label = self.getlabel()
            return mx.io.DataBatch(data=[data], label=[label[0], label[1]], pad=self.getpad(), index=self.getindex())
        else:
            raise StopIteration

    def _list_images(self, root):
        self.items = []

        id_names = os.listdir(root)
        for id_name in id_names:
            print(f'anno-path: {os.path.join(os.path.join(root, id_name), "annot")}')
            anno_path = os.path.join(os.path.join(root, id_name), 'annot')
            annot_names = os.listdir(anno_path)
            for annot_name in annot_names:
                annot_file = os.path.join(anno_path, annot_name)
                if not os.path.exists(annot_file):
                    warnings.warn(f'{annot_file} is not exist!!!')
                image_file = annot_file.replace('annot', 'images').replace('.pts', '.jpg')
                if not os.path.exists(image_file):
                    warnings.warn(f'{image_file} is not exist!!!')
                self.items.append([image_file, annot_file])

    def __next__(self):
        return self.next()

    def read_annot(self, annot_file):
        lines = None
        # print(f'annot_file: {annot_file}')
        with open(annot_file, 'r') as lf:
            lines = lf.readlines()
        lines = lines[3:-1]
        if len(lines) != 68:
            warnings.warn(f'number of keypoints is not correct!!!')
        keypoints = []
        for line in lines:
            point = np.fromstring(line.strip(), dtype=np.float32, sep=' ')
            keypoints.append(point)
        keypoints = np.stack(keypoints, axis=0).reshape((-1, 2))
        # print(f'keypoints: {keypoints.shape}')
        return keypoints

    def read_batch(self):
        images, labels1, labels2 = [], [], []
        for i in range(self.batch_size):
            # print(f'[============]  image file: {self.items[self.cur][0]}')
            img = cv.imread(self.items[self.cur][0], self._flag)

            annot_file = self.items[self.cur][1]
            keypoints = self.read_annot(annot_file)
            bbox = [np.min(keypoints, axis=0), np.max(keypoints, axis=0)]
            # print(f'[1]  bbox: {bbox}')
            bbox = random_scale_bbox(bbox, scale=np.random.randint(1, 10) / 10)
            # print(f'[2]  bbox: {bbox}  data-shape: {self.data_shape}')
            bbox = square_bbox(bbox, (self.data_shape[0][1], self.data_shape[0][2])).astype(np.int32)
            # print(f'[3]  bbox: {bbox}')
            # CVShow.cv_show_bbox(image=img, bbox_xyxy=bbox, wait_time=0)

            img = img[ToInt(bbox[0][1]):ToInt(bbox[1][1]), ToInt(bbox[0][0]):ToInt(bbox[1][0]), :]
            # print(f'img: {img.shape}  bbox: {bbox}')
            # CVShow.cv_show_image(img)

            zero_point = bbox[0]
            keypoints = keypoints - zero_point
            # CVShow.cv_show_points(img, keypoints, wait_time=300)

            ori_shape = img.shape
            # print(f'zero point: {zero_point}  ori-shape: {ori_shape}')
            img = cv.resize(img, (self.data_shape[0][1], self.data_shape[0][2]))
            # img_ = img.copy()
            img = img.transpose((2, 0, 1))
            # img = img[np.newaxis, :, :, :]

            # print(f'keypoints: {keypoints}  zeors-points: {zero_point}  ori-shape: {ori_shape}')
            # print(f'[2]  data-shape: {self.data_shape}  ori-shape: {ori_shape}')
            w_scale = self.data_shape[0][1] / ori_shape[0]
            h_scale = self.data_shape[0][2] / ori_shape[1]
            keypoints[:, 0] *= w_scale
            keypoints[:, 1] *= h_scale
            # CVShow.cv_show_points(img_, keypoints, wait_time=300)

            if self._transform is not None:
                for transform in self._transform:
                    img, keypoints = transform(img, keypoints)
            images.append(img)
            labels1.append(keypoints.reshape((-1)))
            labels2.append(keypoints[30, :].reshape((-1)))
            # img_ = img.transpose((1, 2, 0))
            # CVShow.cv_show_points(img_, keypoints, wait_time=300)

            self.cur += 1
        data = default_batchify_fn(images)
        data = data.astype(np.float32)
        labels1 = default_batchify_fn(labels1).astype(data.dtype)
        labels2 = default_batchify_fn(labels2).astype(data.dtype)
        # targets = (labels1, labels2)
        # print(f'[read_batch] data: {data.shape}  {data.dtype}')
        # print(f'[read_batch] targets: {targets.shape}  {targets.dtype}')
        self.current_batch[0] = data
        self.current_batch[1] = labels1
        self.current_batch[2] = labels2

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
        return self.current_batch[1], self.current_batch[2]

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


def TEST_FMDataIterV1():

    root='/data2/datasets/test'
    data_shape = ((3, 128, 128),)
    label_shape = (136, 2)
    data_names = ['data',]
    label_names = ['kpts68-regression', 'kptsn-regression']
    flag = 1
    transform = lambda x, y: (x / 255.0, y)
    batch_size = 2

    fm_dataiter = FMDataIterV1(
        root=root,
        data_shape=data_shape,
        label_shape=label_shape,
        data_names=data_names,
        label_names=label_names,
        flag=flag,
        transform=transform,
        batch_size=batch_size
    )
    print(f'iter: {fm_dataiter.max_sample}')

    for i, data_batch in enumerate(fm_dataiter):
        print(f'batch: {len(data_batch.data)} --> {data_batch.data[0].shape}  \n{len(data_batch.label)}  -->  {data_batch.label[0].shape}')


def TEST_FMDataIterV1_300VW():

    root='/data2/datasets/300VW/300VW_Dataset_2015_12_14'
    data_shape = ((3, 128, 128),)
    label_shape = (136, 2)
    data_names = ['data',]
    label_names = ['kpts68-regression', 'kptsn-regression']
    flag = 1
    transform = None
    batch_size = 2

    fm_dataiter = FMDataIterV1_300VW(
        root=root,
        data_shape=data_shape,
        label_shape=label_shape,
        data_names=data_names,
        label_names=label_names,
        flag=flag,
        transform=transform,
        batch_size=batch_size
    )
    print(f'iter: {fm_dataiter.max_sample}')

    for i, data_batch in enumerate(fm_dataiter):
        print(f'batch: {len(data_batch.data)} --> {data_batch.data[0].shape}  \n{len(data_batch.label)}  -->  {data_batch.label[0].shape}')
        data = data_batch.data[0]
        label = data_batch.label[0]
        for i in range(data.shape[0]):
            datai = data[i, :, :, :].asnumpy()
            labeli = label[i, :].asnumpy()
            print(f'datai: {datai.shape} / {type(datai)} / {datai.dtype}  labeli: {labeli.shape}')
            image = np.transpose(datai, (1, 2, 0))
            print(f'image: {image.shape}')
            CVShow.cv_show_points(image=cv.UMat(image.astype(np.uint8)), points=labeli.reshape((-1, 2)), wait_time=300)


def TEST_FMDataLoader():
    # root = '/data2/datasets/300VW/300VW_Dataset_2015_12_14'
    root = '/data2/datasets/300w-format'
    data_shape = ((3, 128, 128),)
    label_shape = ((136,), (2,))
    data_names = ['data',]
    label_names = ['kpts68-regression', 'kptsn-regression']
    flag = 1
    transform = None
    batch_size = 2

    fm_dataset = FMDataset(
        root=root,
        data_shape=data_shape,
        label_shape=label_shape,
        data_names=data_names,
        label_names=label_names,
        flag=flag,
        transform=transform,
        batch_size=batch_size
    )
    print(f'iter: {fm_dataset.num_samples}')
    fm_dataloader = FMDataLoader(fm_dataset, batch_size, data_names, label_names=label_names, data_shape=data_shape, label_shape=label_shape)

    for i, (data, (labels1, labels2)) in enumerate(fm_dataloader):
        print(f'data: {data.shape}  labels1: {labels1.shape}  labels2: {labels2.shape}')
        for i in range(data.shape[0]):
            datai = data[i, :, :, :].asnumpy()
            label1i = labels1[i, :].asnumpy()
            label2i = labels2[i, :].asnumpy()
            print(f'datai: {datai.shape} / {type(datai)} / {datai.dtype}  label1i: {label1i.shape}  label2i: {label2i.shape}')
            image = np.transpose(datai, (1, 2, 0))
            print(f'image: {image.shape}')
            # image = CVShow.cv_draw_points(image=cv.UMat(image.astype(np.uint8)), points=label1i.reshape((-1, 2)), color=(0, 0, 255))
            # image = CVShow.cv_draw_points(image=cv.UMat(image.astype(np.uint8)), points=label2i.reshape((-1, 2)), color=(255, 0, 0))
            image = CVShow.cv_draw_points(image=cv.UMat(image), points=label1i.reshape((-1, 2)), color=(0, 0, 255))
            image = CVShow.cv_draw_points(image=cv.UMat(image), points=label2i.reshape((-1, 2)), color=(255, 0, 0))
            CVShow.cv_show_image(image, wait_time=300)



if __name__ == '__main__':

    # items = ''
    # root = '/data2/datasets/test'
    # lines = None
    # # anno_file = '/data2/datasets/test/annos/000001/000001.txt'
    # anno_file = '/data2/datasets/test/annos/000002/000002.txt'
    # with open(anno_file, 'r') as lf:
    #     lines = lf.readlines()
    # for idx, line in enumerate(lines):
    #     image_file = line.strip()
    #     line = line.strip() + ' '
    #     for i in range(136):
    #         line += str(np.random.randint(10, 100)) + ','
    #     line = line[:-1]
    #     line += ' '
    #     for i in range(68):
    #         line += str(int(np.random.randint(10, 100)%2)) + ','
    #     line = line[:-1] + '\n'
    #     items += line
    # with open(anno_file, 'w') as df:
    #     df.writelines(items)


    # TEST_FMDataIterV1()
    # TEST_FMDataIterV1_300VW()
    TEST_FMDataLoader()

