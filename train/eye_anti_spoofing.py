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
        self._flag = flag
        self._transform = transform
        self._exts = ['.jpg', '.jpeg', '.png']
        self._list_images(self._root)

        self.max_sample = len(self.items)
        self.cur = 0
        self.current_batch = [None, None]

        self.provide_data = mx.io.DataDesc(data_names[0], (batch_size, data_shape[0], data_shape[1], data_shape[2]))
        self.provide_label = mx.io.DataDesc(label_names[0], (batch_size, label_shape[0]))

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
        data = ImageFolderDataIter.default_batchify_fn(images)
        targets = ImageFolderDataIter.default_batchify_fn(labels).astype(np.float32)
        print(f'[read_batch] data: {data.shape}  {data.dtype}')
        print(f'[read_batch] targets: {targets.shape}  {targets.dtype}')
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
        # img = cv.imread(self.items[self.cur][0], self._flag)
        # img = cv.resize(img, (128, 128))
        # img = img.transpose((2, 0, 1))
        # img = img[np.newaxis, :, :, :]
        # return img
        return self.current_batch[0]

    def getlabel(self):
        # return self.items[self.cur][1]
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

    @staticmethod
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


class Config(object):
    num_class = 2
    image_dir = r'/home/intellif/Desktop/北京活体检测/20200806/test'
    checkpoint_dir = r'./'
    log_interval = 30
    gpus = '0,1'
    weight_decay = 1e-4
    learning_rate = 1e-3
    min_epoch = 0
    max_epoch = 10
    batch_size = 24
    pretrained_name = './model/eye-hole'
    pretrained_epoch = 0
    with_fit = True


class Train():

    def __init__(self, config):
        self.config = config

        self.ctxs = [mx.gpu(int(i)) for i in config.gpus.split(',')]

        self.net = self.load_pretrained(pretrained=None, epoch=0)
        self.net.collect_params().reset_ctx(self.ctxs)

        self.trainer = mx.gluon.Trainer(
            self.net.collect_params(),
            'sgd',
            {'learning_rate': config.learning_rate, 'wd': config.weight_decay}
        )

        if self.config.with_fit:
            self.train_iter = ImageFolderDataIter(
                root=self.config.image_dir,
                data_shape=(3, 128, 128),
                label_shape=(2,),
                data_names=['data'],
                label_names=['softmax_label'],
                flag=1,
                transform=lambda data, label: (data.astype(np.float32)/255, label),
                batch_size=self.config.batch_size
            )
            # data_iter = iter(self.train_iter)
            # batch = next(data_iter)
            # print(f'[] label: {batch.label}')
            # # sys.exit(0)
            # self.train_dataset = ImageFolderDataset(
            #     root=config.image_dir,
            #     flag=1,
            #     transform=lambda data, label: (data.astype(np.float32)/255, label)
            # )
            # self.train_dataloader = mx.gluon.data.DataLoader(
            #     dataset=self.train_dataset,
            #     batch_size=self.config.batch_size,
            #     shuffle=True
            # )
            # self.eval_dataset = ImageFolderDataset(
            #     root=config.image_dir,
            #     flag=1,
            #     transform=lambda data, label: (data.astype(np.float32)/255, label)
            # )
            # self.eval_dataloader = mx.gluon.data.DataLoader(
            #     dataset=self.eval_dataset,
            #     batch_size=self.config.batch_size,
            #     shuffle=True
            # )
            # self.train_iter = mx.contrib.io.DataLoaderIter(self.train_dataloader)
        else:
            self.train_dataset = ImageFolderDataset(
                root=config.image_dir,
                flag=1,
                transform=lambda data, label: (data.astype(np.float32)/255, label)
            )
            self.train_dataloader = mx.gluon.data.DataLoader(
                dataset=self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )
            self.eval_dataset = ImageFolderDataset(
                root=config.image_dir,
                flag=1,
                transform=lambda data, label: (data.astype(np.float32)/255, label)
            )
            self.eval_dataloader = mx.gluon.data.DataLoader(
                dataset=self.eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )

        self.sec_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()

        self.train_acc = mx.metric.Accuracy()
        self.eval_acc = mx.metric.Accuracy()

    def load_pretrained(self, pretrained=None, epoch=0):
        if pretrained is None:
            net = mx.gluon.model_zoo.vision.mobilenet_v2_1_0(classes=1000, pretrained=True, ctx=mx.cpu(), root=r'./model')
            net.collect_params().initialize(init=mx.init.Xavier(), ctx=[mx.cpu()], force_reinit=True)  # 多卡同时初始化
            net.output = mx.gluon.nn.HybridSequential()
            net.output.add(
                mx.gluon.nn.Conv2D(kernel_size=1, channels=self.config.num_class, strides=1),
                mx.gluon.nn.Flatten()
            )
        else:
            net = mx.gluon.SymbolBlock.imports(pretrained+'-symbol.json', ['data'], pretrained+f'-{epoch:04d}.params', ctx=mx.cpu())
            print(f'[load_pretrained] net: {net}')
            # sys.exit(0)
        net.output.initialize(ctx=[mx.cpu()])
        # print(f'[] net: {net}')
        print(f'{net.summary(mx.nd.uniform(shape=(1,3,128,128)))}')

        net.hybridize()
        x = mx.nd.uniform(shape=(1, 3, 128, 128), ctx=(mx.cpu()))
        y = net(x)
        print(f'x: {x.shape}  y: {y.shape}')
        net.export(self.config.pretrained_name, 0)
        return net

    def save_model(self, model_prefix, model_epoch=0):
        self.net.export(model_prefix, model_epoch)

    def train_with_fit(self):
        sym, arg_params, aux_params = mx.model.load_checkpoint(
            prefix=self.config.pretrained_name,
            epoch=self.config.pretrained_epoch
        )
        softmax_sym = mx.sym.Softmax(data=sym, name='softmax')
        print(f'[train_with_fit] sym: {sym}')
        print(f'[train_with_fit] softmax_sym: {softmax_sym}')
        model = mx.mod.Module(
            symbol=softmax_sym,
            context=self.ctxs
        )
        print(f'[0] {model.output_names}')
        print(f'[0] {model.label_names}')
        print(f'[0] {model.data_names}')
        print(f'[0] {self.train_iter.provide_data}')
        print(f'[0] {self.train_iter.provide_label}')
        print(f'[0] model.data_names: {model.data_names}')
        print(f'[0] model.label_names: {model.label_names}')
        data_shapes = [('data', (self.config.batch_size, 3, 128, 128))]
        label_shapes = [('softmax_label', (self.config.batch_size, ))]
        print(f'[0] data_shapes: {data_shapes}')
        print(f'[0] label_shapes: {label_shapes}')
        model.bind(
            # data_shapes=self.train_iter.provide_data,  # [('data', (self.config.batch_size, 3, 128, 128))],
            # label_shapes=self.train_iter.provide_label,  # [('softmax_label', (self.config.batch_size, self.config.num_class))],
            # data_shapes=self.train_iter.provide_data,  # [('data', (self.config.batch_size, 3, 128, 128))],
            # label_shapes=self.train_iter.provide_label  # [('softmax_label', (self.config.batch_size, self.config.num_class))],
            data_shapes=data_shapes,
            label_shapes=label_shapes,
        )
        print(f'[1] {model.output_names}')
        print(f'[1] {model.label_names}')
        print(f'[1] {model.label_names}  {model.label_shapes}')
        print(f'[1] {model.data_names}  {model.data_shapes}')
        model.set_params(arg_params=arg_params, aux_params=aux_params, allow_missing=False)
        model.fit(
            train_data=self.train_iter,
            optimizer='sgd',
            optimizer_params={'learning_rate': self.config.learning_rate, 'wd': self.config.weight_decay},
            num_epoch=self.config.max_epoch,
            batch_end_callback=[
                mx.callback.Speedometer(self.config.batch_size, 1),
            ],
            epoch_end_callback=mx.callback.do_checkpoint(self.config.pretrained_name, 1),
            eval_metric='acc'
        )

    def train(self):

        for epoch in range(self.config.min_epoch, self.config.max_epoch):
            total_loss = 0
            for idx, (feature, label) in enumerate(self.train_dataloader):
                gpu_datas = mx.gluon.utils.split_and_load(feature, self.ctxs)
                gpu_labels = mx.gluon.utils.split_and_load(label, self.ctxs)
                with mx.autograd.record():
                    # losses = [self.sec_loss(self.net(x), y) for x, y in zip(gpu_datas, gpu_labels)]
                    losses = []
                    for x, y in zip(gpu_datas, gpu_labels):
                        print(f'[train] x: {x.shape}  y: {y.shape}  {x.context}')
                        preds = self.net(x)
                        loss = self.sec_loss(preds, y)
                        print(f'loss: {loss.context}')
                        losses.append(loss)
                    cur_loss = sum([loss.sum().asscalar() for loss in losses])
                    total_loss += cur_loss
                for loss in losses:
                    loss.backward()
                self.trainer.step(self.config.batch_size)
                mx.nd.waitall()
                if idx % self.config.log_interval == 0:
                    print(f'[TRAIN] [{epoch}/{idx}]  loss: {total_loss:8f} / {cur_loss:8f}')
            self.save_model(self.config.pretrained_name, epoch+1)

            eval_loss = 0
            for idx, (feature, label) in enumerate(self.eval_dataloader):
                gpu_datas = mx.gluon.utils.split_and_load(feature, self.ctxs)
                gpu_labels = mx.gluon.utils.split_and_load(label, self.ctxs)
                # losses = [self.sec_loss(self.net(x), y) for x, y in zip(gpu_datas, gpu_labels)]
                for x, y in zip(gpu_datas, gpu_labels):
                    preds = self.net(x)
                    losses.append(self.sec_loss(preds, y))
                    self.eval_acc.update(y, preds)
                cur_loss = sum([loss.sum().asscalar() for loss in losses])
                eval_loss += cur_loss
                if idx % self.config.log_interval == 0:
                    print(f'[EVAL] [{epoch}/{idx}]  {self.eval_acc.get()[0]}:{self.eval_acc.get()[1]:8f}  {self.eval_acc.get()[0]}:{self.eval_acc.get()[1]:8f}  loss: {eval_loss:8f} / {cur_loss:8f}')

    def eval(self):
        pass


if __name__ == '__main__':

    config = Config()

    trainer = Train(config)

    # trainer.train()
    trainer.train_with_fit()