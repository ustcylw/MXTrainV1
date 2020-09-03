#! /usr/bin/env python
# coding: utf-8
import os, sys
import mxnet as mx
import numpy as np
import pickle


class MXRec(object):
    def __init__(self, rec_dir, prefix):
        self.rec_dir = rec_dir
        self.prefix = prefix
        self.rec_file = self.prefix + '.rec'
        self.idx_file = self.prefix + '.idx'
        if not os.path.exists(self.rec_dir):
            os.makedirs(os.path.join(self.rec_dir, self.rec_file))
        if not os.path.exists(os.path.join(self.rec_dir, self.rec_file)):
            with open(os.path.join(self.rec_dir, self.rec_file), 'w+'):
                pass
        if not os.path.exists(os.path.join(self.rec_dir, self.rec_file)):
            with open(os.path.join(self.rec_dir, self.rec_file), 'w+'):
                pass
        self.read_recordio = None
        self.write_recordio = None
        self.idx = 0
        self.num_bytes = 0
        self.idx_list = []
        self.num_bytes_list = []

    def pack(self, data):
        packed_s = pickle.dumps(data)
        return packed_s

    def unpack(self, packed_s):
        data = pickle.loads(packed_s)
        return data

    def write_rec(self, data):
        # idx_string = ''
        if self.write_recordio is None:
            self.write_recordio = mx.recordio.MXIndexedRecordIO(os.path.join(self.rec_dir, self.idx_file), os.path.join(self.rec_dir, self.rec_file), 'w')
        packed_s = self.pack(data=data)
        self.write_recordio.write_idx(self.idx, packed_s)
        self.num_bytes += len(packed_s)
        self.idx_list.append(self.idx)
        self.num_bytes_list.append(self.num_bytes)
        self.idx += 1
        idx_string = str(self.idx) + '\t' + str(self.num_bytes) + '\n'
        with open(os.path.join(self.rec_dir, self.idx_file), 'a+') as df:
            df.write(idx_string)
            df.flush()
            # print('write: {}'.format(idx_string))

    def read_rec(self, idx):
        if self.read_recordio is None:
            self.read_recordio = mx.recordio.MXIndexedRecordIO(os.path.join(self.rec_dir, self.idx_file), os.path.join(self.rec_dir, self.rec_file), 'r')
        packed_s = self.read_recordio.read_idx(idx=idx)
        data = self.unpack(packed_s)
        return data


def test_1():
    a = np.array(np.linspace(0, 5 * 6 * 3 - 1, 5 * 6 * 3).reshape((5, 6, 3)))

    write = False
    read = True

    if write:
        rec = MXRec('./', 'test_idx_')
        for i in range(5):
            d = a + i
            data = {'name':'name_{}'.format(i), 'data1': a+(i*2), 'data2': a+(i*2+1)}
            rec.write_rec(data=data)

    if read:
        rec = MXRec('./', 'test_idx_')
        for i in range(5):
            data = rec.read_rec(i)
            print('data: {}'.format(data))


########################################################################################################################
## generate rec file
########################################################################################################################
import cv2

def generate_rec_file(image_list, rec_dir, rec_prefix):
    rec_handler = MXRec(rec_dir=rec_dir, prefix=rec_prefix)
    for index, image_file in enumerate(image_list):
        image = cv2.imread(image_file)
        print('[{}/{}]  image: {}'.format(index, len(image_list), image.shape))
        # print(f'[{index}/{len(image_list)}]  image: {image.shape}')
        if image_file.endswith('.jpg'):
            point_file = image_file.replace('.jpg', '.pts')
        elif image_file.endswith('.png'):
            point_file = image_file.replace('.png', '.pts')
        else:
            print('image file is not correct!!!')
            continue
        lines = None
        with open(point_file, 'r') as lf:
            lines = lf.readlines()
        line = lines[0].strip()
        points = np.fromstring(string=line, dtype=np.float, sep='\t')##.reshape(68, 2)
        # print(f'points: {points.shape}  {points}')
        profile = np.zeros(image.shape, dtype=np.uint8)
        points_ = points.copy().reshape(68, 2)
        ps = points_[0:17, :]
        for i in range(16):
            cv2.line(profile,(int(round(ps[i][0])), int(round(ps[i][1]))), (int(round(ps[i+1][0])), int(round(ps[i+1][1]))), (0, 0, 255), thickness=3)
        ps = points_[17:22, :]
        for i in range(4):
            cv2.line(profile,(int(round(ps[i][0])), int(round(ps[i][1]))), (int(round(ps[i+1][0])), int(round(ps[i+1][1]))), (0, 0, 255), thickness=3)
        ps = points_[22:27, :]
        for i in range(4):
            cv2.line(profile,(int(round(ps[i][0])), int(round(ps[i][1]))), (int(round(ps[i+1][0])), int(round(ps[i+1][1]))), (0, 0, 255), thickness=3)
        ps = points_[27:31, :]
        for i in range(3):
            cv2.line(profile,(int(round(ps[i][0])), int(round(ps[i][1]))), (int(round(ps[i+1][0])), int(round(ps[i+1][1]))), (0, 0, 255), thickness=3)
        ps = points_[31:36, :]
        for i in range(4):
            cv2.line(profile,(int(round(ps[i][0])), int(round(ps[i][1]))), (int(round(ps[i+1][0])), int(round(ps[i+1][1]))), (0, 0, 255), thickness=3)
        ps = points_[36:42, :]
        for i in range(5):
            cv2.line(profile,(int(round(ps[i][0])), int(round(ps[i][1]))), (int(round(ps[i+1][0])), int(round(ps[i+1][1]))), (0, 0, 255), thickness=2)
        cv2.line(profile, (int(round(ps[-1][0])), int(round(ps[-1][1]))), (int(round(ps[0][0])), int(round(ps[0][1]))), (0, 0, 255), thickness=2)
        ps = points_[42:48, :]
        for i in range(5):
            cv2.line(profile,(int(round(ps[i][0])), int(round(ps[i][1]))), (int(round(ps[i+1][0])), int(round(ps[i+1][1]))), (0, 0, 255), thickness=2)
        cv2.line(profile, (int(round(ps[-1][0])), int(round(ps[-1][1]))), (int(round(ps[0][0])), int(round(ps[0][1]))), (0, 0, 255), thickness=2)
        ps = points_[48:68, :]
        for i in range(19):
            cv2.line(profile,(int(round(ps[i][0])), int(round(ps[i][1]))), (int(round(ps[i+1][0])), int(round(ps[i+1][1]))), (0, 0, 255), thickness=2)
        cv2.line(profile,(int(round(ps[-1][0])), int(round(ps[-1][1]))), (int(round(ps[0][0])), int(round(ps[0][1]))), (0, 0, 255), thickness=2)

        rec_handler.write_rec({'image':image, 'profile':profile[:, :, 2], 'points':points})

def get_image_list(root_dir, image_list_file):
        image_list = []
        if os.path.exists(image_list_file):
            lines = None
            with open(image_list_file, 'r') as lf:
                lines = lf.readlines()
            for line in lines:
                image_list.append(line.strip())
            return image_list
        subjects = os.listdir(root_dir)
        for subject in subjects:
            subject_dir = os.path.join(root_dir, subject)
            if not os.path.isdir(subject_dir):
                continue
            file_list = os.listdir(subject_dir)
            for file in file_list:
                _, ext = os.path.splitext(file)
                if ext in ['.jpg', '.png']:
                    image_file = os.path.join(subject_dir, file)
                    point_file = image_file
                    if ext == '.jpg':
                        point_file = point_file.replace('.jpg', '.pts')
                    elif ext == '.png':
                        point_file = point_file.replace('.png', '.pts')
                    else:
                        print('image file is not correct!!!')
                        continue
                    if not os.path.exists(image_file):
                        continue
                    if not os.path.exists(point_file):
                        continue
                    image_list.append(image_file)
        print('image list: ', len(image_list))
        to_string = ''
        for image_file in image_list:
            to_string += image_file
            to_string += '\n'
        with open(image_list_file, 'w') as df:
            df.write(to_string)
            df.flush()
        return image_list

def check_rec_file(rec_dir, rec_prefix):
    rec_handler = MXRec(rec_dir=rec_dir, prefix=rec_prefix)
    lines = None
    rec_file = os.path.join(rec_dir, rec_prefix+'.rec')
    idx_file = ''.join(rec_file).replace('.rec', '.idx')
    with open(idx_file, 'r') as lf:
        lines = lf.readlines()
    idx_list = []
    for line in lines:
        line = line.strip()
        idx, num_bytes = line.split('\t')
        idx, num_bytes = int(idx), int(num_bytes)
        idx_list.append(idx)
    for i in range(len(idx_list)):
        data = rec_handler.read_rec(i)
        image = data['image']
        points = data['points'].reshape((68, 2))
        for i in range(68):
            point = points[i, :]
            image = cv2.circle(
                img=image,
                center=(int(round(point[0])), int(round(point[1]))),
                radius=1,
                color=(0, 0, 255)
            )
        cv2.imshow('image', image)
        if cv2.waitKey(1000) == ord('q'):
            sys.exit(0)

if __name__ == '__main__':

    ## test-1
    # test_1()


    if True:
        # generate facial landmark train data
        root_dir = r'/data1/dataset/300/300vw'
        image_list_file = r'/data1/dataset/300/rec/002/300vw_1_1.lst'
        rec_dir = r'/data1/dataset/300/rec/002'
        rec_prefix = '300vw_1_1'
        # root_dir = r'/media/bill/cc18c1ef-ab04-47e4-ad1f-f431ab2785a3/datasets/300vw/test'
        # image_list_file = r'./test.lst'
        # rec_dir = './'
        # rec_prefix = 'test'
        generate_rec_file(
            image_list=get_image_list(root_dir, image_list_file),
            rec_dir=rec_dir,
            rec_prefix=rec_prefix
        )
    else:
        # generate facial landmark eval data
        # root_dir = r'/data1/dataset/300/300w'
        # image_list_file = r'/data1/dataset/300/rec/300w_001/300w.lst'
        # rec_dir = '/data1/dataset/300/rec/300w_001'
        # rec_prefix = '300w'
        root_dir = r'/data1/dataset/300/300vw-test'
        image_list_file = r'/data1/dataset/300/rec/300vw-test/300vw-test.lst'
        rec_dir = '/data1/dataset/300/rec/300vw-test'
        rec_prefix = '300vw-test'
        generate_rec_file(
            image_list=get_image_list(root_dir, image_list_file),
            rec_dir=rec_dir,
            rec_prefix=rec_prefix
        )


    # check_rec_file(rec_dir=rec_dir, rec_prefix=rec_prefix)
