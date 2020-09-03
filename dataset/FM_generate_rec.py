#! /usr/bin/env python
# coding: utf-8
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import mxnet as mx
import cv2 as cv
from dataset.mx_rec import MXRec
import utils.cv_show as CVShow
from concurrent.futures import ThreadPoolExecutor, as_completed


class FMGenerateRec(MXRec):
    def __init__(self):
        pass

    @staticmethod
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

    @staticmethod
    def check_rec_file(rec_dir, rec_prefix):
        rec_handler = MXRec(rec_dir=rec_dir, prefix=rec_prefix)
        lines = None
        rec_file = os.path.join(rec_dir, rec_prefix + '.rec')
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
                image = cv.circle(
                    img=image,
                    center=(int(round(point[0])), int(round(point[1]))),
                    radius=1,
                    color=(0, 0, 255)
                )
            CVShow.cv_show_image(image, wait_time=300)

    @staticmethod
    def generate_rec_file(image_list, rec_dir, rec_prefix):
        rec_handler = MXRec(rec_dir=rec_dir, prefix=rec_prefix)
        for index, image_file in enumerate(image_list):
            image = cv.imread(image_file)
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
            points = np.fromstring(string=line, dtype=np.float, sep='\t')  ##.reshape(68, 2)
            # print(f'points: {points.shape}  {points}')
            profile = np.zeros(image.shape, dtype=np.uint8)
            points_ = points.copy().reshape(68, 2)
            ps = points_[0:17, :]
            for i in range(16):
                cv.line(profile, (int(round(ps[i][0])), int(round(ps[i][1]))),
                         (int(round(ps[i + 1][0])), int(round(ps[i + 1][1]))), (0, 0, 255), thickness=3)
            ps = points_[17:22, :]
            for i in range(4):
                cv.line(profile, (int(round(ps[i][0])), int(round(ps[i][1]))),
                         (int(round(ps[i + 1][0])), int(round(ps[i + 1][1]))), (0, 0, 255), thickness=3)
            ps = points_[22:27, :]
            for i in range(4):
                cv.line(profile, (int(round(ps[i][0])), int(round(ps[i][1]))),
                         (int(round(ps[i + 1][0])), int(round(ps[i + 1][1]))), (0, 0, 255), thickness=3)
            ps = points_[27:31, :]
            for i in range(3):
                cv.line(profile, (int(round(ps[i][0])), int(round(ps[i][1]))),
                         (int(round(ps[i + 1][0])), int(round(ps[i + 1][1]))), (0, 0, 255), thickness=3)
            ps = points_[31:36, :]
            for i in range(4):
                cv.line(profile, (int(round(ps[i][0])), int(round(ps[i][1]))),
                         (int(round(ps[i + 1][0])), int(round(ps[i + 1][1]))), (0, 0, 255), thickness=3)
            ps = points_[36:42, :]
            for i in range(5):
                cv.line(profile, (int(round(ps[i][0])), int(round(ps[i][1]))),
                         (int(round(ps[i + 1][0])), int(round(ps[i + 1][1]))), (0, 0, 255), thickness=2)
            cv.line(profile, (int(round(ps[-1][0])), int(round(ps[-1][1]))),
                     (int(round(ps[0][0])), int(round(ps[0][1]))), (0, 0, 255), thickness=2)
            ps = points_[42:48, :]
            for i in range(5):
                cv.line(profile, (int(round(ps[i][0])), int(round(ps[i][1]))),
                         (int(round(ps[i + 1][0])), int(round(ps[i + 1][1]))), (0, 0, 255), thickness=2)
            cv.line(profile, (int(round(ps[-1][0])), int(round(ps[-1][1]))),
                     (int(round(ps[0][0])), int(round(ps[0][1]))), (0, 0, 255), thickness=2)
            ps = points_[48:68, :]
            for i in range(19):
                cv.line(profile, (int(round(ps[i][0])), int(round(ps[i][1]))),
                         (int(round(ps[i + 1][0])), int(round(ps[i + 1][1]))), (0, 0, 255), thickness=2)
            cv.line(profile, (int(round(ps[-1][0])), int(round(ps[-1][1]))),
                     (int(round(ps[0][0])), int(round(ps[0][1]))), (0, 0, 255), thickness=2)

            rec_handler.write_rec({'image': image, 'profile': profile[:, :, 2], 'points': points})

    @staticmethod
    def generate_rec_file_with_multi_thread(image_list, rec_dir, rec_prefix):
        from concurrent.futures import ThreadPoolExecutor
        from queue import Queue

        def load_data(q):
            for index, image_file in enumerate(image_list):
                image = cv.imread(image_file)
                # print('[{}/{}]  image: {}'.format(index, len(image_list), image.shape))
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
                points = np.fromstring(string=line, dtype=np.float, sep='\t')  ##.reshape(68, 2)
                # print(f'points: {points.shape}  {points}')
                profile = np.zeros(image.shape, dtype=np.uint8)
                points_ = points.copy().reshape(68, 2)
                left_eye_point = np.mean(points_[36:42, :], axis=0)
                right_eye_point = np.mean(points_[42:48, :], axis=0)
                nose_point = points_[30, :]
                left_mouth = points_[48, :]
                right_mouth = points_[54, :]
                key_points = np.stack([left_eye_point, right_eye_point, nose_point, left_mouth, right_mouth])
                # img = CVShow.cv_draw_points(image, key_points)
                # CVShow.cv_show_image(img, wait_time=0)

                print('[{}/{}]  image: {}'.format(index, len(image_list), image.shape))
                q.put({'image':image, 'keypoints':key_points.reshape((-1)), 'points':points, 'image_file':image_file})
            print(f'[load_data] load images complete.')
            q.put({'image':None, 'keypoints':None, 'points':None})
            # return image, profile, points

        def dump_data(q, rec_handler):
            while True:
                info = q.get()
                image = info['image']
                keypoints = info['keypoints']
                points = info['points']
                image_file = info['image_file']
                if image is None and keypoints is None and points is None:
                    break
                print(f'[dump_data] dump {image_file}')
                rec_handler.write_rec({'image': image, 'keypoints': keypoints, 'points': points})
            print(f'[dump_data] dump data complete.')

        with ThreadPoolExecutor(2) as excutor:
            q = Queue()
            rec_handler = MXRec(rec_dir=rec_dir, prefix=rec_prefix)
            excutor.submit(load_data, q)
            excutor.submit(dump_data, q, rec_handler)





if __name__ == '__main__':
    root_dir = r'/data1/dataset/300/300vw'
    image_list_file = r'/data1/dataset/300/rec/002/300vw_1_1.lst'
    rec_dir = r'/data1/dataset/300/rec/002'
    rec_prefix = '300vw_1_1'
    # root_dir = r'/media/bill/cc18c1ef-ab04-47e4-ad1f-f431ab2785a3/datasets/300vw/test'
    # image_list_file = r'./test.lst'
    # rec_dir = './'
    # rec_prefix = 'test'
    # FMGenerateRec.generate_rec_file(
    #     image_list=FMGenerateRec.get_image_list(root_dir, image_list_file),
    #     rec_dir=rec_dir,
    #     rec_prefix=rec_prefix
    # )
    FMGenerateRec.generate_rec_file_with_multi_thread(
        image_list=FMGenerateRec.get_image_list(root_dir, image_list_file),
        rec_dir=rec_dir,
        rec_prefix=rec_prefix
    )

