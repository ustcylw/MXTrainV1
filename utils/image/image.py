#! /usr/bin/env python
# coding: utf-8
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import cv2
import torch
import utils.torch_utils.device as Device
import utils.transform.transform as Transform
import utils.transform.affine as Affine
import utils.bbox.bbox as BBoxes


class Image(object):
    '''
    # all data are numpy type.
    # only Image.to_device() will get torch.Tensor() data
    # field以bbox为单位
    # field:
    #   {
    #     'field-name': []
    #   }
    '''
    def __init__(self, image, mode='bgr', device=-1):
        super().__init__()
        self.data = None
        self.extra_fields = {}
        self.device = Device.get_device(device)[0]
        self.mode = mode

        if isinstance(image, str):
            assert os.path.isexists(image), f'{image} file is not exist!!!'
            
            data = cv2.imread(image)
            # self.data = Device.to(data, dtype=torch.float32)
            self.extra_fields['filename'] = image
            self.extra_fields['height'] = data.shape[0]
            self.extra_fields['width'] = data.shape[1]
            
        if isinstance(image, np.ndarray):
            self.data = image
            self.extra_fields['filename'] = None
            self.extra_fields['height'] = image.shape[0]
            self.extra_fields['width'] = image.shape[1]
        
        # self.data = self.to_device(self.device)

    def get_data(self):
        return self.to_numpy()
    
    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def update_field(self, field, field_data):
        self.extra_fields.update({field:field_data})
        
    def get_field(self, field):
        if not self.has_field(field):
            return np.array([])
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def copy(self):
        pass
    
    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def to_device(self, device):
        return Device.to(self.data, device)

    def to_numpy(self):
        if isinstance(self.data, np.ndarray):
            return self.data
        return self.data.numpy()
    
    def resize(self, w=-1, h=-1, ratio=1.0):
        if ((w == -1) or (h == -1)) and (ratio == 1.0):
            return self
        
        if w > 0 and h > 0:
            new_w, new_h = w, h
        else:
            new_w, new_h = int(self.get_field('width') * ratio), int(self.get_field('height') * ratio)

        bboxes, face_landmarks, body_landmarks = np.array([]), np.array([]), np.array([])
        if self.has_field('bboxes'):
            bboxes = self.get_field('bboxes').bboxes.reshape((-1, 2))
            bboxes_num = int(bboxes.shape[0] / 2)
        if self.has_field('face_landmarks'):
            face_landmarks = self.get_field('face_landmarks')
            face_landmarks_shape = face_landmarks.shape
        if self.has_field('body_landmarks'):
            body_landmarks = self.get_field('body_landmarks')
            body_landmarks_shape = body_landmarks.shape
        
        image_new, points_new = Affine.affine_image(
            self.get_data(), 
            [bboxes, face_landmarks.reshape((-1, 2)), body_landmarks.reshape(-1, 2)], 
            dst_shape=(new_w, new_h)
        )
        
        self.data = image_new
        if self.has_field('bboxes'):
            self.update_field('bboxes', BBoxes.BBoxes(points_new[0].reshape((-1, 4)), mode=self.get_field('bboxes').mode))
        if self.has_field('face_landmarks'):
            face_landmarks_new = points_new[1].reshape((face_landmarks_shape))
            self.update_field('face_landmarks', face_landmarks_new)
        if self.has_field('body_landmarks'):
            body_landmarks_new = points_new[2].reshape((body_landmarks_shape))
            self.update_field('body_landmarks', body_landmarks_new)
        self.update_field('width', new_w)
        self.update_field('height', new_h)

        return self

    def get_bboxes_center(self):
        if not self.has_field('bboxes'):
            return np.array([])
        
        bboxes = self.get_field('bboxes')
        center_points = np.zeros(shape=(bboxes.shape[0], 2))
        center_points[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2
        center_points[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2

        return center_points
        
        

if __name__ == "__main__":
    
    import utils.type.type_utils as TUtils
    import utils.draw.cv2_draw as CVDraw

    image_file = '/data2/personal/centernet/test_git_centernet/Lightweight-face-detection-CenterNet/imgs/2.jpg'
    bboxes = np.array([[100, 75, 245, 250], [288, 92, 431, 224], [541, 43, 682, 213], [701, 66, 868, 276]])
    centers = np.zeros(shape=(4, 2))
    centers[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2
    centers[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2
    lm = np.array([
            [157, 165, 202, 164, 182, 187, 160, 208, 197, 207],
            [322, 157, 365, 145, 342, 178, 333, 200, 375, 190],
            [572, 131, 621, 136, 588, 162, 568, 171, 610, 178],
            [739, 189, 786, 186, 757, 217, 745, 236, 793, 233]
    ])

    image = cv2.imread(image_file)
    print(f'image: {image.shape}')

    image_copy = image.copy()
    image_copy = CVDraw.draw_rectangle(image_copy, bboxes, color=(0, 255, 0), thickness=3)
    image_copy = CVDraw.draw_point(image_copy, lm.reshape((-1, 2)), color=(255, 0, 0), thickness=3)
    image_copy = CVDraw.draw_point(image_copy, centers, radius=3, thickness=3)
    CVDraw.draw_image(image_copy, wait=0)

    image_ori = image.copy()
    points = bboxes.copy().reshape(-1, 2)
    IMG = Image(image, device=-1)
    bb = BBoxes.BBoxes(bboxes, mode='xyxy')
    IMG.update_field('bboxes', bb)
    IMG.update_field('face_landmarks', lm)

    IMG = IMG.resize(w=512, h=512)

    centers_new = np.zeros(shape=(4, 2))
    centers_new[:, 0] = (IMG.get_field('bboxes').bboxes[:, 0] + IMG.get_field('bboxes').bboxes[:, 2]) / 2
    centers_new[:, 1] = (IMG.get_field('bboxes').bboxes[:, 1] + IMG.get_field('bboxes').bboxes[:, 3]) / 2

    image_copy = IMG.data.copy()
    
    image_copy = CVDraw.draw_rectangle(image_copy, IMG.get_field('bboxes').bboxes, color=(0, 255, 0), thickness=3)
    image_copy = CVDraw.draw_point(image_copy, IMG.get_field('face_landmarks').reshape((-1, 2)), color=(255, 0, 0), thickness=3)
    image_copy = CVDraw.draw_point(image_copy, centers_new, radius=3, thickness=3)
    CVDraw.draw_image(image_copy, wait=0)
