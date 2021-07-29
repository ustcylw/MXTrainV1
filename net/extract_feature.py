#! /usr/bin/env python
# coding: utf-8
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from utils.bbox.bbox import BBoxes


class ExtractFeature(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, features, bboxes, scale=1.0):
        feature = features[0:1, :, :, :]
        # print(f'[ExtractFeature][forward]  feature: {feature.shape}')
        
        if scale == 1.0:
            rois_bboxes = bboxes.copy_with_fields()
        else:
            rois_bboxes = bboxes.copy_with_fields().scale(scale)
        rois_bboxes.format(feature.shape[3], feature.shape[2])
        
        rois = rois_bboxes.crop(feature)
        
        return rois




if __name__ == "__main__":
    

    import cv2
    
    scale = 1.1
    b = np.array([[180, 40, 260, 230], [50, 110, 370, 290]], dtype=np.float32)
    bboxes = BBoxes(b, mode='xyxy')
    feat = cv2.imread('/data2/personal/docs/MXTrainV1/net/test/000022.jpg')
    feat = feat.transpose((2, 0, 1))[np.newaxis, :, :, :]
    feat_copy = feat.copy()
    print(f'feat: {feat.shape}')
    
    ext = ExtractFeature()
    rois = ext.forward(feat, bboxes, scale=scale)
    for i, roi in enumerate(rois):
        print(f'roi-{i}: {roi.shape}')
        # cv2.rectangle(feat, (int(), int()), (int(), int()), (0, 0, 255), thickness=1.0)
        cv2.imshow(f'roi-{i}', roi[0].transpose((1, 2, 0)))
        if cv2.waitKey(0) == ord('q'):
            continue