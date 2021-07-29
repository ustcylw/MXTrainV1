import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import cv2
import numpy as np


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(
    center,
    scale,
    rot,
    output_size,
    shift=np.array([0, 0], dtype=np.float32),
    inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def affine_transform(pts, t):
    new_pts = np.column_stack((pts, np.ones(shape=(pts.shape[0],))))
    new_pts = np.dot(t, new_pts.T)
    return new_pts[:2]


def affine_image(image, points_list=[], dst_shape=(512, 512)):
    scale = max(image.shape)
    rot = 0
    w, h = dst_shape
    shift = np.array([0, 0], dtype=np.float32),
    inv = 0
    center = np.array([image.shape[1]/2, image.shape[0]/2])

    trans_input = get_affine_transform(
        center,
        scale,
        0,
        [w, h]
    )
    
    image_new = cv2.warpAffine(
        image,
        trans_input,
        (w, h),
        flags=cv2.INTER_LINEAR
    )
    
    points_list_new = []
    for points in points_list:
        if len(points) == 0:
            points_list_new.append(np.array([]))
            continue
        points_new = np.zeros_like(points)
        print(f'')
        points_new[:, :2] = affine_transform(points[:, :2], trans_input).T
        points_list_new.append(points_new)

    return image_new, points_list_new


if __name__ == '__main__':

    import utils.type.type_utils as TUtils
    import utils.draw.cv2_draw as CVDraw

    image_file = '/data2/personal/centernet/test_git_centernet/Lightweight-face-detection-CenterNet/imgs/2.jpg'
    bboxes = np.array([[100, 75, 245, 250], [288, 92, 431, 224], [541, 43, 682, 213], [701, 66, 868, 276]])
    centers = np.zeros(shape=(4, 2))
    centers[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2
    centers[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2

    image = cv2.imread(image_file)
    print(f'image: {image.shape}')

    image_copy = image.copy()
    image_copy = CVDraw.draw_rectangle(image_copy, bboxes, thickness=3)
    image_copy = CVDraw.draw_point(image_copy, centers, radius=3, thickness=3)
    CVDraw.draw_image(image_copy, wait=0)
    # plt.imshow(image_copy)
    # plt.show()

    image_ori = image.copy()
    points = bboxes.copy().reshape(-1, 2)
    image_new, points_new = affine_image(image_ori, [points], dst_shape=(256, 128))  # (512, 512))
    # image_new, points_new = affine_image(image_ori, [points[0:2, :], points[2:, :]], dst_shape=(128, 128))  # (512, 512))
    points_new = np.vstack(points_new)
    bboxes_new = points_new.reshape((bboxes.shape[0], bboxes.shape[1]))
    
    centers_new = np.zeros(shape=(4, 2))
    centers_new[:, 0] = (bboxes_new[:, 0] + bboxes_new[:, 2]) / 2
    centers_new[:, 1] = (bboxes_new[:, 1] + bboxes_new[:, 3]) / 2

    image_copy = image_new.copy()
    image_copy = CVDraw.draw_rectangle(image_copy, bboxes_new, thickness=3)
    image_copy = CVDraw.draw_point(image_copy, centers_new, radius=3, thickness=3)
    CVDraw.draw_image(image_copy, wait=0)
    # plt.imshow(image_copy)
    # plt.show()
