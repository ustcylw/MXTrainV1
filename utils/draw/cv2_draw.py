import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2


def draw_rectangle(img, boxes, color=(0, 0, 255), thickness=1):
    for box in boxes:
        cv2.rectangle(
            img, 
            (int(box[0]), int(box[1])), 
            (int(box[2]), int(box[3])), 
            color=color, 
            thickness=thickness
        )
    return img


def draw_point(img, points, radius=1, color=(0, 0, 255), thickness=1):
    for point in points:
        cv2.circle(
            img, 
            (int(point[0]), int(point[1])),
            radius=radius, 
            color=color, 
            thickness=thickness
        )
    return img


def draw_image(image, wait=0):
    cv2.imshow('image', image)
    if cv2.waitKey(wait) == ord('q'):
        return 0
    return 0


def draw_text(image, pos, text, color=(0, 0, 255)):
    return cv2.putText(image, text, pos, fontScale=1.0, color=color, fontFace=cv2.FONT_HERSHEY_COMPLEX)
