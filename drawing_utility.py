import cv2
import numpy as np
from img_utility import read_img_from_dir
from CCPD_utility import FR_vertices_info, BBCor_info, vertices_info


# img -> cv2.imread format, pts -> np.array([[x1, y1], [x2, y2], [x2, y2], [x2, y2])
def draw_LP_by_vertices(img, pts):
    cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 234), thickness=4)
    return img


if __name__ == '__main__':
    dir = '/home/shaoheng/Documents/cars_label_FRNet/CCPD_2019_first_part/0to4370'
    imgs_paths = read_img_from_dir(dir)

    for img_path in imgs_paths:
        img = cv2.imread(img_path)
        vertices = np.array(vertices_info(img_path))
        img_draw = draw_LP_by_vertices(img, vertices)
        cv2.imshow('img', img_draw)
        cv2.moveWindow('img', 0, 0)
        cv2.waitKey(0)