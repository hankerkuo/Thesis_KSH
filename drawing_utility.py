import cv2
import numpy as np
from img_utility import read_img_from_dir
from CCPD_utility import FR_vertices_info, BBCor_info, vertices_info


# img -> cv2.imread format, pts -> np.array([[x1, y1], [x2, y2], [x2, y2], [x2, y2]])
def draw_LP_by_vertices(img, pts):
    cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 234), thickness=4)
    return img


if __name__ == '__main__':
    pass
