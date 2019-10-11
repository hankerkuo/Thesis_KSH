import cv2
import numpy as np

from img_utility import read_img_from_dir, cut_by_fourpts
from dataset_utility import CCPD_FR_vertices_info, CCPD_vertices_info



# convert the hsv information from GIMP to cv2 hsv system
def hsv_from_GIMP_to_cv2(h, s, v):
    return np.floor(1. * h / 360 * 179).astype(int), \
           np.floor(1. * s / 100 * 255).astype(int), \
           np.floor(1. * v / 100 * 255).astype(int)


# return a lower and upper hsv, based on the LP part of CCPD database, manually picked by GIMP
# the file CCPD_LP_hsv_GIMP.txt has lines of hsv values in the form -> h,s,v
def eyeballing_hsv():
    hsvs = []
    with open('/home/shaoheng/Documents/Thesis_KSH/data_augmentation/CCPD_LP_hsv_GIMP.txt', 'r') as f:
        for line in f.readlines():
            hsvs.append(map(float, line[:-1].split(',')))
    hsvs = np.array([(hsv_from_GIMP_to_cv2(*hsv))for hsv in hsvs])
    hsvs_min = np.array([min(hsvs[:, 0]), min(hsvs[:, 1]), min(hsvs[:, 2])])
    hsvs_max = np.array([max(hsvs[:, 0]), max(hsvs[:, 1]), max(hsvs[:, 2])])
    return hsvs_min, hsvs_max


# segment the img (in cv2.imread format, color space BGR) by bounding of hsv color space
# hsv_lower and hsv_upper -> np.array([h, s, v])
def hsv_segmentation(img, hsv_lower, hsv_upper):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img, hsv_lower, hsv_upper)
    img_after_seg = cv2.bitwise_and(img, img, mask=mask)
    return img_after_seg


if __name__ == '__main__':
    path = '/home/shaoheng/Documents/cars_label_FRNet/ccpd_dataset/ccpd_base'
    imgs_paths = read_img_from_dir(path)

    hsv_lower, hsv_upper = eyeballing_hsv()

    for img_path in imgs_paths:
        img = cv2.imread(img_path)
        LP_part = CCPD_vertices_info(img_path)
        img_LP = cut_by_fourpts(img, *LP_part)
        img_after_seg = hsv_segmentation(img, hsv_lower, hsv_upper)

        cv2.imshow('img', img_after_seg)
        cv2.moveWindow('img', 0, 0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()