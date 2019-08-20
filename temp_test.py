from yolo_utility import yolo_to_BBCor, yolo_readline
from img_utility import cut_by_BBCor, cut_by_fourpts, cor_sys_trans
from CCPD_utility import vertices_info, BBCor_info
import cv2
import numpy as np

img_path = '/home/shaoheng/Documents/cars_label_FRNet/CCPD_2019_first_part/0to4370/0036-0_0-323&482_403&520-402&520_323&519_324&482_403&483-0_0_27_10_26_24_26-79-3.jpg'
yolo_label = '/home/shaoheng/Documents/cars_label_FRNet/CCPD_2019_first_part/0to4370/0036-0_0-323&482_403&520-402&520_323&519_324&482_403&483-0_0_27_10_26_24_26-79-3.txt'
img = cv2.imread(img_path)

FR_pts = yolo_readline(yolo_label)
BBCor = yolo_to_BBCor(np.shape(img), *FR_pts)
car_img = cut_by_BBCor(img, *BBCor)

LP_pts = BBCor_info(img_path)
# LP_img = cut_by_fourpts(img, *LP_pts)

LP_pts = cor_sys_trans(BBCor[0], *LP_pts)
LP_img = cut_by_BBCor(car_img, *LP_pts)

cv2.imshow('img', LP_img)
cv2.waitKey(0)