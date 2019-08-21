from yolo_utility import yolo_to_BBCor, yolo_readline
from img_utility import cut_by_BBCor, cut_by_fourpts, cor_sys_trans
from CCPD_utility import vertices_info, BBCor_info, FR_vertices_info
import cv2
from os.path import basename

img_path = '/home/shaoheng/Documents/Thesis_KSH/training_data/CCPD_FR/602&283_281&296_284&193_605&180.jpg'
yolo_label = '/home/shaoheng/Documents/cars_label_FRNet/CCPD_2019_first_part/0to4370/0036-0_0-323&482_403&520-402&520_323&519_324&482_403&483-0_0_27_10_26_24_26-79-3.txt'
img = cv2.imread(img_path)

FR_img = cut_by_fourpts(img, *FR_vertices_info(basename(img_path)))

cv2.imshow('img', FR_img)
cv2.waitKey(0)