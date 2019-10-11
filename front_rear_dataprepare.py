"""
this script is used to cut the CCPD image files into FR part with corresponding vertices information
just need to give the input image directory and output directory
"""
from os.path import basename, join, isfile

import cv2
import numpy as np

from dataset_utility import CCPD_vertices_info, openALPR_BBCor_info, CCPD_FR_vertices_info
from yolo_utility import yolo_readline, yolo_to_BBCor
from img_utility import cut_by_BBCor, read_img_from_dir, cor_sys_trans, BBCor_to_pts


if __name__ == '__main__':
    output_dir = '/home/shaoheng/Documents/Thesis_KSH/training_data/CCPD_FR_tilt'
    input_dir = '/home/shaoheng/Documents/cars_label_FRNet/CCPD_2019_first_part/rotandtilt_2000'

    img_paths = read_img_from_dir(input_dir)
    print 'reading all image file paths from input dir done'

    for img_path in img_paths:
        txt_path = img_path.split('.')[0] + '.txt'
        # skip when there is no annotation txt file (in here, txt file includes front-rear in YOLO format)
        if not isfile(txt_path):
            continue

        img = cv2.imread(img_path)
        print 'processing', img_path

        # find the front-rear BB coordinate using YOLO label
        BBCor = yolo_to_BBCor(np.shape(img), *yolo_readline(txt_path))
        FR_img = cut_by_BBCor(img, *BBCor)

        LP_vertices = CCPD_vertices_info(basename(img_path))
        LP_vertices = cor_sys_trans(BBCor[0], *LP_vertices)

        file_name = ''
        # make the file name format similar to CCPD files
        for i, vertex in enumerate(LP_vertices):
            file_name += str(vertex[0]) + '&' + str(vertex[1])
            if i == 3:
                file_name += '.jpg'
            else:
                file_name += '_'

        cv2.imwrite(join(output_dir, file_name), FR_img)
