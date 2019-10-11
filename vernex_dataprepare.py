"""
vernex dataset
used for training vernex
format -> lpx1&lpy1_lpx2&lpy2_lpx3&lpy3_lpx4&lpy4_frx1&fry1_frx2&fry2_frx3&fry3_frx4&fry4_class.jpg
class: front, rear, IMPORTANT-> but in label encoding, background=0, front=1, rear=2
"""
from os.path import splitext, isfile, join

import cv2
import traceback

from yolo_utility import yolo_readline, yolo_to_BBCor, yolo_class
from img_utility import read_img_from_dir, BBCor_to_pts
from dataset_utility import CCPD_vertices_info, CCPD_FR_vertices_info, json_lp_fr


def vernex_from_json():
    in_dir = '/home/shaoheng/Documents/Thesis_KSH/samples/kr_tilt'
    out_dir = '/home/shaoheng/Documents/Thesis_KSH/training_data/kr_tilt_vernex'
    imgs_paths = read_img_from_dir(in_dir)

    for img_path in imgs_paths:
        json_path = splitext(img_path)[0] + '.json'
        if not isfile(json_path):
            continue

        print 'processing', img_path

        try:
            w, h, cls, lp_fr_vertices = json_lp_fr(json_path)
        except AssertionError:
            print traceback.print_exc()
            continue

        img = cv2.imread(img_path)
        # acquiring fr bbcor part
        fr_vertices = lp_fr_vertices[cls]

        # acquiring lp vertices part
        lp_vertices = lp_fr_vertices['lp']

        # acquiring class, front=0, rear=1
        '''For CCPD, openALPR FR'''
        class_name = cls

        # write to new name
        file_name = ''
        # make the file name format similar to CCPD files
        for i, vertex in enumerate(lp_vertices + fr_vertices):
            file_name += str(vertex[0]) + '&' + str(vertex[1])
            if i == 7:
                file_name = file_name + '_' + class_name + '.jpg'
            else:
                file_name += '_'

        print 'write to', join(out_dir, file_name)
        cv2.imwrite(join(out_dir, file_name), img)


def vernex_from_datasets():
    in_dir = '/home/shaoheng/Documents/cars_label_FRNet/openALPR_refined/FR/us'
    out_dir = '/home/shaoheng/Documents/Thesis_KSH/training_data/vernex'
    imgs_paths = read_img_from_dir(in_dir)

    for img_path in imgs_paths:
        txt_path = splitext(img_path)[0] + '.txt'
        if not isfile(txt_path):
            continue

        print 'processing', img_path

        # acquiring fr bbcor part
        '''For CCPD, openALPR FR'''
        img = cv2.imread(img_path)
        fr_bbcor = yolo_to_BBCor(img.shape, *yolo_readline(txt_path))
        fr_vertices = BBCor_to_pts(*fr_bbcor)

        # acquiring lp vertices part, different for each dataset
        '''For CCPD'''
        lp_vertices = CCPD_vertices_info(img_path)

        '''For openALPR FR
        lp_vertices = CCPD_FR_vertices_info(img_path)'''

        # acquiring class, front=0, rear=1
        '''For CCPD, openALPR FR'''
        fr_class = yolo_class(txt_path)
        if fr_class == 0:
            class_name = 'front'
        elif fr_class == 1:
            class_name = 'rear'

        # write to new name
        file_name = ''
        # make the file name format similar to CCPD files
        for i, vertex in enumerate(lp_vertices + fr_vertices):
            file_name += str(vertex[0]) + '&' + str(vertex[1])
            if i == 7:
                file_name = file_name + '_' + class_name + '.jpg'
            else:
                file_name += '_'

        print 'write to', join(out_dir, file_name)
        cv2.imwrite(join(out_dir, file_name), img)


if __name__ == '__main__':
    vernex_from_json()
