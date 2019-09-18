import numpy as np
import imgaug.augmenters as iaa
from img_utility import read_img_from_dir
from dataset_utility import CCPD_FR_vertices_info, CCPD_vertices_info, CCPD_FR_front_rear_info
from drawing_utility import draw_LP_by_vertices
import cv2
from config import Configs


# now CCPD_origin=True is not support for the front-rear BB augmentation
def data_aug(img_paths, CCPD_origin=False):
    c = Configs()

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    imgs = []
    key_pts = []

    for img_path in img_paths:
        img = cv2.imread(img_path)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)

        if CCPD_origin:
            if c.model_code in ['Hourglass+Vernex_lp', 'Hourglass+WPOD', 'WPOD+WPOD']:
                vertices = np.array(CCPD_vertices_info(img_path))
            elif c.model_code == 'Hourglass+Vernex_lpfr':
                print 'function not implemented, pass'
                pass
        elif not CCPD_origin:
            if c.model_code in ['Hourglass+Vernex_lp', 'Hourglass+WPOD', 'WPOD+WPOD']:
                vertices = np.array(CCPD_FR_vertices_info(img_path))
            elif c.model_code == 'Hourglass+Vernex_lpfr':
                vertices = np.array(CCPD_FR_vertices_info(img_path) + CCPD_FR_front_rear_info(img_path))

        key_pts.append(vertices)

    seq = iaa.Sequential([
                        sometimes(iaa.Affine(scale={"x": (0.5, 1), "y": (0.5, 1)}, shear=(-60, 60), rotate=(-25, 25))),
                        sometimes(iaa.PerspectiveTransform(scale=(0.05, 0.1), keep_size=False)),
                        iaa.AddToHueAndSaturation(value=(-50, 50)),
                        iaa.Fliplr(0.5),
                        iaa.Affine(scale=(0.2, 1))
                                                   ], random_order=True)
    '''
    seq = iaa.Sequential([iaa.Fliplr(0.5)])
    '''

    imgs_aug, vertices_aug = seq(images=imgs, keypoints=key_pts)

    for img_aug in imgs_aug:
        cv2.cvtColor(img_aug, cv2.COLOR_RGB2BGR)

    return imgs_aug, vertices_aug


if __name__ == '__main__':
    img_paths = read_img_from_dir('/home/shaoheng/Documents/Thesis_KSH/training_data/openALPR_us')[:32]
    while 1:
        images_aug, keypoints_aug = data_aug(img_paths)
        for image_aug, keypoint_aug in zip(images_aug, keypoints_aug):
            image_aug = draw_LP_by_vertices(image_aug, keypoint_aug[0:4])
            image_aug = draw_LP_by_vertices(image_aug, keypoint_aug[4:8])
            cv2.imwrite('0.jpg', image_aug)
            cv2.imshow('img', image_aug)
            cv2.waitKey(0)


