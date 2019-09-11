import numpy as np
import imgaug.augmenters as iaa
from img_utility import read_img_from_dir
from CCPD_utility import FR_vertices_info, vertices_info
from drawing_utility import draw_LP_by_vertices
import cv2


def data_aug(img_paths, CCPD_origin=False):
    imgs = []
    key_pts = []

    for img_path in img_paths:
        img = cv2.imread(img_path)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)

        if CCPD_origin:
            vertices = np.array(vertices_info(img_path))
        elif not CCPD_origin:
            vertices = np.array(FR_vertices_info(img_path))
        key_pts.append(vertices)

    seq = iaa.Sequential([iaa.Affine(scale={"x": 1, "y": (0.5, 1)}, shear=(-45, 45), rotate=(-25, 25)),
                          iaa.AddToHueAndSaturation(value=(-100, 100)),
                          iaa.Fliplr(0.5)])

    imgs_aug, vertices_aug = seq(images=imgs, keypoints=key_pts)

    for img_aug in imgs_aug:
        cv2.cvtColor(img_aug, cv2.COLOR_RGB2BGR)

    return imgs_aug, vertices_aug


if __name__ == '__main__':
    img_paths = read_img_from_dir('/home/shaoheng/Documents/Thesis_KSH/training_data/CCPD_FR_total2333')[:32]
    images_aug, keypoints_aug = data_aug(img_paths)
    for image_aug, keypoint_aug in zip(images_aug, keypoints_aug):
        image_aug = draw_LP_by_vertices(image_aug, keypoint_aug)
        cv2.imshow('img', image_aug)
        cv2.waitKey(0)


