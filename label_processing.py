# this script is used to process the CCPD_FR labels and make them fit the required training format

import cv2
from img_utility import pts_to_BBCor, read_img_from_dir
from CCPD_utility import FR_vertices_info
from os.path import basename


# return the mean value of LP size in a dataset of CCPD_FR format images
# need the fixed training input dimension (square for input images)
# can pass total_stride argument (total stride of model)
def mean_size_LP(path_to_images, training_dim, total_stride=1):
    imgs_path = read_img_from_dir(path_to_images)
    imgs_amount = len(imgs_path)
    W, H = 0., 0.
    for img_path in imgs_path:
        img_size = cv2.imread(img_path).shape  # cv2.imread.shape -> (h, w, ch)
        vertices = FR_vertices_info(img_path)
        BBCor = pts_to_BBCor(*vertices)
        width, height = BBCor[1][0] - BBCor[0][0], BBCor[1][1] - BBCor[0][1]
        W += width * training_dim / img_size[1]
        H += height * training_dim / img_size[0]
    return (W + H) / 2 / imgs_amount / total_stride



if __name__ == '__main__':
    path = '/home/shaoheng/Documents/Thesis_KSH/training_data/CCPD_FR_total746'
    size = mean_size_LP(path, 208, 16)
    print size
