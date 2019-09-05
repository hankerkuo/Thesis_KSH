import numpy as np
from shutil import copy
from img_utility import read_img_from_dir
from CCPD_utility import vertices_info

# return the horizontal and vertical rotation degree of a LP by giving its four vertices
# vertices format -> [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] from br and clockwise
# the angle is based on the coordination system below:
'''
--------> x
|
|
v
y
'''
def rotation_degrees(pt1, pt2, pt3, pt4):
    pt1, pt2, pt3, pt4 = np.array(pt1), np.array(pt2), np.array(pt3), np.array(pt4)
    hor_vec1, hor_vec2 = pt1 - pt2, pt4 - pt3
    ver_vec1, ver_vec2 = pt2 - pt3, pt1 - pt4
    horizontal_angle = np.average([np.angle(complex(hor_vec1[0], hor_vec1[1]), deg=True),
                                   np.angle(complex(hor_vec2[0], hor_vec2[1]), deg=True)])
    vertical_angle = np.average([np.angle(complex(ver_vec1[0], ver_vec1[1]), deg=True),
                                 np.angle(complex(ver_vec2[0], ver_vec2[1]), deg=True)])

    return horizontal_angle, vertical_angle


# given the degree threshold, the images LP rotation degree beyond these thresholds will be picked and copy
# to the output_folder
def pick_range_of_angle(img_folder, output_folder, hor_degree_threshold, ver_degree_threshold):
    for img_path in read_img_from_dir(img_folder):
        vertices = vertices_info(img_path)
        hor_angle, ver_angle = rotation_degrees(*vertices)

        if abs(hor_angle) > hor_degree_threshold or abs(ver_angle - 90) > ver_degree_threshold:
            copy(img_path, output_folder)


if __name__ == '__main__':
    img_folder = '/home/shaoheng/Documents/cars_label_FRNet/CCPD_2019_first_part/ccpd_base'
    output_folder = '/home/shaoheng/Documents/cars_label_FRNet/CCPD_2019_first_part/rotated_and_tilted'
    pick_range_of_angle(img_folder, output_folder, hor_degree_threshold=20, ver_degree_threshold=10)
