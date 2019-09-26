import numpy as np
import cv2
from shutil import copy
from img_utility import read_img_from_dir
from dataset_utility import CCPD_vertices_info
from shapely.geometry import Polygon, Point

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
        vertices = CCPD_vertices_info(img_path)
        hor_angle, ver_angle = rotation_degrees(*vertices)

        if abs(hor_angle) > hor_degree_threshold or abs(ver_angle - 90) > ver_degree_threshold:
            copy(img_path, output_folder)


# perspective warp: from original predicted license plate vertices to a planar rectangle
# width and height need to be defined
def planar_rectification(img, vertices, width=400, height=100):
    vertices = np.array([tuple(vertice) for vertice in vertices]).astype(np.float32)  # vertices is from br and clockwise
    dst = np.array([(width, height), (0, height), (0, 0), (width, 0)]).astype(np.float32)
    maxtrix = cv2.getPerspectiveTransform(vertices, dst)
    lp_after_transform = cv2.warpPerspective(img, M=maxtrix, dsize=(width, height))

    return lp_after_transform


# the polygon iou of between a polygon and the same polygon on a specific point
def polygon_iou(polygon_vertices, pt):
    polygon_vertices = np.array(polygon_vertices)
    poly_1 = Polygon([tuple(vertice) for vertice in polygon_vertices])
    centroid = np.array(poly_1.centroid)

    poly_on_pt = polygon_vertices + np.array(pt) - centroid
    poly_2 = Polygon([tuple(vertice) for vertice in poly_on_pt])

    return poly_1.intersection(poly_2).area / poly_1.union(poly_2).area


# return True if the point is within the polygon, else return False
def pt_within_polygon(pt, polygon_vertices):
    polygon_vertices = np.array(polygon_vertices)
    polygon = Polygon([tuple(vertice) for vertice in polygon_vertices])
    point = Point(tuple(pt))

    return point.within(polygon)


if __name__ == '__main__':
    pt = (50, 50)
    polygon = Polygon([(95, 100), (10, 50), (5, 10), (80, 60)])
    print pt.within(polygon)
