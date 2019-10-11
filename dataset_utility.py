from os.path import basename, splitext

import cv2
import json

from img_utility import BBCor_to_pts, vertices_rearange


# return a list of BB coordinates [[x1, y1], [x2, y2]]
def CCPD_BBCor_info(img_path):
    img_path = basename(img_path)
    BBCor = img_path.split('-')[2].split('_')
    return [map(int, BBCor[0].split('&')), map(int, BBCor[1].split('&'))]


# return a list of vertices coordinates [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
def CCPD_vertices_info(img_path):
    img_path = basename(img_path)
    vertices = img_path.split('-')[3].split('_')
    return [map(int, vertices[0].split('&')), map(int, vertices[1].split('&')),
            map(int, vertices[2].split('&')), map(int, vertices[3].split('&'))]


# used for the CCPD_FR training data, read the LP vertices [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
def CCPD_FR_vertices_info(img_path):
    img_path = basename(img_path)
    vertices = img_path.split('.')[0].split('_')
    return [map(int, vertices[0].split('&')), map(int, vertices[1].split('&')),
            map(int, vertices[2].split('&')), map(int, vertices[3].split('&'))]


# return the vertices for front and rear for CCPD_FR format [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
def CCPD_FR_front_rear_info(img_path):
    shape = cv2.imread(img_path).shape
    w = shape[1]
    h = shape[0]
    return [[w, h], [0, h], [0, 0], [w, 0]]


# return [[tl], [br]], tl, br in format [x, y]
def openALPR_BBCor_info(img_path):
    notation_file = splitext(img_path)[0] + '.txt'
    shape = cv2.imread(img_path).shape
    with open(notation_file, 'r') as f:
        context = f.readline().split()
        BBCor = context[1:5]
        BBCor = map(int, BBCor)
        x_min = max(BBCor[0], 0)
        x_max = min(BBCor[0] + BBCor[2], shape[1])
        y_min = max(BBCor[1], 0)
        y_max = min(BBCor[1] + BBCor[3], shape[0])
        return [[x_min, y_min], [x_max, y_max]]


# [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
def vernex_front_rear_info(img_path):
    img_name = basename(img_path)
    vertices = img_name.split('.')[0].split('_')
    return [map(int, vertices[4].split('&')), map(int, vertices[5].split('&')),
            map(int, vertices[6].split('&')), map(int, vertices[7].split('&'))]


# [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
def vernex_vertices_info(img_path):
    img_name = basename(img_path)
    vertices = img_name.split('.')[0].split('_')
    return [map(int, vertices[0].split('&')), map(int, vertices[1].split('&')),
            map(int, vertices[2].split('&')), map(int, vertices[3].split('&'))]


# return string, 'front' ro 'rear'
def vernex_fr_class_info(img_path):
    img_name = basename(img_path)
    ele = img_name.split('.')[0].split('_')
    return ele[8]


# read the json file including lp and fr annotations
# it will return the lp and fr coordinate start at br and clock-wise
# return -> w, h, class, {vertices indexed by 'fr' or 'front' or 'rear'}
def json_lp_fr(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    weight = data['imageWidth']
    height = data['imageHeight']
    lp_fr_vertices = {}
    for annotation in data['shapes']:
        if annotation['shape_type'] == 'rectangle':
            pts = BBCor_to_pts(*[map(int, pt) for pt in annotation['points']])
        elif annotation['shape_type'] == 'polygon':
            pts = vertices_rearange([map(int, pt) for pt in annotation['points']])
        lp_fr_vertices[annotation['label']] = pts
        if annotation['label'] in ['front', 'rear']:
            cls = annotation['label']

    assert len(lp_fr_vertices) == 2, 'data set length not matched, please check the data'
    assert 'lp' in lp_fr_vertices and ('front' in lp_fr_vertices or 'rear' in lp_fr_vertices),\
           'Now this function is only supported for one-lp and one-fr'

    return weight, height, cls, lp_fr_vertices


if __name__ == '__main__':
    path = '/home/shaoheng/Documents/Thesis_KSH/training_data/cd_hard/05344.json'
    w, h, cls, lp_fr = json_lp_fr(path)
    print w, h, cls, lp_fr