from os.path import basename


# return a list of BB coordinates [[x1, y1], [x2, y2]]
def BBCor_info(img_name):
    img_name = basename(img_name)
    BBCor = img_name.split('-')[2].split('_')
    return [map(int, BBCor[0].split('&')), map(int, BBCor[1].split('&'))]


# return a list of vertices coordinates [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
def vertices_info(img_name):
    img_name = basename(img_name)
    vertices = img_name.split('-')[3].split('_')
    return [map(int, vertices[0].split('&')), map(int, vertices[1].split('&')),
            map(int, vertices[2].split('&')), map(int, vertices[3].split('&'))]


# used for the CCPD_FR training data, read the LP vertices [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
def FR_vertices_info(img_name):
    img_name = basename(img_name)
    vertices = img_name.split('.')[0].split('_')
    return [map(int, vertices[0].split('&')), map(int, vertices[1].split('&')),
            map(int, vertices[2].split('&')), map(int, vertices[3].split('&'))]


if __name__ == '__main__':
    pass