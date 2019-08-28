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
    img_name = '01-0_1-249&528_393&586-392&584_249&586_250&530_393&528-0_0_25_27_7_26_29-131-21.jpg'
    vertices = FR_vertices_info()