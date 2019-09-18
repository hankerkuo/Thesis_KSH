from os.path import basename, splitext
import cv2

# return a list of BB coordinates [[x1, y1], [x2, y2]]
def CCPD_BBCor_info(img_name):
    img_name = basename(img_name)
    BBCor = img_name.split('-')[2].split('_')
    return [map(int, BBCor[0].split('&')), map(int, BBCor[1].split('&'))]


# return a list of vertices coordinates [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
def CCPD_vertices_info(img_name):
    img_name = basename(img_name)
    vertices = img_name.split('-')[3].split('_')
    return [map(int, vertices[0].split('&')), map(int, vertices[1].split('&')),
            map(int, vertices[2].split('&')), map(int, vertices[3].split('&'))]


# used for the CCPD_FR training data, read the LP vertices [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
def CCPD_FR_vertices_info(img_name):
    img_name = basename(img_name)
    vertices = img_name.split('.')[0].split('_')
    return [map(int, vertices[0].split('&')), map(int, vertices[1].split('&')),
            map(int, vertices[2].split('&')), map(int, vertices[3].split('&'))]


# return the boudning box for front and rear for CCPD_FR format
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


if __name__ == '__main__':
    path = '/home/shaoheng/Documents/Thesis_KSH/training_data/CCPD_FR/122&172_35&162_32&116_119&126.jpg'
    print CCPD_FR_front_rear_info(path)
    pass