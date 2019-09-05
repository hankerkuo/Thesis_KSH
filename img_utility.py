from os.path import join
from os import listdir
import numpy as np


# return the BBCor according to the vertices of a LP, pt1~pt4 -> [x, y], return [tl, br] with tl, br -> [x, y]
def pts_to_BBCor(pt1, pt2, pt3, pt4):
    x_min = min(pt1[0], pt2[0], pt3[0], pt4[0])
    x_max = max(pt1[0], pt2[0], pt3[0], pt4[0])
    y_min = min(pt1[1], pt2[1], pt3[1], pt4[1])
    y_max = max(pt1[1], pt2[1], pt3[1], pt4[1])
    return [[x_min, y_min], [x_max, y_max]]


# cut the img into a rectangle when having four vertices
# img -> numpy array, pt1~pt4 -> tuple or list with format (x, y)
def cut_by_fourpts(img, pt1, pt2, pt3, pt4):
    x_min = min(pt1[0], pt2[0], pt3[0], pt4[0])
    x_max = max(pt1[0], pt2[0], pt3[0], pt4[0])
    y_min = min(pt1[1], pt2[1], pt3[1], pt4[1])
    y_max = max(pt1[1], pt2[1], pt3[1], pt4[1])
    img = img[y_min:y_max, x_min:x_max]
    return img


# cut the img into a rectangle when having BB coordinates
# img -> numpy array, pt1~pt2 -> tuple or list with format (x, y)
def cut_by_BBCor(img, pt1, pt2):
    return img[pt1[1]:pt2[1], pt1[0]:pt2[0]]


# modify the coordinate system according to one specific point
# reference_pt -> the specific point
# *pts -> can receive arbitrary amounts of points, with format [pt1, pt2, ...], pt1 format -> [x, y]
def cor_sys_trans(reference_pt, *pts):
    pts_modified = []
    for pt in pts:
        pts_modified.append([pt[0] - reference_pt[0], pt[1] - reference_pt[1]])
    return pts_modified


# transfer the pixel-based points information to ratio-based
# img_shape -> an array, first channel:height, second channel:width, CAUTION : [y-axis, x-axis] for better with cv2
# *pts_pixel -> can receive arbitrary amounts of points, with format [pt1, pt2, ...], pt1 format -> [x, y]
def pixel_to_ratio(img_shape, *pts_pixel):
    h = img_shape[0]
    w = img_shape[1]
    pts_ratio = []
    for pt in pts_pixel:
        pt = map(float, pt)
        pt_ratio = [pt[0] / w, pt[1] / h]
        pts_ratio.append(pt_ratio)
    return pts_ratio


# return all the images path from a directory
def read_img_from_dir(path):
    imgs = []
    for img in listdir(path):
        try:
            if img.split('.')[1] in ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']:
                imgs.append(join(path, img))
        except:
            print 'in read_img_from_dir function, "', img, '" has been omitted'
    return imgs


# return the area of a BB, given tl and br, format: tl, br -> [x, y]
def area_by_tl_br(tl, br):
    tl = np.array(tl)
    br = np.array(br)
    w = (br-tl)[0]
    h = (br-tl)[1]
    return w * h


# return the IoU value, format: BB_1, BB_2 ->[tl, br], tl, br -> [x, y]
def IoU(BB_1, BB_2):
    xs = [BB_1[0][0], BB_1[1][0], BB_2[0][0], BB_2[1][0]]
    ys = [BB_1[0][1], BB_1[1][1], BB_2[0][1], BB_2[1][1]]
    xs.sort()
    ys.sort()

    intersec_tl_br = [[xs[1], ys[1]], [xs[2], ys[2]]]  # intersection will be the middle two values (for both x and y)
    union_tl_br = [[xs[0], ys[0]], [xs[3], ys[3]]]  # union will be the smallest and largest value

    area_intersec = float(area_by_tl_br(*intersec_tl_br))
    area_union = float(area_by_tl_br(*union_tl_br))
    if area_union == 0:
        return 0
    else:
        return area_intersec / area_union


if __name__ == '__main__':
    iou = IoU([[5, 4], [9, 6]], [[5.5, 4.5], [9.5, 6.5]])
    print iou