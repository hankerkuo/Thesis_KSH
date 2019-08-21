import cv2
from os.path import join
from os import listdir


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


# return all the images path from a directory
def read_img_from_dir(path):
    imgs = []
    for img in listdir(path):
        if img.split('.')[1] in ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']:
            imgs.append(join(path, img))
    return imgs


if __name__ == '__main__':
    img = cv2.imread('01-0_1-249&528_393&586-392&584_249&586_250&530_393&528-0_0_25_27_7_26_29-131-21.jpg')