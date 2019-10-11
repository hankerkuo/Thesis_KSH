import cv2
import numpy as np

from img_utility import pts_to_BBCor


# img -> cv2.imread format, pts -> [[x1, y1], [x2, y2], [x2, y2], [x2, y2]]
def draw_LP_by_vertices(img, pts, color=(255, 0, 234)):
    cv2.polylines(img, [np.array(pts)], isClosed=True, color=color, thickness=4)
    return img


# img -> cv2.imread format, BBCor -> [[x1, y1], [x2, y2]]
# cv2.rectangle need tuple format for points
def draw_LP_by_BBCor(img, BBCor):
    cv2.rectangle(img, tuple(BBCor[0]), tuple(BBCor[1]), color=(255, 0, 234), thickness=4)
    return img


# img -> cv2.imread format, pts -> [[x1, y1], [x2, y2], [x2, y2], [x2, y2]]
# fr_class: 0 -> bg, 1 -> front, 2 -> rear
def draw_FR_color_by_class(img, lp_prob, pts, fr_class, class_prob):

    tl, br = pts_to_BBCor(*pts)
    bl = [tl[0], br[1]]

    # back ground class
    if fr_class in [0]:
        color = (0, 0, 0)  # BLACK
        draw_LP_by_vertices(img, pts, color=color)
        cv2.putText(img, 'Unknown:' + '%.3f' % class_prob, tuple(tl),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=color, thickness=3)
    # front class
    elif fr_class in [1]:
        color = (36, 247, 255)  # YELLOW
        draw_LP_by_vertices(img, pts, color=color)
        cv2.putText(img, 'Front:' + '%.3f' % class_prob, tuple(tl),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=color, thickness=3)
    # rear class
    elif fr_class in [2]:
        color = (255, 194, 36)  # LIGHT BLUE
        draw_LP_by_vertices(img, pts, color=color)
        cv2.putText(img, 'Rear:' + '%.3f' % class_prob, tuple(tl),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=color, thickness=3)

    # lp probability
    cv2.putText(img, 'lp_prob:' + '%.3f' % lp_prob, tuple(bl),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 0), thickness=2)

    return img


if __name__ == '__main__':
    pass
