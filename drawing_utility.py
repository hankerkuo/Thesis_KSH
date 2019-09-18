import cv2


# img -> cv2.imread format, pts -> np.array([[x1, y1], [x2, y2], [x2, y2], [x2, y2]])
def draw_LP_by_vertices(img, pts):
    cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 234), thickness=4)
    return img


# img -> cv2.imread format, BBCor -> [[x1, y1], [x2, y2]]
# cv2.rectangle need tuple format for points
def draw_LP_by_BBCor(img, BBCor):
    cv2.rectangle(img, tuple(BBCor[0]), tuple(BBCor[1]), color=(255, 0, 234), thickness=4)
    return img


if __name__ == '__main__':
    pass
