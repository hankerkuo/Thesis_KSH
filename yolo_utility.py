# for one-line yolo data, return the x, y, w, h into a list with float data type [x, y, w, h]
def yolo_readline(txt_file):
    with open(txt_file) as file:
        line = file.readline()[:-1]  # -1 for not reading the '\n'
        ele = line.split(' ')
        return map(float, ele[1:])


# transfer yolo format to the BB coordinates with actual pixel value
# img_shape -> (y, x), usually use np.shape() to get the shape from cv2.imread object
# x, y, w, h is the yolo format label
# return value: BBCor -> [[x1, y1], [x2, y2]]
def yolo_to_BBCor(img_shape, x, y, w, h):
    BBCor_tl = [int(max(img_shape[1] * (x - w/2), 0)), int(max(img_shape[0] * (y - h/2), 0))]
    BBCor_br = [int(min(img_shape[1] * (x + w/2), img_shape[1])), int(min(img_shape[0] * (y + h/2), img_shape[0]))]
    return [BBCor_tl, BBCor_br]


# return the class label for a single line yolo label
def yolo_class(txt_file):
    with open(txt_file) as file:
        line = file.readline()[:-1]  # -1 for not reading the '\n'
        ele = line.split(' ')
    return int(ele[0])
