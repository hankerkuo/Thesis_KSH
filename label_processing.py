"""
this script is used to process the CCPD_FR labels and make them fit the required training format
"""
import cv2
import numpy as np
from img_utility import pts_to_BBCor, read_img_from_dir, pixel_to_ratio, IoU
from CCPD_utility import FR_vertices_info, vertices_info
from random import sample, shuffle
from collections import deque
from threading import Thread, Lock
from time import sleep


"""
label pre-processing
"""


# return the mean value of LP size in a dataset of CCPD_FR format images
# need the fixed training input dimension (square for input images)
# can pass total_stride argument (total stride of model)
def mean_size_LP(img_folder, training_dim, total_stride=1):
    imgs_path = read_img_from_dir(img_folder)
    imgs_amount = len(imgs_path)
    W, H = 0., 0.
    for img_path in imgs_path:
        img_size = cv2.imread(img_path).shape  # cv2.imread.shape -> (h, w, ch)
        vertices = FR_vertices_info(img_path)
        BBCor = pts_to_BBCor(*vertices)
        BBCor = pixel_to_ratio(img_size, *BBCor)
        w_ratio, h_ratio = BBCor[1][0] - BBCor[0][0], BBCor[1][1] - BBCor[0][1]
        W += w_ratio * training_dim
        H += h_ratio * training_dim
    return (W + H) / 2 / imgs_amount / total_stride


# read the CCPD_FR images and return the label for training
# need to give the dimension for training and the total stride in the model
# label shape -> [y, x, 1 + 2*4], y and x are the downsampled output map size
# label format -> [object_1or0, x1, y1, x2, y2, x3, y3, x4, y4] pts from bottom right and clockwise
def CCDP_FR_to_training_label(img_path, training_dim, stride, CCPD_origin=False):

    side = 3.5  # calculated by training_dim = 208 and stride = 16
    img_shape = cv2.imread(img_path).shape
    out_size = training_dim / stride

    assert training_dim % stride == 0, 'training_dim dividing stride must be a integer'

    if CCPD_origin:
        vertices = vertices_info(img_path)
    else:
        vertices = FR_vertices_info(img_path)

    LP_Cor = np.array(pixel_to_ratio(img_shape, *vertices)) * training_dim
    LP_BB = np.array(pts_to_BBCor(*LP_Cor))

    LP_Cor_outdim = LP_Cor / stride
    LP_BB_outdim = [np.maximum(LP_BB[0] / stride, 0).astype(int), np.minimum(LP_BB[1] / stride, out_size).astype(int)]
    label = np.zeros((out_size, out_size, 1 + 2 * 4))

    for y in range(LP_BB_outdim[0][1], LP_BB_outdim[1][1]):
        for x in range(LP_BB_outdim[0][0], LP_BB_outdim[1][0]):

            now_pixel = np.array([x + 0.5, y + 0.5])

            LP_BB_wh = LP_BB_outdim[1] - LP_BB_outdim[0]
            same_BB_on_now_pixel = [now_pixel - LP_BB_wh / 2., now_pixel + LP_BB_wh / 2.]
            # print LP_BB_outdim
            # print same_BB_on_now_pixel
            iou = IoU(LP_BB_outdim, same_BB_on_now_pixel)

            if iou > 0.5:
                LP_Cor_recenter = (np.array(LP_Cor_outdim) - now_pixel) / side
                label[y, x, 0] = 1
                label[y, x, 1:] = LP_Cor_recenter.flatten()

    return label


# combine the labels of four images and make it a huge training label
# be sure the label having same training dim and model stride
def label_splicing(label1, label2, label3, label4):
    w = label1.shape[1]
    h = label1.shape[0]
    label2 += np.array([0, w, 0, w, 0, w, 0, w, 0])
    label3 += np.array([0, 0, h, 0, h, 0, h, 0, h])
    label4 += np.array([0, w, h, w, h, w, h, w, h])
    top_row = np.concatenate([label1, label2], axis=1)
    bottom_row = np.concatenate([label3, label4], axis=1)
    final_label = np.concatenate([top_row, bottom_row], axis=0)

    return final_label


# also splicing function but for images
def img_splicing(img1, img2, img3, img4):
    top_row = np.concatenate([img1, img2], axis=1)
    bottom_row = np.concatenate([img3, img4], axis=1)
    final_img = np.concatenate([top_row, bottom_row], axis=0)

    return final_img


# return the needed data for training
# including: x_data -> the image data with numpy array format
#            y_data -> the corresponding label of the image
def batch_data_generator(img_folder, batch_size, training_dim, stride):
    assert training_dim % stride == 0, 'training_dim dividing stride must be a integer'
    output_dim = training_dim / stride

    imgs_path = read_img_from_dir(img_folder)
    samples = sample(imgs_path, batch_size)
    x_data = []  # for some reason np.empty met some problems when giving a array from cv2.imread, so use list
    y_data = np.empty(shape=(batch_size, output_dim, output_dim, 1 + 2 * 4))
    for i, img_path in enumerate(samples):
        x_data.append(cv2.resize(cv2.imread(img_path), (training_dim, training_dim)))
        y_data[i] = CCDP_FR_to_training_label(img_path, training_dim, stride)
    return np.array(x_data), y_data


# the class provides batch training data
# need to give arguments -> img_folder, batch_size, training_dim, stride
# it can be served as 1. a infinite iterator which keeps providing data, each image in the folder will be provided
#                        before next epoch, the image will be randomly selected (the 'shuffle' argument) in every epoch
#                     2. a daemon threading data provider, which is preferred and much faster than the iterator
# * if set splice_train = true, then the output training data's dim will be twice the given value
class DataProvider:

    def __init__(self, img_folder, batch_size, training_dim, stride, CCPD_origin=False, shuffle=True, splice_train=False):
        self.imgs_paths = read_img_from_dir(img_folder)
        self.batch_size = batch_size
        self.training_dim = training_dim
        self.stride = stride
        self.out_dim = training_dim / stride
        self.samples = deque(self.imgs_paths)
        self.CCPD_origin = CCPD_origin
        self.shuffle = shuffle
        self.splic_train = splice_train
        self.x_data, self.y_data = self.create_buffer(batch_size)
        self.buffer_loaded = False
        self._lock = Lock()
        self.thread = Thread()
        self.stop_buffer = False

    def __iter__(self):
        if shuffle:
            shuffle(self.samples)
        return self

    def next(self):
        x_data = []
        y_data = []

        for b in range(self.batch_size):
            if len(self.samples) == 0:
                self.samples = deque(self.imgs_paths)
                if shuffle:
                    shuffle(self.samples)
                break
            img_path = self.samples.pop()
            x_data.append(cv2.resize(cv2.imread(img_path), (self.training_dim, self.training_dim)))
            y_data.append(CCDP_FR_to_training_label(img_path, self.training_dim,
                                                    self.stride, CCPD_origin=self.CCPD_origin))

        if len(x_data) == 0:
            return self.next()
        else:
            return np.array(x_data), np.array(y_data)

    def create_buffer(self, batch_size):
        x = np.empty((batch_size, self.training_dim, self.training_dim, 3))
        y = np.empty((batch_size, self.out_dim, self.out_dim, 1 + 2 * 4))
        return x, y

    def get_batch(self):
        while self.buffer_loaded is False:
            sleep(0.01)
        with self._lock:
            self.buffer_loaded = False
            return self.x_data, self.y_data

    def load(self):
        while True:
            while self.buffer_loaded is True:
                sleep(0.01)
                if self.stop_buffer:
                    return 0
            with self._lock:
                x_data = []
                y_data = []

                for b in range(self.batch_size):
                    # print 'now %d samples left' % len(self.samples)
                    if len(self.samples) == 0:
                        self.samples = deque(self.imgs_paths)
                        if shuffle:
                            shuffle(self.samples)
                        break
                    if self.splic_train:

                        if len(self.samples) < self.batch_size * 4:
                            training_dim = self.training_dim * 2
                            img_path = self.samples.pop()
                            x_data.append(
                                cv2.resize(cv2.imread(img_path), (training_dim, training_dim)) / 255.)
                            y_data.append(CCDP_FR_to_training_label(img_path, training_dim,
                                                                    self.stride, CCPD_origin=self.CCPD_origin))
                        else:
                            img_paths = [self.samples.pop() for _ in range(4)]
                            imgs = [cv2.resize(cv2.imread(img_path), (self.training_dim, self.training_dim)) / 255.
                                    for img_path in img_paths]
                            img_huge = img_splicing(*imgs)
                            labels = [CCDP_FR_to_training_label(img_path, self.training_dim, self.stride,
                                      CCPD_origin=self.CCPD_origin) for img_path in img_paths]
                            label_huge = label_splicing(*labels)
                            x_data.append(img_huge)
                            y_data.append(label_huge)

                    elif not self.splic_train:
                        img_path = self.samples.pop()
                        x_data.append(cv2.resize(cv2.imread(img_path), (self.training_dim, self.training_dim)) / 255.)
                        y_data.append(CCDP_FR_to_training_label(img_path, self.training_dim,
                                                                self.stride, CCPD_origin=self.CCPD_origin))

                if len(x_data) == 0:
                    return self.load()
                else:
                    self.x_data = np.array(x_data)
                    self.y_data = np.array(y_data)

                    self.buffer_loaded = True
                    print 'loading batch image to buffer... done!'

    def start_loading(self):
        self.thread = Thread(target=self.load)
        self.thread.setDaemon(True)
        self.thread.start()

    def stop_loading(self):
        self.stop_buffer = True
        self.thread.join()

    def load_whole_folder(self):  # risky
        images = []
        for i, sample in enumerate(self.samples):
            images.append(cv2.imread(sample))
            print 1. * i / len(self.samples)
        return images


"""
label post-processing
"""


# receive the output of the network and map the label to the original image
# now this function only work with single image prediction label
# in each label -> [prob, cor_after_affine]
def predicted_label_to_origin_image(ori_image_shape, label, stride, prob_threshold=0.9, use_nms=True):
    side = 3.5

    out_w = label.shape[1]
    out_h = label.shape[0]

    label_to_origin = []
    for y in range(out_h):
        for x in range(out_w):
            prob = label[y, x, 0]      # integer

            if prob >= prob_threshold:
                now_pixel = np.array([x + 0.5, y + 0.5])

                affinex = label[y, x, 2:5]  # shape = [3, ]
                affiney = label[y, x, 5:]   # shape = [3, ]
                affinex[0] = max(affinex[0], 0)
                affiney[1] = max(affiney[1], 0)

                # base rectangle from br and clock-wise
                base_rectangle = np.array([[0.5, 0.5, 1], [-0.5, 0.5, 1], [-0.5, -0.5, 1], [0.5, -0.5, 1]])

                # cor_after_affine -> [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                cor_after_affine = np.stack([np.sum(affinex * base_rectangle, axis=1),
                                             np.sum(affiney * base_rectangle, axis=1)], axis=1)  # shape = [4, 2]
                cor_after_affine = cor_after_affine * side
                cor_after_affine += now_pixel
                cor_after_affine *= stride
                cor_after_affine *= np.array([ori_image_shape[1] / (1. * out_w * stride),
                                              ori_image_shape[0] / (1. * out_h * stride)])
                cor_after_affine = cor_after_affine.astype(int)

                # clip according to the size of original size of image
                for pts in cor_after_affine:
                    pts[0] = np.clip(pts[0], 0, ori_image_shape[1])
                    pts[1] = np.clip(pts[1], 0, ori_image_shape[0])

                label_to_origin.append([prob, cor_after_affine])
    if use_nms:
        label_to_origin = nms(label_to_origin)

    return label_to_origin


# nms function, labels -> a list of labels, its element is [probability, coordinates]
def nms(labels, threshold=0.5):
    labels.sort(key=lambda x: x[0], reverse=True)
    labels = deque(labels)
    labels_nms = []
    while len(labels) > 0:
        now_handle = labels.popleft()
        while len(labels) > 0 and IoU(pts_to_BBCor(*now_handle[1]), pts_to_BBCor(*labels[0][1])) > threshold:
            labels.popleft()
        labels_nms.append(now_handle)
    return labels_nms


if __name__ == '__main__':
    path = '/home/shaoheng/Documents/Thesis_KSH/training_data/CCPD_FR_total746'
    data_provider = DataProvider(path, 64, 208, 16, CCPD_origin=False, shuffle=True, splice_train=True)
    data_provider.start_loading()
    while 1:
        x, y = data_provider.get_batch()
        print x.shape


