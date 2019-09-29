"""
the class provides batch training data
need to give arguments -> img_folder, batch_size, training_dim, stride
it can be served as 1. a infinite iterator which keeps providing data, each image in the folder will be provided
                       before next epoch, the image will be randomly selected (the 'shuffle' argument) in every epoch
                    2. a daemon threading data provider, which is preferred and much faster than the iterator
* if set splice_train = true, then the output training data's dim will be twice the given value
"""
from collections import deque
from random import shuffle, sample
from threading import Lock, Thread
from time import sleep

import cv2
import numpy as np

from img_utility import read_img_from_dir
from label_processing import batch_CCPD_to_training_label, batch_CCPD_to_training_label_vernex_lpfr, \
    CCDP_FR_to_training_label


class DataProvider:

    def __init__(self, img_folder, batch_size, training_dim, stride,
                 shuffling=True, splice_train=False, mixing_train=False, side=3.5, model_code=''):
        self.img_folder = img_folder
        self.batch_size = batch_size
        self.training_dim = training_dim
        self.stride = stride
        self.out_dim = training_dim / stride
        self.shuffle = shuffling
        self.splice_train = splice_train
        self.mixing_train = mixing_train
        self.side = side
        self.x_data, self.y_data = self.create_buffer(batch_size)
        self.buffer_loaded = False
        self._lock = Lock()
        self.thread = Thread()
        self.stop_buffer = False
        self.imgs_paths = []
        self.samples = deque(self.imgs_paths)
        '''MODEL, in ['Hourglass+Vernex_lpfr', 'Hourglass+Vernex_lp', 'Hourglass+WPOD', 'WPOD+WPOD']'''
        if model_code in ['Hourglass+Vernex_lp', 'Hourglass+WPOD', 'WPOD+WPOD']:
            self.to_training_label = batch_CCPD_to_training_label
        elif model_code == 'Hourglass+Vernex_lpfr':
            self.to_training_label = batch_CCPD_to_training_label_vernex_lpfr

    def __iter__(self):
        if self.shuffle:
            shuffle(self.samples)
        return self

    def next(self):
        x_data = []
        y_data = []

        for b in range(self.batch_size):
            if len(self.samples) == 0:
                self.samples = deque(self.imgs_paths)
                if self.shuffle:
                    shuffle(self.samples)
                break
            img_path = self.samples.pop()
            x_data.append(cv2.resize(cv2.imread(img_path), (self.training_dim, self.training_dim)))
            y_data.append(CCDP_FR_to_training_label(img_path, self.training_dim,
                                                    self.stride, side=self.side))

        if len(x_data) == 0:
            return self.next()
        else:
            return np.array(x_data), np.array(y_data)

    def create_buffer(self, batch_size):
        x = np.empty((batch_size, self.training_dim, self.training_dim, 3))
        y = np.empty((batch_size, self.out_dim, self.out_dim, 1 + 2 * 4))
        return x, y

    def renew_img_paths(self):
        if self.mixing_train:
            '''for randomization
            CCPD_FR = read_img_from_dir('/home/shaoheng/Documents/Thesis_KSH/training_data/CCPD_FR')
            openALPR_br = read_img_from_dir('/home/shaoheng/Documents/Thesis_KSH/training_data/openALPR_br')
            openALPR_us = read_img_from_dir('/home/shaoheng/Documents/Thesis_KSH/training_data/openALPR_us')
            openALPR_eu = read_img_from_dir('/home/shaoheng/Documents/Thesis_KSH/training_data/openALPR_eu')
            CCPD_FR = sample(CCPD_FR, 200)
            '''
            vernex_origin = read_img_from_dir('/home/shaoheng/Documents/Thesis_KSH/training_data/vernex')
            kr_tilt_vernex = read_img_from_dir('/home/shaoheng/Documents/Thesis_KSH/training_data/kr_tilt_vernex')
            # vernex_origin = sample(vernex_origin, 900)

            self.imgs_paths = vernex_origin + kr_tilt_vernex

        elif not self.mixing_train:
            self.imgs_paths = read_img_from_dir(self.img_folder)

    def get_batch(self):
        while self.buffer_loaded is False:
            pass
        with self._lock:
            self.buffer_loaded = False
            return self.x_data, self.y_data

    def load(self):
        while True:
            while self.buffer_loaded is True:
                pass
                if self.stop_buffer:
                    return 0
            with self._lock:
                img_paths = []
                for b in range(self.batch_size):
                    # print 'now %d samples left' % len(self.samples)
                    if len(self.samples) == 0:
                        self.renew_img_paths()
                        self.samples = deque(self.imgs_paths)
                        if self.shuffle:
                            shuffle(self.samples)
                    img_paths.append(self.samples.pop())

                x_data, y_data = self.to_training_label(img_paths, self.training_dim, self.stride, side=self.side)
                self.x_data = np.array(x_data)
                self.y_data = np.array(y_data)
                self.buffer_loaded = True
                print 'training size: %d provider, load batch image to buffer!' % self.training_dim

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


if __name__ == '__main__':
    from config import Configs
    c = Configs()
    data_generator = DataProvider(c.training_data_folder, c.batch_size, c.training_dim_1, c.stride,
                                  side=c.side, mixing_train=c.mixing_train, model_code=c.model_code)
    data_generator.start_loading()
    while 1:
        x, y = data_generator.get_batch()
        print np.shape(y)
