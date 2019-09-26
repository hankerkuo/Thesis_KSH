from label_processing import predicted_label_to_origin_image_WPOD, predicted_label_to_origin_image_Vernex_lp
from label_processing import predicted_label_to_origin_image_Vernex_lpfr
from img_utility import read_img_from_dir, vertices_rearange
from os.path import join, basename, isdir, splitext
from os import mkdir
from drawing_utility import draw_LP_by_vertices, draw_FR_color_by_class
from geometry_calc import planar_rectification
from config import Configs
from model_define import model_and_loss
from time import time
import numpy as np
import cv2


# for testing SINGLE image
# return a list of possible license plates, in each label -> [prob, vertex_predicted_lp,
#                              in 'lpfr', additional info -> vertex_predicted_fr, [fr_class, class_prob]]
def single_img_predict(img_path, input_dim=(0, 0), input_norm=True, model_code=''):
    if model_code in ['WPOD+WPOD', 'Hourglass+WPOD']:
        label_to_origin = predicted_label_to_origin_image_WPOD
    elif model_code in ['Hourglass+Vernex_lp']:
        label_to_origin = predicted_label_to_origin_image_Vernex_lp
    elif model_code in ['Hourglass+Vernex_lpfr']:
        label_to_origin = predicted_label_to_origin_image_Vernex_lpfr

    img = cv2.imread(img_path)
    img_shape = img.shape
    if input_norm:
        div = 255.
    else:
        div = 1.
    img_feed = cv2.resize(img, input_dim) / div
    img_feed = np.expand_dims(img_feed, 0)

    start_pred = time()
    output_labels = model.predict(img_feed)
    time_spent = time() - start_pred

    final_labels = label_to_origin(img_shape, output_labels[0], stride=c.stride,
                                   prob_threshold=c.prob_threshold, use_nms=True, side=c.side)

    return final_labels, time_spent


if __name__ == '__main__':

    c = Configs()

    model = model_and_loss()[0]

    model.load_weights(c.weight)

    if not isdir(c.output_dir):
        mkdir(c.output_dir)
    imgs_paths = read_img_from_dir(c.input_dir)
    time_spent = 0
    for img_path in imgs_paths:

        print 'processing', img_path
        final_labels, sec = single_img_predict(img_path=img_path, input_dim=c.test_input_dim,
                                               input_norm=c.input_norm, model_code=c.model_code)

        time_spent += sec
        if len(final_labels) == 0:
            print 'fail to detect'
        else:
            print '%d LPs found' % len(final_labels)

        img = cv2.imread(img_path)

        for i, final_label in enumerate(final_labels[:c.LPs_to_find]):

            prob, vertices_lp = final_label[:2]
            vertices_lp = vertices_rearange(vertices_lp)

            # save each license plate
            '''
            lp_img = planar_rectification(img, vertices_lp)
            cv2.imwrite(join(c.output_dir, splitext(basename(img_path))[0] + '_%d' % i + '.jpg'), lp_img)
            '''

            # draw visualization results
            img = draw_LP_by_vertices(img, vertices_lp)
            # if it's lpfr model, then draw front and rear
            if c.model_code in ['Hourglass+Vernex_lpfr']:
                vertices_fr = final_label[2]
                fr_class, class_prob = final_label[3]
                img = draw_FR_color_by_class(img, prob, vertices_fr, fr_class, class_prob)

        cv2.imwrite(join(c.output_dir, basename(img_path)), img)
        print 'write to:', join(c.output_dir, basename(img_path))

    print 'processing, %d images, spend:%.3f seconds' % (len(imgs_paths), time_spent)



