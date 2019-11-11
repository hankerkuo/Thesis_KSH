from os import mkdir, remove
from os.path import join, basename, isdir, splitext, isfile
from time import time

import numpy as np
import cv2
import json

from src.label_processing import predicted_label_to_origin_image_WPOD, predicted_label_to_origin_image_Vernex_lp
from src.label_processing import predicted_label_to_origin_image_Vernex_lpfr, nms
from src.img_utility import read_img_from_dir, vertices_rearange
from src.drawing_utility import draw_LP_by_vertices, draw_FR_color_by_class
from config import Configs
from src.model_define import model_and_loss


# for testing SINGLE image
# return a list of possible license plates, in each label -> [prob, np.array(vertex_predicted_lp),
#                              in 'lpfr', additional info -> np.array(vertex_predicted_fr), [fr_class, class_prob]]
def single_img_predict(img_path, input_norm=True, model_code=''):
    if model_code in ['WPOD+WPOD', 'Hourglass+WPOD']:
        label_to_origin = predicted_label_to_origin_image_WPOD
    elif model_code in ['Hourglass+Vernex_lp']:
        label_to_origin = predicted_label_to_origin_image_Vernex_lp
    elif model_code in ['Hourglass+Vernex_lpfr', 'WPOD+vernex_lpfr']:
        label_to_origin = predicted_label_to_origin_image_Vernex_lpfr

    img = cv2.imread(img_path)
    img_shape = img.shape
    if input_norm:
        div = 255.
    else:
        div = 1.

    time_spent = 0.

    final_labels = []
    for scale in c.multi_scales:
        img_feed = cv2.resize(img, scale) / div
        img_feed = np.expand_dims(img_feed, 0)

        start_pred = time()
        output_labels = model.predict(img_feed)
        time_spent += time() - start_pred

        final_label = label_to_origin(img_shape, output_labels[0], stride=c.stride,
                                      prob_threshold=c.prob_threshold, use_nms=False, side=c.side)
        final_labels.extend(final_label)

    if c.use_nms:
        final_labels = nms(final_labels)

    return final_labels, time_spent


if __name__ == '__main__':

    c = Configs()

    model = model_and_loss(training=False)

    if not isdir(c.output_dir):
        mkdir(c.output_dir)
    imgs_paths = read_img_from_dir(c.input_dir)
    time_spent = 0
    for img_path in imgs_paths:

        print 'processing', img_path
        final_labels, sec = single_img_predict(img_path=img_path, input_norm=c.input_norm, model_code=c.model_code)

        time_spent += sec
        if len(final_labels) == 0:
            print 'fail to detect'
        else:
            print '%d LPs found' % len(final_labels)

        img = cv2.imread(img_path)

        infos = {'lps': []}

        for i, final_label in enumerate(final_labels[:c.LPs_to_find]):

            prob, vertices_lp = final_label[:2]
            vertices_lp = vertices_lp.tolist()
            vertices_lp = vertices_rearange(vertices_lp)

            # save each license plate
            '''
            lp_img = planar_rectification(img, vertices_lp)
            cv2.imwrite(join(c.output_dir, splitext(basename(img_path))[0] + '_%d' % i + '.jpg'), lp_img)
            '''

            # draw visualization results
            img = draw_LP_by_vertices(img, vertices_lp)
            # if it's lpfr model, then draw front and rear
            if c.model_code in ['Hourglass+Vernex_lpfr', 'WPOD+vernex_lpfr']:
                vertices_fr = final_label[2].tolist()
                fr_class, class_prob = final_label[3]  # fr_class : 0->BG, 1->front, 2->rear
                img = draw_FR_color_by_class(img, prob, vertices_fr, fr_class, class_prob)

                # add output results in order to save into json file
                infos['lps'].append({'lp_prob': float(prob), 'vertices_lp': vertices_lp, 'vertices_fr': vertices_fr,
                                     'fr_class': fr_class, 'class_prob': float(class_prob)})

        '''
        save result for mAP calculation'''
        json_path = join(c.output_dir, splitext(basename(img_path))[0] + '_result.json')
        if isfile(json_path):
            remove(json_path)
        with open(json_path, 'a+') as f:
            json.dump(infos, f, indent=2)

        cv2.imwrite(join(c.output_dir, basename(img_path)), img)
        print 'write to:', join(c.output_dir, basename(img_path))

    print 'processing, %d images, spend:%.3f seconds' % (len(imgs_paths), time_spent)



