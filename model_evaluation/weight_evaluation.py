import cv2
import json
import numpy as np

from model_evaluation.mAP_COCO import coco_mAP_vernex
from model_define import model_and_loss
from img_utility import read_img_from_dir, vertices_rearange
from drawing_utility import draw_LP_by_vertices, draw_FR_color_by_class
from os import listdir, mkdir, remove
from os.path import splitext, join, isdir, basename, isfile
from config import Configs
from time import time
from label_processing import predicted_label_to_origin_image_WPOD, predicted_label_to_origin_image_Vernex_lp
from label_processing import predicted_label_to_origin_image_Vernex_lpfr


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


c = Configs()
model = model_and_loss()[0]

for weight in listdir(c.weight_folder):
    if splitext(weight)[1] not in ['.h5']:
        print weight, 'skipped'
        continue
    if isfile(join(c.info_saving_folder, splitext(weight)[0] + '.txt')):
        print splitext(weight)[0] + '.txt', 'existed already, skipped'
        continue

    print 'loading weight:', weight
    model.load_weights(join(c.weight_folder, weight))

    if not isdir(c.temp_outout_folder):
        mkdir(c.temp_outout_folder)

    imgs_paths = read_img_from_dir(c.valid_data_folder)
    time_spent = 0

    print 'processing image results ...'
    for img_path in imgs_paths:

        final_labels, sec = single_img_predict(img_path=img_path, input_dim=c.test_input_dim,
                                               input_norm=c.input_norm, model_code=c.model_code)

        time_spent += sec

        img = cv2.imread(img_path)

        infos = {'lps': []}

        for i, final_label in enumerate(final_labels[:c.LPs_to_find]):

            prob, vertices_lp = final_label[:2]
            vertices_lp = vertices_lp.tolist()
            vertices_lp = vertices_rearange(vertices_lp)

            # save each license plate
            '''
            lp_img = planar_rectification(img, vertices_lp)
            cv2.imwrite(join(c.temp_outout_folder, splitext(basename(img_path))[0] + '_%d' % i + '.jpg'), lp_img)
            '''

            # draw visualization results
            img = draw_LP_by_vertices(img, vertices_lp)
            # if it's lpfr model, then draw front and rear
            if c.model_code in ['Hourglass+Vernex_lpfr']:
                vertices_fr = final_label[2].tolist()
                fr_class, class_prob = final_label[3]
                img = draw_FR_color_by_class(img, prob, vertices_fr, fr_class, class_prob)

                # add output results in order to save into json file
                infos['lps'].append({'lp_prob': float(prob), 'vertices_lp': vertices_lp, 'vertices_fr': vertices_fr,
                                     'fr_class': fr_class, 'class_prob': float(class_prob)})

        '''
        save result for mAP calculation'''
        json_path = join(c.temp_outout_folder, splitext(basename(img_path))[0] + '_result.json')
        if isfile(json_path):
            remove(json_path)
        with open(json_path, 'a+') as f:
            json.dump(infos, f)

        cv2.imwrite(join(c.temp_outout_folder, basename(img_path)), img)

    print 'processing, %d images, spend:%.3f seconds' % (len(imgs_paths), time_spent)

    # coco mAP
    mAP = 0.
    for threshold in range(50, 100, 5):
        mAP += coco_mAP_vernex(c.temp_outout_folder, threshold / 100., classify_cal=False)[0]
    mAP = mAP / 10

    # coco mAP50
    coco_map_50, class_accuracy, iou_front_rear = coco_mAP_vernex(c.temp_outout_folder, 0.5, classify_cal=True)
    # coco mAP75
    coco_map_75 = coco_mAP_vernex(c.temp_outout_folder, 0.75, classify_cal=False)[0]

    print '\tCOCO mAP:', '%.1f' % (mAP * 100)
    print '\tCOCO mAP50:', '%.1f' % (coco_map_50 * 100)
    print '\tCOCO mAP75:', '%.1f' % (coco_map_75 * 100)
    print '\tclassification accuracy:', '%.1f' % (class_accuracy * 100)
    print '\taverage iou for front-rear', '%.1f' % (iou_front_rear * 100)

    with open(join(c.info_saving_folder, splitext(weight)[0] + '.txt'), 'w+') as f:
        f.writelines(['COCO mAP:', '%.1f\n' % (mAP * 100),
                      'COCO mAP50:', '%.1f\n' % (coco_map_50 * 100),
                      'COCO mAP75:', '%.1f\n' % (coco_map_75 * 100),
                      'classification accuracy:', '%.1f\n' % (class_accuracy * 100),
                      'average iou for front-rear:', '%.1f\n' % (iou_front_rear * 100)])




