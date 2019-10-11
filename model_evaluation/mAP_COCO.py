from geometry_calc import polygons_iou
from os.path import splitext

import json
import traceback
import matplotlib.pyplot as plt

from dataset_utility import vernex_vertices_info, vernex_front_rear_info, vernex_fr_class_info
from img_utility import read_img_from_dir, pts_to_BBCor, IoU


def coco_mAP_vernex(output_result_folder, iou_threshold, classify_cal=True):
    data_val_folder = output_result_folder
    imgs_paths_val = read_img_from_dir(data_val_folder)
    total_gt = len(imgs_paths_val)
    class_dict = {'front': 1, 'rear': 2}
    lp_results = []  # -> format [prob, True or False], True for true positive, False for false positive
    fr_results = []  # -> format [True or False, IoU], True for true class, False for false class, IoU for quadrilateral IoU
    res_pres = []

    # read all the prediction result and sort them
    for img_path in imgs_paths_val:
        vertices_lp_gt = vernex_vertices_info(img_path)
        vertices_fr_gt = vernex_front_rear_info(img_path)
        fr_class = vernex_fr_class_info(img_path)

        f = open(splitext(img_path)[0] + '_result.json', 'r')
        data = json.load(f)

        '''for vernex'''
        for lp in data['lps']:

            true_or_false = False  # true positive or false positive
            try:
                # if polygons_iou(lp['vertices_lp'], vertices_lp_gt) >= iou_threshold:
                if IoU(pts_to_BBCor(*lp['vertices_lp']), pts_to_BBCor(*vertices_lp_gt)) >= iou_threshold:
                    true_or_false = True
            except:
                # traceback.print_exc()
                if IoU(pts_to_BBCor(*lp['vertices_lp']), pts_to_BBCor(*vertices_lp_gt)) >= iou_threshold:
                    true_or_false = True

            if true_or_false and classify_cal:
                classify = False
                if class_dict[fr_class] == lp['fr_class']:
                    classify = True
                try:
                    fr_results.append([classify, polygons_iou(lp['vertices_fr'], vertices_fr_gt)])
                except:
                    # traceback.print_exc()
                    fr_results.append([classify, 0])

            lp_results.append([lp['lp_prob'], true_or_false])

        '''for openALPR
        for lp in data['results']:

            true_or_false = False  # true positive or false positive
            lp_vertices = [[s['x'], s['y']] for s in lp['coordinates']]
            try:
                # if polygons_iou(lp_vertices, vertices_lp_gt) >= iou_threshold:
                if IoU(pts_to_BBCor(*lp_vertices), pts_to_BBCor(*vertices_lp_gt)) >= iou_threshold:
                    true_or_false = True
            except:
                # traceback.print_exc()
                if IoU(pts_to_BBCor(*lp_vertices), pts_to_BBCor(*vertices_lp_gt)) >= iou_threshold:
                    true_or_false = True

            lp_results.append([lp['confidence'], true_or_false])'''


        '''for Sighthound
        for lp in data['objects']:

            annotation = lp['licenseplateAnnotation']

            true_or_false = False  # true positive or false positive
            lp_vertices = [[s['x'], s['y']] for s in annotation['bounding']['vertices']]
            try:
                if polygons_iou(lp_vertices, vertices_lp_gt) >= iou_threshold:
                # if IoU(pts_to_BBCor(*lp_vertices), pts_to_BBCor(*vertices_lp_gt)) >= iou_threshold:
                    true_or_false = True
            except:
                # traceback.print_exc()
                if IoU(pts_to_BBCor(*lp_vertices), pts_to_BBCor(*vertices_lp_gt)) >= iou_threshold:
                    true_or_false = True

            lp_results.append([annotation['attributes']['system']['string']['confidence'], true_or_false])'''

        f.close()

    lp_results.sort(key=lambda s: s[0], reverse=True)

    # calculate the recall and precision and save into the list recalls_precisions
    true_positive = 0.
    detections = 0.
    for lp_result in lp_results:
        detections += 1
        if lp_result[1]:
            true_positive += 1
        recall = true_positive / total_gt
        precision = true_positive / detections
        res_pres.append([recall, precision])

    # assign the bigger precision value to eliminate the zig-zag
    for i, recall_precision in enumerate(res_pres):
        recall_precision[1] = max([x[1] for x in res_pres[i:]])

    # remove the duplications and make a finer list for recall_precision
    pres_dupfree = []
    res_met = []
    for recall_precision in res_pres:
        if recall_precision[0] not in res_met:
            pres_dupfree.append(recall_precision)
            res_met.append(recall_precision[0])
    res_pres_dupfree = []
    pres_met = []
    for recall_precision in pres_dupfree[-1::-1]:
        if recall_precision[1] not in pres_met:
            res_pres_dupfree.append(recall_precision)
            pres_met.append(recall_precision[1])

    res_pres_dupfree = res_pres_dupfree[-1::-1]

    # interpolation of 101 points as the evaluation method used in COCO
    i = 0
    precision = 0
    for points in range(101):
        recall = points / 100.

        while res_pres_dupfree[i][0] < recall:
            i += 1
            if i == len(res_pres_dupfree):
                break
        if i == len(res_pres_dupfree):
            break
        precision += res_pres_dupfree[i][1]
    precision = precision / 101.

    # to calculate the classification precision
    class_precision, average_iou = 0., 0.
    if classify_cal and fr_results:
        true_classification = 0.
        total_iou = 0.
        for fr_result in fr_results:
            if fr_result[0]:
                true_classification += 1
                total_iou += fr_result[1]
        class_precision = true_classification / len(fr_results)
        try:
            average_iou = total_iou / true_classification
        except ZeroDivisionError:
            pass

    '''
    plt.plot([x[0] for x in res_pres_dupfree], [y[1] for y in res_pres_dupfree])
    plt.plot([x[0] for x in res_pres], [y[1] for y in res_pres])
    plt.show()
    '''

    return precision, class_precision, average_iou


if __name__ == '__main__':
    output_result_folder = '/home/shaoheng/Documents/ALPR_commercials/outputs/Sighthound_cd_hard_vernex'

    # coco mAP
    mAP = 0.
    for threshold in range(50, 100, 5):
        mAP += coco_mAP_vernex(output_result_folder, threshold / 100., classify_cal=False)[0]
    mAP = mAP / 10

    # coco mAP50
    coco_map_50, class_accuracy, iou_front_rear = coco_mAP_vernex(output_result_folder, 0.5, classify_cal=True)
    # coco mAP75
    coco_map_75 = coco_mAP_vernex(output_result_folder, 0.75, classify_cal=False)[0]

    print 'COCO mAP:', '%.1f' % (mAP * 100)
    print 'COCO mAP50:', '%.1f' % (coco_map_50 * 100)
    print 'COCO mAP75:', '%.1f' % (coco_map_75 * 100)
    print 'classification accuracy:', '%.1f' % (class_accuracy * 100)
    print 'average iou for front-rear', '%.1f' % (iou_front_rear * 100)




