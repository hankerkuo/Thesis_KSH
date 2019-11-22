<<<<<<< HEAD
from geometry_calc import polygons_iou
from os.path import splitext, join, basename, isfile
=======
from src.geometry_calc import polygons_iou
from os.path import splitext
>>>>>>> d12b71c9aa4d9d625c9931e6c030ac51d51757f0

import json

<<<<<<< HEAD
from dataset_utility import vernex_vertices_info, vernex_front_rear_info, vernex_fr_class_info
from dataset_utility import json_lp_fr_couples
from img_utility import read_img_from_dir, pts_to_BBCor, IoU
=======
from src.dataset_utility import vernex_vertices_info, vernex_front_rear_info, vernex_fr_class_info
from src.img_utility import read_img_from_dir, pts_to_BBCor, IoU
>>>>>>> d12b71c9aa4d9d625c9931e6c030ac51d51757f0


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
                if polygons_iou(lp['vertices_lp'], vertices_lp_gt) >= iou_threshold:
                # if IoU(pts_to_BBCor(*lp['vertices_lp']), pts_to_BBCor(*vertices_lp_gt)) >= iou_threshold:
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

        '''for WPOD
        lps = []
        if not isfile(splitext(img_path)[0] + '_result.txt'):
            continue
        with open(splitext(img_path)[0] + '_result.txt', 'r') as f:
            for line in f.readlines():
                cor, prob = line.split('-')
                cor = eval(cor)
                lps.append({'lp_cor': cor, 'prob': float(prob[:-1])})

        for lp in lps:

            true_or_false = False  # true positive or false positive
            try:
                # if polygons_iou(lp['lp_cor'], vertices_lp_gt) >= iou_threshold:
                if IoU(pts_to_BBCor(*lp['vertices_lp']), pts_to_BBCor(*vertices_lp_gt)) >= iou_threshold:
                    true_or_false = True
            except:
                # traceback.print_exc()
                if IoU(pts_to_BBCor(*lp['lp_cor']), pts_to_BBCor(*vertices_lp_gt)) >= iou_threshold:
                    true_or_false = True

            lp_results.append([lp['prob'], true_or_false])'''

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
            lp_vertices = [[s['x'], s['y']] for s in annotation['bounding']['vertices']]

            true_or_false = False  # true positive or false positive
            
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


def coco_mAP(gt_folder, output_result_folder, iou_threshold, classify_cal=True):
    imgs_paths_gt = read_img_from_dir(gt_folder)
    total_gt = 0.
    class_dict = {'front': 1, 'rear': 2}
    lp_results = []  # -> format [prob, True or False], True for true positive, False for false positive
    fr_results = []  # -> format [True or False, IoU], True for true class, False for false class, IoU for quadrilateral IoU
    res_pres = []

    # read all the prediction result and sort them
    for img_path in imgs_paths_gt:
        w, h, cp_lst = json_lp_fr_couples(splitext(img_path)[0] + '.json')
        total_gt += len(cp_lst)
        viewed = [False for n in range(len(cp_lst))]

        f = open(join(output_result_folder, splitext(basename(img_path))[0] + '_result.json'), 'r')
        data = json.load(f)

        '''for vernex'''
        for lp in data['lps']:
            true_or_false = False  # true positive or false positive

            for i, cp in enumerate(cp_lst):
                if viewed[i]:
                    continue

                vertices_lp_gt = cp['lp_cor']
                vertices_fr_gt = cp['fr_cor']
                fr_class = cp['fr_class']

                try:
                    # if polygons_iou(lp['vertices_lp'], vertices_lp_gt) >= iou_threshold:
                    if IoU(pts_to_BBCor(*lp['vertices_lp']), pts_to_BBCor(*vertices_lp_gt)) >= iou_threshold:
                        true_or_false = True
                        viewed[i] = True
                except:
                    # traceback.print_exc()
                    if IoU(pts_to_BBCor(*lp['vertices_lp']), pts_to_BBCor(*vertices_lp_gt)) >= iou_threshold:
                        true_or_false = True
                        viewed[i] = True

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


        '''for WPOD and modified by me
        lps = []
        if not isfile(join(output_result_folder, splitext(basename(img_path))[0] + '_result.txt')):
            continue
        with open(join(output_result_folder, splitext(basename(img_path))[0] + '_result.txt'), 'r') as f:
            for i, line in enumerate(f.readlines()):
                cor, prob = line.split('-')
                cor = eval(cor)
                lps.append({'lp_cor': cor, 'prob': float(prob[:-1])})

        for lp in lps:
            true_or_false = False  # true positive or false positive

            for i, cp in enumerate(cp_lst):
                if viewed[i]:
                    continue

                vertices_lp_gt = cp['lp_cor']

                try:
                    # if polygons_iou(lp['lp_cor'], vertices_lp_gt) >= iou_threshold:
                    if IoU(pts_to_BBCor(*lp['vertices_lp']), pts_to_BBCor(*vertices_lp_gt)) >= iou_threshold:
                        true_or_false = True
                        viewed[i] = True
                except:
                    # traceback.print_exc()
                    if IoU(pts_to_BBCor(*lp['lp_cor']), pts_to_BBCor(*vertices_lp_gt)) >= iou_threshold:
                        true_or_false = True
                        viewed[i] = True

            lp_results.append([lp['prob'], true_or_false])'''


        '''for openALPR
        for lp in data['results']:

            true_or_false = False  # true positive or false positive

            lp_vertices = [[s['x'], s['y']] for s in lp['coordinates']]

            for i, cp in enumerate(cp_lst):
                if viewed[i]:
                    continue

                vertices_lp_gt = cp['lp_cor']

                try:
                    if polygons_iou(lp_vertices, vertices_lp_gt) >= iou_threshold:
                    # if IoU(pts_to_BBCor(*lp_vertices), pts_to_BBCor(*vertices_lp_gt)) >= iou_threshold:
                        true_or_false = True
                        viewed[i] = True
                except:
                    # traceback.print_exc()
                    if IoU(pts_to_BBCor(*lp_vertices), pts_to_BBCor(*vertices_lp_gt)) >= iou_threshold:
                        true_or_false = True
                        viewed[i] = True

            lp_results.append([lp['confidence'], true_or_false])'''


        '''for Sighthound
        for lp in data['objects']:

            true_or_false = False  # true positive or false positive

            annotation = lp['licenseplateAnnotation']
            lp_vertices = [[s['x'], s['y']] for s in annotation['bounding']['vertices']]

            for i, cp in enumerate(cp_lst):
                if viewed[i]:
                    continue

                vertices_lp_gt = cp['lp_cor']

                try:
                    if polygons_iou(lp_vertices, vertices_lp_gt) >= iou_threshold:
                    # if IoU(pts_to_BBCor(*lp_vertices), pts_to_BBCor(*vertices_lp_gt)) >= iou_threshold:
                        true_or_false = True
                        viewed[i] = True
                except:
                    # traceback.print_exc()
                    if IoU(pts_to_BBCor(*lp_vertices), pts_to_BBCor(*vertices_lp_gt)) >= iou_threshold:
                        true_or_false = True
                        viewed[i] = True
    
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
    plt.show()'''


    return precision, class_precision, average_iou


if __name__ == '__main__':
    gt_folder = '/home/shaoheng/Documents/Thesis_KSH/benchmark/cd_hard_vernex'
    output_result_folder = '/home/shaoheng/Documents/alpr-unconstrained-master/tmp/cd_hard_vernex'

    # coco mAP
    mAP = 0.
    for threshold in range(50, 100, 5):
        mAP += coco_mAP_vernex(output_result_folder, threshold / 100., classify_cal=False)[0]
        # mAP += coco_mAP(gt_folder, output_result_folder, threshold / 100., classify_cal=False)[0]
    mAP = mAP / 10

    # coco mAP50
    coco_map_50, class_accuracy, iou_front_rear = coco_mAP_vernex(output_result_folder, 0.5, classify_cal=False)
    # coco_map_50, class_accuracy, iou_front_rear = coco_mAP(gt_folder, output_result_folder, 0.5, classify_cal=False)
    # coco mAP75
    coco_map_75 = coco_mAP_vernex(output_result_folder, 0.75, classify_cal=False)[0]
    # coco_map_75 = coco_mAP(gt_folder, output_result_folder, 0.75, classify_cal=False)[0]

    print 'COCO mAP:', '%.1f' % (mAP * 100)
    print 'COCO mAP50:', '%.1f' % (coco_map_50 * 100)
    print 'COCO mAP75:', '%.1f' % (coco_map_75 * 100)
    print 'classification accuracy:', '%.1f' % (class_accuracy * 100)
    print 'average iou for front-rear', '%.1f' % (iou_front_rear * 100)




