from lpdetect_model import model_WPOD
from label_processing import predicted_label_to_origin_image
from img_utility import cut_by_fourpts, read_img_from_dir
from os.path import join, basename, isdir
from drawing_utility import draw_LP_by_vertices
from os import mkdir
import numpy as np
import cv2


def find_outmap_high_prob(label):
    label_batch_free = label[0]
    img_w = 1. * label.shape[2]
    img_h = 1. * label.shape[1]
    for i, y in enumerate(label_batch_free):
        for j, x in enumerate(y):
            if x[0] > 0.9:
                print 'high possibility =%.10f (x, y) = %.2f %.2f' % (x[0], j / img_w, i / img_h)


# for testing SINGLE image
# return a list of possible license plates, in each label -> [prob, cor_after_affine]
def single_img_predict(img, input_dim=(208, 208)):
    img_shape = img.shape
    img_feed = cv2.resize(img, input_dim) / 255.
    img_feed = np.expand_dims(img_feed, 0)
    output_labels = model.predict(img_feed)
    final_labels = predicted_label_to_origin_image(img_shape, output_labels[0], 16, prob_threshold=0.9, use_nms=True)

    return final_labels


if __name__ == '__main__':

    model = model_WPOD()
    model.load_weights('/home/shaoheng/Documents/Thesis_KSH/training_result/CCPD_FR_2333/Dim208It79000Bsize64.h5')

    input_dir ='/home/shaoheng/Documents/Thesis_KSH/training_data/CCPD_FR_total2333'
    output_dir = '/home/shaoheng/Documents/Thesis_KSH/output_results/CCPD_FR_746'
    if not isdir(output_dir):
        mkdir(output_dir)
    imgs_paths = read_img_from_dir(input_dir)

    """
    for img_path in imgs_paths:
        print 'processing', img_path
        final_labels = single_img_predict(cv2.imread(img_path), input_dim=(208, 208))
        if len(final_labels) == 0:
            print img_path, 'fail to detect'

        for i, final_label in enumerate(final_labels):
            if i != 0:
                continue

            prob, vertices = final_label
            img_out = cut_by_fourpts(cv2.imread(img_path), *vertices)
            # print 'probability:', prob
            try:
                '''
                cv2.imshow('%d' % i, img_out)
                cv2.waitKey(0)
                '''
                cv2.imwrite(join(output_dir, basename(img_path)), img_out)
            except:
                print '%d LP area cutting failed' % i
                continue
    """
    img_1 = cv2.imread(imgs_paths[0])
    img_1 = cv2.resize(img_1, (208, 208))
    img_2 = cv2.imread(imgs_paths[1])
    img_2 = cv2.resize(img_2, (208, 208))
    cv2.resize(img_2, (208, 208))
    # big_img = img_1
    big_img = np.concatenate((img_1, img_2), axis=1)
    final_labels = single_img_predict(big_img, input_dim=(416, 416))
    if len(final_labels) == 0:
        print big_img, 'fail to detect'

    for i, final_label in enumerate(final_labels):
        prob, vertices = final_label
        big_img = draw_LP_by_vertices(big_img, vertices)

    cv2.imshow('img', big_img)
    cv2.waitKey(0)


