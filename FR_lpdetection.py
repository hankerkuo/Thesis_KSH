from lpdetect_model import model_WPOD
from label_processing import predicted_label_to_origin_image
from img_utility import cut_by_fourpts, read_img_from_dir
from os.path import join, basename, isdir, splitext
from os import mkdir
from src_others.hourglass import create_hourglass_network, bottleneck_block
from drawing_utility import draw_LP_by_vertices
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
def single_img_predict(img, input_dim=(208, 208), input_norm=True):
    img_shape = img.shape
    if input_norm:
        div = 255.
    else:
        div = 1.
    img_feed = cv2.resize(img, input_dim) / div
    img_feed = np.expand_dims(img_feed, 0)
    output_labels = model.predict(img_feed)
    final_labels = predicted_label_to_origin_image(img_shape, output_labels[0], 16,
                                                   prob_threshold=0.9, use_nms=True, side=3.5)

    return final_labels


if __name__ == '__main__':

    '''WPOD'''
    model = model_WPOD()
    # this weight doesn't use norm, stride=16, side=3.5
    model.load_weights('/home/shaoheng/Documents/Thesis_KSH/Link to training_result/CCPD_FR_2333/Dim208It79000Bsize64.h5')


    '''Hourglass
    model = create_hourglass_network(None, num_stacks=2, num_channels=256, inres=(256, 256), outres=(64, 64),
                                     bottleneck=bottleneck_block)
    # this weight uses norm, stride=4, side=16.
    model.load_weights('/home/shaoheng/Documents/Thesis_KSH/Link to training_result/CCPD_FR_2333_hourglass/Dim256It123000Bsize16Lr0.00025.h5')
    '''

    input_dir ='/home/shaoheng/Documents/Thesis_KSH/training_data/CCPD_FR_total746'
    output_dir = '/home/shaoheng/Documents/Thesis_KSH/output_results/weight_CCPD_FR_2333_Dim208It79000Bsize64/CCPD_FR_total746'
    if not isdir(output_dir):
        mkdir(output_dir)
    imgs_paths = read_img_from_dir(input_dir)

    for img_path in imgs_paths:

        print 'processing', img_path
        final_labels = single_img_predict(cv2.imread(img_path), input_dim=(256, 256), input_norm=False)
        print '%d LPs found' % len(final_labels)

        if len(final_labels) == 0:
            print img_path, 'fail to detect'
            continue

        img = cv2.imread(img_path)

        for i, final_label in enumerate(final_labels):
            '''
            if i != 0:
                continue
            '''
            prob, vertices = final_label
            try:
                img = draw_LP_by_vertices(img, vertices)
            except:
                print '%d LP area cutting failed' % i
                continue

        cv2.imwrite(join(output_dir, splitext(basename(img_path))[0]) + '_%d.jpg' % i, img)




