from lpdetect_model import model_WPOD
from label_processing import predicted_label_to_origin_image
from img_utility import cut_by_fourpts, read_img_from_dir
from os.path import join, basename, isdir, splitext
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
def single_img_predict(img, input_dim=(208, 208), input_norm=True):
    img_shape = img.shape
    if input_norm:
        div = 255.
    else:
        div = 1.
    img_feed = cv2.resize(img, input_dim) / div
    img_feed = np.expand_dims(img_feed, 0)
    output_labels = model.predict(img_feed)
    final_labels = predicted_label_to_origin_image(img_shape, output_labels[0], 16, prob_threshold=0.01, use_nms=True)

    return final_labels


if __name__ == '__main__':

    model = model_WPOD()
    model.load_weights('/home/shaoheng/Documents/Thesis_KSH/training_result/CCPD_FR_2333/Dim208It79000Bsize64.h5')
    # model.load_weights('/home/shaoheng/Documents/Thesis_KSH/training_result/CCPD_FR_746_dataprovider/Dim208It303000Bsize64.h5')

    input_dir ='/home/shaoheng/Documents/alpr-unconstrained-master/samples/kr'
    output_dir = '/home/shaoheng/Documents/Thesis_KSH/output_results/kr'
    if not isdir(output_dir):
        mkdir(output_dir)
    imgs_paths = read_img_from_dir(input_dir)

    for img_path in imgs_paths:
        print 'processing', img_path
        final_labels = single_img_predict(cv2.imread(img_path), input_dim=(500, 500), input_norm=False)
        print '%d LPs found' % len(final_labels)
        if len(final_labels) == 0:
            print img_path, 'fail to detect'

        for i, final_label in enumerate(final_labels):
            '''
            if i != 0:
                continue
            '''
            prob, vertices = final_label
            img_out = cut_by_fourpts(cv2.imread(img_path), *vertices)
            # print 'probability:', prob
            try:
                cv2.imwrite(join(output_dir, splitext(basename(img_path))[0]) + '_%d.jpg' % i, img_out)
                print 'writing', join(output_dir, splitext(basename(img_path))[0]) + '_%d.jpg' % i, 'prob:', prob
            except:
                print '%d LP area cutting failed' % i
                continue




