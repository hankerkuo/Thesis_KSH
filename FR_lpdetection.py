from lpdetect_model import model_WPOD
from label_processing import predicted_label_to_origin_image
from img_utility import cut_by_fourpts
import numpy as np
import cv2

if __name__ == '__main__':

    model = model_WPOD()
    model.load_weights('/home/shaoheng/Documents/Thesis_KSH/training_result/CCPD_FR_746_dataprovider/Dim208It3000Bsize64.h5')

    img = cv2.imread('/home/shaoheng/Documents/Thesis_KSH/training_data/CCPD_FR_total746/187&148_91&137_93&101_189&112.jpg')
    # img_feed = cv2.resize(img, (500, 500))
    img_feed = np.expand_dims(img, 0)
    output_label = model.predict(img_feed)

    label_batch_free = output_label[0]

    img_w = 1. * output_label.shape[2]
    img_h = 1. * output_label.shape[1]
    for i, y in enumerate(label_batch_free):
        for j, x in enumerate(y):
            if x[0] > 0.9:
                print 'high possibility =%.10f (x, y) = %.2f %.2f' % (x[0], j / img_w, i / img_h)

    final_labels = predicted_label_to_origin_image(output_label, 16, prob_threshold=0.8)

    for i, final_label in enumerate(final_labels):
        img_out = cut_by_fourpts(img, *final_label)
        try:
            cv2.imshow('%d' % i, img_out)
            cv2.waitKey(0)
        except:
            continue

    print final_labels