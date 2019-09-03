from lpdetect_model import model_WPOD
from label_processing import predicted_label_to_origin_image
from img_utility import cut_by_fourpts
import numpy as np
import cv2

if __name__ == '__main__':

    model = model_WPOD()
    model.load_weights('/home/shaoheng/Documents/Thesis_KSH/training_result/CCPD_FR_746_dataprovider/Dim208It303000Bsize64.h5')

    img = cv2.imread('/home/shaoheng/Documents/cars_label_FRNet/ccpd_dataset/ccpd_rotate/02375-109_71-321&355_525&490-510&479_337&417_338&365_511&426-0_0_10_1_29_33_33-174-34.jpg')
    # img_feed = cv2.resize(img, (3000, 3000))
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
        prob, vertices = final_label
        img_out = cut_by_fourpts(img, *vertices)
        print prob
        try:
            cv2.imshow('%d' % i, img_out)
            cv2.waitKey(0)
        except:
            continue

    print final_labels