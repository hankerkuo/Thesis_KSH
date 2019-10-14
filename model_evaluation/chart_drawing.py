from os import listdir
from os.path import splitext, join

import matplotlib.pyplot as plt
import re



# summarize the txt files under the txts_folder
# and plot the to_print - iteration, y - x chart
# to_print can be in ['mAP', 'mAP50', 'mAP75', 'class_acc', 'FR_IoU']
def draw_something_to_iteration(txts_folder, to_print=''):
    assert to_print in ['mAP', 'mAP50', 'mAP75', 'class_acc', 'FR_IoU'], \
                        'to_print is not defined!'
    info_dict = {'mAP': 1, 'mAP50': 2, 'mAP75': 3, 'class_acc': 4, 'FR_IoU': 5}

    records = []

    for txt_file in listdir(txts_folder):
        if splitext(txt_file)[1] not in ['.txt']:
            continue

        ite = re.match('.*It(.*)Bsize', txt_file).group(1)

        with open(join(txts_folder, txt_file), 'r') as f:
            contents = f.readlines()
            mAP, mAP50, mAP75, class_acc, fr_iou = \
                [float(re.match(r'.*:(.*)\n', content).group(1)) for content in contents]

            records.append([int(ite), mAP, mAP50, mAP75, class_acc, fr_iou])

    records.sort(key=lambda x: x[0])
    records = records[:866]

    plt.plot([x[0] for x in records],
             [y[info_dict[to_print]] for y in records], 'dodgerblue')

    font = {'family': 'sans-serif', 'color': 'black',
            'weight': 'normal', 'size': 16}

    plt.xlabel('iteration', fontdict=font)
    plt.ylabel(to_print, fontdict=font)

    plt.show()


if __name__ == '__main__':
    draw_something_to_iteration('/home/shaoheng/Documents/Thesis_KSH/benchmark/mAP_weights/vernex_lpfr_class_dim512_thres0.3',
                                to_print='FR_IoU')

    # ['mAP', 'mAP50', 'mAP75', 'class_acc', 'FR_IoU']
