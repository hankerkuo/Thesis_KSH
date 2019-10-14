import seaborn
import numpy as np
import matplotlib.pyplot as plt
import cv2

from config import Configs
from model_define import model_and_loss
from img_utility import read_img_from_dir

c = Configs()
model = model_and_loss(training=False)

sample_dir = '/home/shaoheng/Documents/Thesis_KSH/benchmark/cd_hard_vernex'

for img_path in read_img_from_dir(sample_dir):
    img = cv2.imread(img_path)

    plt.subplot(2, 2, 1).set_xlabel('(a)')
    img_to_show = cv2.cvtColor(cv2.resize(img, c.multi_scales[0]), cv2.COLOR_BGR2RGB)
    img_to_show = np.pad(img_to_show, ((0, 0), (0, 50), (0, 0)), 'constant', constant_values=255)
    plt.imshow(img_to_show)
    plt.axis('off')

    if c.input_norm:
        div = 255.
    else:
        div = 1.

    time_spent = 0.

    final_labels = []

    img_feed = cv2.resize(img, c.multi_scales[0]) / div
    img_feed = np.expand_dims(img_feed, 0)

    output_labels = model.predict(img_feed)[0]

    plt.subplot(2, 2, 2)
    heat_map_statistic = output_labels[:, :, 0]
    heatmap = seaborn.heatmap(heat_map_statistic, cmap='jet', center=0.2, xticklabels=False, yticklabels=False,
                              square=True)

    plt.subplot(2, 2, 3)
    heat_map_statistic = output_labels[:, :, 18]
    heatmap = seaborn.heatmap(heat_map_statistic, cmap='jet', center=0.5, xticklabels=False, yticklabels=False,
                              square=True)

    plt.subplot(2, 2, 4)
    heat_map_statistic = output_labels[:, :, 19]
    heatmap = seaborn.heatmap(heat_map_statistic, cmap='jet', center=0.5, xticklabels=False, yticklabels=False,
                              square=True)

    plt.tight_layout()
    plt.show()


