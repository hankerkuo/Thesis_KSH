import os
import cv2


def filename_printwithspace(path):
    files = os.listdir(path)
    for file in files:
        # leave the comma makes the print end with a space rather than a linebreak
        print os.path.join(path, file),


# return the maximum shape of images in a directory
# for CCPD -> max_width : 720, max_height : 1160
def max_width_height(path):
    max_height = 0
    max_width = 0
    for image in os.listdir(path):
        # only read jpg and png files
        if image.split('.')[-1] not in ['jpg', 'png']:
            continue
        # start to check every image's shape
        img = cv2.imread(os.path.join(path, image))
        if img.shape[0] > max_height:
            max_height = img.shape[0]
        if img.shape[1] > max_width:
            max_width = img.shape[1]
    return max_width, max_height


if __name__ == '__main__':
    target_directory = '/home/shaoheng/Documents/cars_label_FRNet/CCPD_2019_first_part/0to4370/'
    filename_printwithspace(target_directory)

