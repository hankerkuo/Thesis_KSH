"""
This file is used to delete the images without labels
(Delete the images without corresponding txt file)
AND THEN
Generate the train.txt and test.txt for YOLO
"""
from os import remove
from os.path import splitext, isfile, join, basename
from glob import glob, iglob

from img_utility import read_img_from_dir


'''
First part
'''


def clean_data(dir):
	# define the dir path which includes all the images and labels

	img_lst = read_img_from_dir(dir)  # modify to png if needed

	for img in img_lst:
		split_former = splitext(img)[0]
		if isfile(split_former + '.txt') is False:
			remove(img)
			print 'remove', img


'''
second part, source:https://github.com/ManivannanMurugavel/YOLO-Annotation-Tool/blob/master/process.py
'''


def training_data_sumup(dir):
	# Percentage of images to be used for the test set
	percentage_test = 0

	# Create and/or truncate train.txt and test.txt
	file_train = open('train.txt', 'w')
	file_test = open('test.txt', 'w')

	# Populate train.txt and test.txt
	counter = 1
	if percentage_test:
		index_test = round(100 / percentage_test)
	for pathAndFilename in iglob(join(dir, "*.jpg")):
		title, ext = splitext(basename(pathAndFilename))
		if percentage_test and counter == index_test:
			counter = 1
			file_test.write(dir + "/" + title + '.jpg' + "\n")
		else:
			file_train.write(dir + "/" + title + '.jpg' + "\n")
			counter = counter + 1


if __name__ == '__main__':
	# root path
	dir = '/home/shaoheng/Documents/cars_label_FRNet/cars/foryolo'
	clean_data(dir)
	training_data_sumup(dir)