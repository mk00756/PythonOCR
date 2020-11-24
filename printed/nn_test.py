import numpy as np 
import os
import itertools
import operator
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage import color, exposure
from scipy.misc import imread,imsave,imresize
import numpy.random as nprnd
import matplotlib
from ffnet import ffnet, mlgraph, savenet, loadnet

if __name__ == '__main__':
	# keep track of characters
	folder_string = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ123456789'
	# load the neural network
	net = loadnet('char.neural')

	# load the path of the testing set
	path = './testing_set/'
        # open a file for writing and clear it
	out_file = open('nn_result.txt','w')
	out_file.truncate(0)
        # save the size of the testing path
	path_size = 550
	print path_size
	# iterate over the files in the path using the numbered filenames
	for i in range(1,path_size):
		#read the images
		char_path = path + str(i) + '.png'
		test_image = imread(char_path).ravel()
		# save the output of the neural network
		out = net(test_image)

		# now process the output to see what character has been recognised
		maximum = -1
		index = 0
		for i in range(len(out)):
			if (out[i] > maximum):
				maximum = out[i]
				index = i
		print 'Character identified: ', folder_string[index]
		# write to the file the identified character
		out_file.write(folder_string[index])	
	out_file.close()
	print 'Done testing'
	print 'Evaluating results'

	# Now to see how many characters were correctly recognized
	string_counter = 0
	correct = 0.0
	total = 0.0
	# read in characters one by one, evaluating their accuracy
	# we are using the order of the original image to our advantage,
	# knowing that the actual characters will appear in a defined order
	with open('nn_result.txt') as f:
		while True:
			c = f.read(1)
			if not c:
				break
			if string_counter > 60:
				string_counter = 0
			if (c == folder_string[string_counter]):
				correct += 1
			total += 1
			string_counter += 1
	average = (correct / total) * 100
	print 'Accuracy:', str(average), '%'


