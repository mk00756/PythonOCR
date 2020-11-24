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
    #create lists that will hold ALL the data and the targets
    data = []
    target = []
    # folder string which keeps track of characters loaded
    folder_string = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ123456789'
    # loop over every single character
    for i in range(len(folder_string)):
        # set up the full paths for every image of the character
        path_characters = './training_set/' + folder_string[i]
        char_filenames = sorted([filename for filename in os.listdir(path_characters)])
        char_filenames = [path_characters+'/'+filename for filename in char_filenames]
        print 'Number of training images (' + folder_string[i] + '): ' + str(len(char_filenames))

        # 1) load in the character image
        # 2) append it to the data list
        # 3) create a target to be used by the neural network
        # 4) append the target to the target list
        for filename in char_filenames:
            image = imread(filename,1)
            data.append(image.ravel())
            vector = [0] * 61
            vector[i] = 1
            target.append(vector)
    print 'Finished adding characters samples to dataset'

    print 'Training the neural network'
    inputs = 400
    hidden_nodes = 61	# Best results with no hidden nodes
    outputs = 61
    conec = mlgraph((inputs,outputs))
    # create the neural network and train it with the lists we created
    net = ffnet(conec)
    net.train_tnc(data, target, maxfun = 10000, messages=1)

    print 'Finished training neural network'
    # save the neural network for later use
    savenet(net, 'char.neural')
