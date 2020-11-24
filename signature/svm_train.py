import numpy as np 
import os
import itertools
import operator
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.feature import hog
from skimage import color, exposure
from scipy.misc import imread,imsave,imresize
import numpy.random as nprnd
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.svm import LinearSVC
import matplotlib
import pickle

if __name__ == '__main__':
    #create lists that will hold ALL the data and the targets
    data = []
    target = []
    folder_string = 'AB'
    for i in range(len(folder_string)):
        path_characters = './training_set/' + folder_string[i]

        #get all the images in the above folders
        #note that we are looking for the png extension
        char_filenames = sorted([filename for filename in os.listdir(path_characters)])
 
        #add the full path to all the filenames
        char_filenames = [path_characters+'/'+filename for filename in char_filenames]
        print 'Number of training images (' + folder_string[i] + '): ' + str(len(char_filenames))

        #fill the training dataset
        # the flow is
        # 1) load sample
        # 2) save them in the data list that holds all the hog features
        # 3) also save the target of that sample in the labels list
        for filename in char_filenames:
            #read the images
            image = imread(filename,1)
            #get hog features
            hog_img = hog(image, orientations=12, pixels_per_cell=(2, 2),
                    cells_per_block=(1, 1))
            data.append(hog_img)
            target.append(i)
    print 'Finished adding samples to dataset'

    print 'Training the SVM'
    #create the SVC
    clf = LinearSVC(dual=False,verbose=1,max_iter=2500)
    #train the svm
    clf.fit(data, target)

    #pickle it - save it to a file
    pickle.dump( clf, open( "char.detector", "wb" ) )
    print 'Finished training SVM - detector saved'
