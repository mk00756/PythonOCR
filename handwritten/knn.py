import numpy as np
import os
from skimage import color, exposure
from scipy.misc import imread,imsave,imresize
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.neighbors import KNeighborsClassifier

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
            target.append(i)
    print 'Finished adding characters samples to dataset'

    print 'Now training the knn classifier'
    k=3
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(data, target)
    print 'Trained the knn-classifier'

    print 'Testing the knn-classifier'
    # load the path of the testing set
    path = './testing_set/'
    # open a file for writing and clear it
    out_file = open('knn_result.txt','w')
    out_file.truncate(0)
    # save the size of the testing path, we could calculate this instead
    path_size = 62
    print path_size, "test images found"
    # iterate over the files in the path using the numbered filenames
    for i in range(1,path_size):
        #read the images
        char_path = path + str(i) + '.png'
        test_image = imread(char_path).ravel().reshape(1,-1)
        # save the output of the neural network
        out = knn_classifier.predict_proba(test_image)[0]

        # now process the output to see what character has been recognised
        maximum = -1
        index = 0
        for j,result in enumerate(out):
            if (result > maximum):
                maximum = result
                index = j
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
    with open('knn_result.txt') as f:
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
