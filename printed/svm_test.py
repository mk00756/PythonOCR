from skimage.feature import hog
from scipy.misc import imread,imresize
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import pickle

### Remove forced-depreciation warnings about outdated python modules
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
### End warning removal

if __name__ == '__main__':
    # convert class to character
    folder_string = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ123456789'
    #load the detector
    clf = pickle.load( open("char.detector","rb"))
    # load the path of the testing set
    path = './testing_set/'
    # open a file for writing and clear it
    out_file = open('svm_result.txt','w')
    out_file.truncate(0)
    # save the size of the testing path, we could calculate this instead
    path_size = 550
    print path_size, "test images found"
    # iterate over the files in the path using the numbered filenames
    for i in range(1,path_size):
        #read the images
        char_path = path + str(i) + '.png'
        test_image = imread(char_path,1)
        hog_img = hog(test_image, orientations=12, pixels_per_cell=(2, 2),
                    cells_per_block=(1, 1))
        # save the output of the neural network
        out = int(clf.predict(hog_img.reshape(1,-1)))
        # write to the file the identified character
        out_file.write(folder_string[out])	
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
    with open('svm_result.txt') as f:
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
