import os
import pickle
import urllib
from shutil import copyfileobj

import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.segmentation import clear_border
from skimage.transform import resize
from sklearn import svm, metrics
from sklearn.datasets import get_data_home, fetch_mldata
from sklearn.utils import shuffle

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def fetch_mnist(data_home=None):
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    data_home = get_data_home(data_home=data_home)
    data_home = os.path.join(data_home, 'mldata')
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    mnist_save_path = os.path.join(data_home, "mnist-original.mat")
    if not os.path.exists(mnist_save_path):
        mnist_url = urllib.urlopen(mnist_alternative_url)
        with open(mnist_save_path, "wb") as matlab_file:
            copyfileobj(mnist_url, matlab_file)


def get_classifier():
    model_file = 'classifier.pickle'
    if os.path.isfile(model_file):
        with open(model_file, 'r') as f:
            return pickle.load(f)
    fetch_mnist(data_home=DIR_PATH)
    mnist = fetch_mldata("MNIST original", data_home=DIR_PATH)
    images = mnist.data
    images = images / 255. * 2 - 1
    target = mnist.target

    images, target = shuffle(images, target)
    n_samples = len(images)
    data = images
    # Create a classifier: a support vector classifier
    classifier = svm.SVC()
    # We learn the digits on the first half of the digits
    classifier.fit(data[:n_samples // 2], target[:n_samples // 2])

    # Now predict the value of the digit on the second half:
    expected = target[n_samples // 2:]
    predicted = classifier.predict(data[n_samples // 2:])

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    with open(model_file, 'wb') as f:
        pickle.dump(classifier, f)
    return classifier


CLASSIFIER = get_classifier()


def predict(image):
    image = np.pad(image, [(50, 50), (50, 50)], mode='constant')
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(1))
    cleared = clear_border(bw)
    label_image = label(cleared)

    results = []
    for region in regionprops(label_image):
        if region.area >= 10:
            oy, ox, maxr, maxc = (np.round(x) for x in region.bbox)
            width, height = maxc - ox, maxr - oy

            side = max(width, height)
            dx = (side-width)/2
            dy = (side-height)/2
            x = ox - dx
            y = oy - dy
            gcy, gcx = (np.round(x) for x in region.centroid)
            cx_proportion = (gcx - x) / float(side)
            cy_proportion = (gcy - y) / float(side)
            cropped = label_image[y:y + side, x:x + side]

            normalized = np.interp(cropped, (cropped.min(), cropped.max()), (0, 255))
            resized_img = resize(normalized, (20, 20), preserve_range=True, anti_aliasing=True)
            cx = int(np.round(cx_proportion * 20))
            cy = int(np.round(cy_proportion * 20))

            result = np.zeros((28, 28))
            pad_x = max(4 + 10 - cx, 0)
            pad_y = max(4 + 10 - cy, 0)
            result[pad_y:pad_y+20, pad_x:pad_x+20] = resized_img
            result = result / 255. * 2 - 1
            results.append(CLASSIFIER.predict(result.reshape((1, 784)))[0])
    return results or [-1]
