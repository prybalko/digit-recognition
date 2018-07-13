import os

import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.segmentation import clear_border
from skimage.transform import resize
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

dir_path = os.path.dirname(os.path.realpath(__file__))


class Classifier(object):

    def __init__(self):
        self.mnist = datasets.fetch_mldata('MNIST original', data_home=dir_path)
        self.classifier = KNeighborsClassifier()
        self.classifier.fit(self.mnist.data[::5], self.mnist.target[::5])

    def plot(self, image):
        plt.imshow(image)
        plt.show()

    def predict(self, image):
        image = np.pad(image, [(50, 50), (50, 50)], mode='constant')
        thresh = threshold_otsu(image)
        bw = closing(image > thresh, square(1))
        cleared = clear_border(bw)
        label_image = label(cleared)
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
                return self.classifier.predict(result.reshape((1, 784)))[0]
        return -1


classifier = Classifier()
