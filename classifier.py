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

    def _add_padding(self, img):
        return np.pad(img, [(4, 4), (4, 4)], mode='constant')

    def _nornalize(self, img):
        normalized_img = resize(img, (20, 20), anti_aliasing=True)
        normalized_img *= 255. / img.max()
        return normalized_img

    def _crop_to_region(self, image, region):
        oy, ox, maxr, maxc = (int(x) for x in region.bbox)
        width, height = maxc - ox, maxr - oy
        cy, cx = (int(x) for x in region.centroid)
        half_side = int(max(cx - ox, width - (cx - ox), cy - oy, height - (cy - oy)))
        size_size = half_side * 2
        x = cx - half_side
        y = cy - half_side
        return image[y:y + size_size, x:x + size_size]

    def plot(self, image):
        plt.imshow(image)
        plt.show()

    def predict(self, image):
        thresh = threshold_otsu(image)
        bw = closing(image > thresh, square(1))
        cleared = clear_border(bw)
        label_image = label(cleared)
        label_image = self._add_padding(label_image)
        for region in regionprops(label_image):
            if region.area >= 10:
                cropped = self._crop_to_region(image, region)
                img = self._nornalize(cropped)
                img = self._add_padding(img)
                return self.classifier.predict(img.reshape((1, 784)))[0]
        return -1


classifier = Classifier()
