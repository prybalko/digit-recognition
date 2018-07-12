import os

from skimage.transform import resize
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))


class Classifier(object):

    def __init__(self):
        self.mnist = datasets.fetch_mldata('MNIST original', data_home=dir_path)
        self.classifier = KNeighborsClassifier()
        self.classifier.fit(self.mnist.data, self.mnist.target)

    def predict(self, image):
        thresh = threshold_otsu(image)
        bw = closing(image > thresh, square(1))
        # remove artifacts connected to image border
        cleared = clear_border(bw)

        # label image regions
        label_image = label(cleared)
        # image_label_overlay = label2rgb(label_image, image=image)

        fig, ax = plt.subplots(figsize=(8, 6))

        for region in regionprops(label_image):
            print region.bbox, region.area, region.centroid
            # take regions with large enough areas
            if region.area >= 10:
                # draw rectangle around segmented coins
                oy, ox, maxr, maxc = (int(x) for x in region.bbox)
                width, height = maxc - ox, maxr - oy
                cy, cx = (int(x) for x in region.centroid)
                half_side = int(max(cx-ox, width-(cx-ox), cy-oy, height-(cy-oy)))
                size_size = half_side*2
                x = cx-half_side
                y = cy-half_side
                cropped = image[y:y+size_size, x:x+size_size]
                resized_img = resize(cropped, (20, 20), anti_aliasing=True)
                resized_img *= 255. / resized_img.max()
                result = np.zeros((28, 28))
                result[4:-4, 4:-4] = resized_img
                return self.classifier.predict(result.reshape((1, 784)))[0]

classifier = Classifier()
