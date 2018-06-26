import os

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square

dir_path = os.path.dirname(os.path.realpath(__file__))


class Classifier(object):

    def __init__(self):
        self.mnist = datasets.fetch_mldata('MNIST original', data_home=dir_path)
        self.classifier = KNeighborsClassifier()
        self.classifier.fit(self.mnist.data[::5], self.mnist.target[::5])

    def predict(self, image):
        thresh = threshold_otsu(image)
        bw = closing(image > thresh, square(1))
        # remove artifacts connected to image border
        cleared = clear_border(bw)

        # label image regions
        label_image = label(cleared)
        # image_label_overlay = label2rgb(label_image, image=image)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(cleared)

        for region in regionprops(label_image):
            print region.bbox, region.area, region.centroid
            # take regions with large enough areas
            if region.area >= 10:
                # draw rectangle around segmented coins
                oy, ox, maxr, maxc = region.bbox
                width, height = maxc - ox, maxr - oy
                rect = mpatches.Rectangle((ox, oy), width, height,
                                          fill=False, edgecolor='green', linewidth=2)
                ax.add_patch(rect)
                cy, cx = region.centroid

                half_side = max(cx-ox, width-(cx-ox), cy-oy, height-(cy-oy))
                big_rect = mpatches.Rectangle((cx-half_side, cy-half_side), half_side*2, half_side*2,
                                              fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(big_rect)
                ax.scatter(cx, cy, s=400, c='C0', marker='+', linewidth=2)

        ax.set_axis_off()
        plt.tight_layout()
        plt.show()



        # resized_img = resize(image, (28, 28), anti_aliasing=True)
        # resized_img *= 255. / resized_img.max()
        # return self.classifier.predict(image.reshape((1, 784)))[0]


classifier = Classifier()
