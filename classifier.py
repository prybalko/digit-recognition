import os

from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle

dir_path = os.path.dirname(os.path.realpath(__file__))


class Classifier(object):

    def __init__(self):
        mnist = datasets.fetch_mldata('MNIST original', data_home=dir_path)
        self.classifier = KNeighborsClassifier()
        self.classifier.fit(mnist.data[::2], mnist.target[::2])

    def predict(self, image):
        return self.classifier.predict(image.reshape((1, 784)))


classifier = Classifier()
