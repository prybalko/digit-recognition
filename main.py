import os

from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle

dir_path = os.path.dirname(os.path.realpath(__file__))

mnist = datasets.fetch_mldata('MNIST original', data_home=dir_path)
mnist.data, mnist.target = shuffle(mnist.data, mnist.target)

images_and_labels = list(zip(mnist.data, mnist.target))

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = mnist.data.shape[0]
data = mnist.data

# Create a classifier: a KNN classifier
classifier = KNeighborsClassifier()

# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples // 2], mnist.target[:n_samples // 2])

# Now predict the value of the digit on the second half:
expected = mnist.target[-10:]
predicted = classifier.predict(data[-10:])

images_and_predictions = list(zip(mnist.data[-10:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:-1]):
    plt.subplot(3, 3, index+1)
    plt.axis('off')
    image = image.reshape((28, 28))
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()
