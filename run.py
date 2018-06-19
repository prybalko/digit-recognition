import base64
import io

import cherrypy
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage.transform import resize

from classifier import classifier


class HelloWorld(object):

    @cherrypy.expose
    def index(self):
        return open('index.html')

    @cherrypy.expose
    @cherrypy.tools.allow(methods=['POST'])
    @cherrypy.tools.json_out()
    def recognize(self):
        img_data = cherrypy.request.body.read().split('base64,')[1]
        bin_image = Image.open(io.BytesIO(base64.b64decode(img_data))).convert('L')
        original_img = np.asarray(bin_image)
        resized_img = resize(original_img, (28, 28), anti_aliasing=True)
        resized_img *= 255./resized_img.max()
        # plt.imshow(resized_img)
        # plt.show()
        prediction = classifier.predict(resized_img)[0]
        return {'status': 'ok', 'prediction': int(prediction)}


if __name__ == '__main__':
    cherrypy.quickstart(HelloWorld())
