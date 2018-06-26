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

        # plt.imshow(resized_img)
        # plt.show()
        prediction = classifier.predict(original_img)
        return {'status': 'ok', 'prediction': int(0)}


if __name__ == '__main__':
    cherrypy.quickstart(HelloWorld())
