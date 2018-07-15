import base64
import io

import cherrypy
import numpy as np
from PIL import Image

import classifier


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
        prediction = classifier.predict(original_img)
        return {'status': 'ok', 'prediction': int(prediction)}


if __name__ == '__main__':
    cherrypy.quickstart(HelloWorld())
