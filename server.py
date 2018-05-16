import base64
import io

import cherrypy
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


class HelloWorld(object):

    @cherrypy.expose
    def index(self):
        return open('index.html')

    @cherrypy.expose
    @cherrypy.tools.allow(methods=['POST'])
    @cherrypy.tools.json_out()
    def recognize(self):
        img_data = cherrypy.request.body.read().split('base64,')[1]
        image = Image.open(io.BytesIO(base64.b64decode(img_data))).convert('L')
        np_img = np.asarray(image)
        return {'status': 'ok'}


if __name__ == '__main__':
    cherrypy.quickstart(HelloWorld())
