import cherrypy


class HelloWorld(object):

    @cherrypy.expose
    def index(self):
        return open('index.html')


if __name__ == '__main__':
    cherrypy.quickstart(HelloWorld())
