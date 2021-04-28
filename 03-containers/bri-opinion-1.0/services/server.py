from flask import jsonify
from flask import Flask
from flask import request
from flask import make_response
from flask_cors import CORS

from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop

import opinion

app = Flask(__name__)
CORS(app)

@app.route('/opinion')
def hello_world():
	return 'Opinion Mining!'

@app.route('/opinion/mine',methods=["POST"])
def return_opinions():
    sentences = request.json["sentences"]
    opinions = opinion.find_opinions(sentences)
    res = {'opinion': opinions}
    return make_response(jsonify(res), 200)

@app.route('/opinion/status',methods=["GET","POST"])
def return_status():
    return make_response(jsonify({'phrase' : 'Up and running'}), 200)

if __name__ == '__main__':
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(6001)
    IOLoop.instance().start()