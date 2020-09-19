from flask import Flask, render_template, request, Response, jsonify
from camera import Camera
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

stream = ''

def gen():
    while True:
        yield b'--frame\r\n' \
              b'Content-Type: image/jpeg\r\n\r\n' + stream + b'\r\n\r\n'

        # yield (b'--frame\r\n'
        #        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('test.html')

@app.route('/update_stream', methods=['GET', 'POST'])
def update():
    global stream
    stream = request.data  # update stream
    return 'Complete update'

@app.route('/live_cam')
def live_cam():
    global stream
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)

