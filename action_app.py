from flask import Flask, render_template, request, Response, jsonify
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

stream = ''
_action = 'Waiting...'
_action_hist = ['Waiting...']

def gen_for_livecam():
    while True:
        yield b'--frame\r\n' \
              b'Content-Type: image/jpeg\r\n\r\n' + stream + b'\r\n\r\n'
        # sleep(0.01)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('web_demo_offline_1.html')

@app.route('/update_stream', methods=['GET', 'POST'])
def update_stream():
    global stream
    stream = request.data  # update stream
    return 'complete'

@app.route('/state/set/action')
def set_action():
    global _action
    _action = request.args.get('action') # update action
    return 'complete'

@app.route('/state/get/action')
def get_action():
    return jsonify(action=_action)

@app.route('/live_cam')
def live_cam():
    # global stream
    return Response(gen_for_livecam(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True)
    #app.run(host='192.168.0.241', port=5001)
