from flask import Flask, render_template, request, Response, jsonify
from camera import Camera
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

stream = ''
_action = _intent = 'Waiting...'
_action_panel = {'labels':['None']*3, 'probs':[0.0]*3}

def gen():
    while True:
        yield b'--frame\r\n' \
              b'Content-Type: image/jpeg\r\n\r\n' + stream + b'\r\n\r\n'

        # yield (b'--frame\r\n'
        #        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/state/set/action')
def set_action():
    global _action
    _action = request.args.get('action') # update action
    return 'complete'

@app.route('/state/set/action_panel')
def set_action_panel():
    global _action_panel
    _action_panel = {"labels": eval(request.args.get("labels")),
                     "probs": eval(request.args.get("probs"))}
    return 'complete'

@app.route('/state/get/action')
def get_action():
    return jsonify(action=_action)

@app.route('/state/get/action_panel')
def get_action_panel():
    return jsonify(labels=_action_panel["labels"],
                   probs=_action_panel["probs"])

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

