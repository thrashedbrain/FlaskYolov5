import io

import flask
import torch
from PIL import Image
from flask import Flask, request, send_file

app = Flask(__name__)

DETECTION_ENDPOINT = "/v1/detect"
INIT_ENDPOINT = "/v1/init"

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')


@app.route(INIT_ENDPOINT, methods=["POST"])
def init():
    global model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
    return flask.jsonify(status="ok")


usage_counter = 0


@app.route(DETECTION_ENDPOINT, methods=["POST"])
def detect():
    global model, usage_counter
    if usage_counter == 4:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
    if not request.method == "POST":
        return

    image_file = request.files["image"]
    image_bytes = image_file.read()
    img = Image.open(io.BytesIO(image_bytes))
    result = model(img, 640)
    usage_counter = usage_counter + 1
    img_io = io.BytesIO()
    data = result.pandas().xyxy[0]
    line = data.iloc[[0]]

    xmin = line['xmin'][0]
    xmax = line['xmax'][0]

    ymin = line['ymin'][0]
    ymax = line['ymax'][0]

    area = (xmin, ymin, xmax, ymax)
    croped = img.crop(area)
    croped.save(img_io, format="JPEG")
    img_io.seek(0)
    return send_file(
        img_io,
        download_name="result",
        mimetype='image/jpeg'
    )


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=1236)
