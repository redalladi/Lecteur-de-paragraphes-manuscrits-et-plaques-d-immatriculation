from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2
import glob
import os
import warnings
warnings.filterwarnings("ignore")
import pytesseract
from flask_cors import CORS, cross_origin
import base64
from PIL import Image
import io
import json
import matplotlib.pyplot as plt

import urllib.request as req_url
from IPython import get_ipython

# home made modules
import scanner
import text_detection
import prediction


# pytesseract.pytesseract.tesseract_cmd = r'H:\\Program Files\\Tesseract-OCR\\tesseract.exe'


# Initialize the Flask application
app = Flask(__name__)
run_with_ngrok(app)
cors = CORS(app)


img_list=[]

os.chdir(os.path.dirname(os.path.realpath(__file__)))


def processDocument(doc, model):
    save_img_step = True
    # get document scan (binary, transformed)
    scan = scanner.get_scan(doc, save_img_step)
    if scan is None:
        return Response(response=jsonpickle.encode({"txt_read":"aucun document détecté"}), status=200, mimetype="application/json")
    # get spatialy sorted list of paragraph images
    text_regions = text_detection.get_text_regions(scan, save_img_step)
    if text_regions == []:
        return Response(response=jsonpickle.encode({"txt_read":"aucune région de texte détectée"}), status=200, mimetype="application/json")
    print("text_regions :" +str(len(text_regions)))

    text_lines = []
    for i in range(len(text_regions)):
        text_lines.append(text_detection.get_lines(text_regions[i], i, save_img_step))


    text_words = []
    for i in range(len(text_lines)):
        lines = []
        for l in range(len(text_lines[i])):
            lines.append(text_detection.get_words(text_lines[i][l], i, l, save_img_step))
        text_words.append(lines)


    predictions = []
    for i, tr in enumerate(text_words):
        pred_lines = []
        for l, line in enumerate(tr):
            pred_lines.append(prediction.predict_words(line, model, i, l))
        predictions.append(pred_lines)

    predictions = list(map(lambda paragraph: list(map(lambda line: " ".join(line), paragraph)), predictions))
    predictions = list(map(lambda paragraph: "\n".join(paragraph), predictions))
    predictions = "\n\n".join(predictions)

    print('number of text regions:', len(text_words))
    print('number of text lines:', *list(len(i) for i in text_words))
    print()
    print(predictions)
    return predictions
#cv2.imshow('hi',img_list[0])


# route http posts to this method
@app.route('/post_maj', methods=['POST'])
@cross_origin()
def post_img_maj():
    print('request to predicet upper case text')
    r = request
    # cleaning debug image folder
    for f in glob.glob('./img_debug/*.jpg'):
        os.remove(f)
    # convert string of image data to uint8
    try:
        imtxt=r.json['data'].split(',')[1]
        print('ok')
    except:
        imtxt=r.json['data']
        print("couldn't split")
    # convert to openCV image
    image_64 = io.BytesIO(base64.b64decode(imtxt))
    document = cv2.imdecode(np.fromstring(image_64.read(), np.uint8), 1)
    model = prediction.get_model("f11.h5", "CRNN_architecture_capitale.json")
    text_read = processDocument(document, model)
    return Response(response=jsonpickle.encode({'txt_read':text_read}), status=200, mimetype="application/json")


@app.route('/post_min', methods=['POST'])
@cross_origin()
def post_img_min():
    print('request to predicet lower case text')
    r = request
    # cleaning debug image folder
    for f in glob.glob('./img_debug/*.jpg'):
        os.remove(f)
    # convert string of image data to uint8
    try:
        imtxt=r.json['data'].split(',')[1]
        print('ok')
    except:
        imtxt=r.json['data']
        print("couldn't split")
    # convert to openCV image
    image_64 = io.BytesIO(base64.b64decode(imtxt))
    document = cv2.imdecode(np.fromstring(image_64.read(), np.uint8), 1)
    model = prediction.get_model("htr6.h5", "CRNN_architecture.json")
    text_read = processDocument(document, model)

    return Response(response=jsonpickle.encode({'txt_read':text_read}), status=200, mimetype="application/json")


# route http posts to this method

@app.route('/test_get', methods=['GET'])
@cross_origin()
def show():
    response = {'txt_read': "bravo, l'API marche"}
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")

if __name__ == "__main__":
    # start flask app
    app.run()
