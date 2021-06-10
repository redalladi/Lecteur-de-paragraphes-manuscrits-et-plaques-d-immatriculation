import os
import cv2
import random
import numpy as np
import pandas as pd

import tensorflow as tf
import keras
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout
from keras.optimizers import Adam


def num_to_label(num):

    alphabets = u"abcdefghijklmnopqrstuvwxyz-' !()"
    max_str_len = 24 # max length of input labels
    num_of_characters = len(alphabets) + 1 # +1 for ctc pseudo blank
    num_of_timestamps = 64 # max length of predicted labels
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret+=alphabets[ch]
    return ret


def get_model(model_name, architecture_json):
    print('TensorFlow version:', tf.__version__)
    print('Keras version:', keras.__version__)
    print('cv2 version:', cv2.__version__)
    model_final, model = load_trained_model(model_name, architecture_json)
    # the loss calculation occurs elsewhere, so we use a dummy lambda function for the loss
    model_final.compile(loss={'ctc': lambda y_true, y_pred: y_pred},
                        optimizer=Adam(lr = 0.0001))
    return model



# LOAD MODEL

def load_trained_model(model_path, architecture_json):
    with open(architecture_json) as f:
        json_string = f.read()
    ctc = keras.models.model_from_json(json_string)
    ctc.load_weights(model_path)
    model = Model(inputs=ctc.input, outputs=ctc.get_layer("softmax").output)
    return ctc, model






# preprocess

def resize_to_fit_dim(img, height=64, width=256):
	(h,w) = img.shape
	dim = (w, h)
	if w > width or h > height:
		if width/float(w) < height/float(h):
			r = width / float(w)
			dim = (width, int(h*r))
		else:
			r = height / float(h)
			dim = (int(w * r), height)
	if w<width and h<height:
		if w/float(width) > h/float(height):
			r = w/float(width)
			dim = (width, int(h/r))
		else:
			r = h/float(height)
			dim = (int(w/r), height)
	resized = cv2.resize(img, dim)
	return resized

def preprocess(img, tr, l, w):
    resized = resize_to_fit_dim(img, 64, 256)
    x, y = resized.shape
    preprocessed = np.ones([64, 256])*255 # blank white image
    start_x = int((64 - x)/2)
    start_y = int((256 - y)/2)
    preprocessed[start_x:start_x+x, start_y:start_y+y] = resized
    cv2.imwrite('./img_debug/11_words_preprocessed'+str(tr)+'_'+str(l)+'_'+str(w)+'.jpg', preprocessed)
    return cv2.rotate(preprocessed, cv2.ROTATE_90_CLOCKWISE)/255.



# prediction

def predict_words(words, model, tr, l):
    batch = []
    for w,img in enumerate(words):
        batch.append(preprocess(img, tr, l, w))
    if (len(batch) == 0):
        return ''
    batch = np.array(batch).reshape(-1, 256, 64, 1)
    pred = model.predict(batch)
    decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], greedy=True)[0][0])
    res = list(map(lambda x: num_to_label(x), decoded))
    return res
