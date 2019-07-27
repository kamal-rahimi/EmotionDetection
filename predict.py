"""
"""

import tensorflow as tf 
import numpy as np
import pickle
import argparse
import os

from cvision_tools import read_image, detect_face, crop_face, convert_to_gray, resize_with_pad
from CohnKanadeDataset import CohnKanadeDataset

import cv2 as cv2

image_height = 64
image_width = 64
image_n_channels = 1

EMOTION_DECODER_PATH = "./model/emotion_decoder.pickle"
EMOTION_PREDICTION_MODEL_PATH = "./model/emotion_model"

TEST_DATA_PATH = "./data/test/"


def prepare_image(image):
    image = np.array(image)
    face  = crop_face(image)
    face  = resize_with_pad(face, image_height, image_width)
    face  = convert_to_gray(face)
    face  = np.array(face)
    face  = face.reshape(-1, image_height, image_width, image_n_channels)
    face  = face.astype('float32')
    face  = face / 128 - 1
    return image, face


def indetify_emotion(face):
    emotion_decoder = pickle.load(open(EMOTION_DECODER_PATH, 'rb'))

    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(EMOTION_PREDICTION_MODEL_PATH + '.meta')
    graph = tf.get_default_graph()

    X = graph.get_tensor_by_name("cnn_graph_emotion/X:0")
    Y_proba_emotion = graph.get_tensor_by_name("cnn_graph_emotion/Y_proba_emotion:0")

    with tf.Session() as predict_sess:
        saver.restore(predict_sess, EMOTION_PREDICTION_MODEL_PATH)
        probs = Y_proba_emotion.eval(feed_dict={X: face})
        emotion_index = np.argmax(probs)
        predicted_emotion = emotion_decoder[emotion_index]
        prob_emotion = probs[0, emotion_index]

        print(probs)
        print( predicted_emotion )
        print("Confidennce={}".format(prob_emotion))

    return predicted_emotion, prob_emotion


def display_image(image, predicted_label, prob_label, show_face_area=True, wait_time=0, image_out_path=''):
    image = resize_with_pad(image, 600, 600)
    if show_face_area:
        x, y, w, h = detect_face(image)
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.putText(image, "{}".format(predicted_label), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)    
    cv2.putText(image, "Prob: {:.2f}".format(prob_label), (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(wait_time)
    print(image_out_path)
    if image_out_path != '':
        cv2.imwrite(image_out_path, image)

def predict_image(image_path, image_out_path=''):
    image = read_image(image_path)
    image, face = prepare_image(image)
    predicted_emotion, prob_emotion = indetify_emotion(face)
    display_image(image, predicted_emotion, prob_emotion, image_out_path=image_out_path)


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", type=str, default="", help="Specify the path to the input image")
    args = vars(ap.parse_args())
    image_path = args["path"]
    
    if (image_path != ""):
        predict_image(image_path)
    else:
        for filename in os.listdir(TEST_DATA_PATH):
            if os.path.isfile(os.path.join(TEST_DATA_PATH, filename)):
                image_path = os.path.join(TEST_DATA_PATH, filename)
                image_out_path = os.path.join( os.path.join(TEST_DATA_PATH, "out/"), filename)
                predict_image(image_path, image_out_path)


if __name__ == "__main__":
    main()