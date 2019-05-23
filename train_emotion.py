"""
Creates a convolutionaly neural network (CNN) in Tensorflow and trains the network
to identify the facial emotion in an input image.
"""

import tensorflow as tf

from cvision_tools import detect_face, crop_face, convert_to_gray, resize_with_pad, over_sample, under_sample

from CohnKanadeDataset import CohnKanadeDataset

import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import os


image_height = 64
image_width = 64
image_n_channels = 1

n_epochs = 100
batch_size = 10

initial_learning_rate = 0.001
decay_steps = 2000
decay_rate = 1/2

reg_constant = 0

EMOTION_DECODER_PATH = "./model/emotion_decoder.pickle"
EMOTION_PREDICTION_MODEL_PATH = "./model/emotion_model1"

COHN_KANADE_DATA_FILE_PATH = "./data/cohn_kanade.pickle"


def prepare_emotion_data():
    """ Prepares training and test data for emotion detection model from Cohn-Kanade image dataset 
    (Face area in each image is detected and cropped)  
    Args:
    Returns:
        X_train: a numpy array of the train face image data
        X_test: a numpy array of the test face image data
        y_emotion_train: a numpy array of the train emotion lables
        y_emotion_test: a numpy array of the test emotion lables
    """
    dataset = CohnKanadeDataset()
    dataset.read(image_height=image_height, image_width=image_width)

    X_train = np.array(dataset.X_train).astype('float32')
    X_test  = np.array(dataset.X_test).astype('float32')
    X_train = X_train / 128 - 1
    X_test  = X_test  / 128 - 1


    y_emotion_train = np.array(dataset.y_train)
    y_emotion_test  = np.array(dataset.y_test)

    return X_train, X_test, y_emotion_train, y_emotion_test


def create_emotion_decoder():
    """ Creates a emotion label decoder for Cohn-Kanade dataset
    Args:
        
    Returns:
        emotion_dec: a python dictionary to decode emotion labels
    """
    emotion_dec = {0 : 'neutral',
                   1 : 'anger',
                   2 : 'contempt', 
                   3 : 'disgust', 
                   4 : 'fear', 
                   5 : 'happy',
                   6 : 'sadness', 
                   7 : 'surprise'}

    pickle.dump(emotion_dec, open(EMOTION_DECODER_PATH, 'wb'))
    return emotion_dec

def train_emotion_model(X_train, X_test, y_train, y_test):
    """ Creates a convoluational neural network (CNN) and trains the model to detect facial
     emotion in an input image
    Args:
        X_train: a numpy array of the train image data
        X_test: a numpy array of the test image data
        y_train: a numpy array of the train emotion lables
        y_test: a numpy array of the test emotion lables
    Returns:
    """
    emotion_uniques, emotion_counts = np.unique(y_train, return_counts=True)
    print(emotion_uniques)
    print(emotion_counts)
    num_emotion_outputs = len(emotion_uniques)

    with tf.name_scope("cnn_graph_emotion"):
        X = tf.placeholder(tf.float32, shape = [None, image_height, image_width, image_n_channels], name="X")
        #y = tf.placeholder(tf.float32, shape = [None, image_height, image_weight, image_n_channels])
        y = tf.placeholder(tf.int32, shape = [None])
        training = tf.placeholder_with_default(False, shape=[], name='training')

#        
        residual1 = tf.layers.conv2d(X, filters=16, kernel_size=4,
                                strides=4, padding='SAME',
                                activation=tf.nn.elu, name="residual1")

        conv11 = tf.layers.conv2d(X, filters=8, kernel_size=2,
                                strides=1, padding='SAME',
                                activation=tf.nn.elu, name="conv11")

        pool12 = tf.nn.max_pool(conv11, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        conv13 = tf.layers.conv2d(pool12, filters=16, kernel_size=4,
                                strides=1, padding='SAME',
                                activation=tf.nn.elu, name="conv13")

        pool14 = tf.nn.max_pool(conv13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        layer1_out = tf.add(residual1, pool14)
#
        layer1_out_drop = tf.layers.dropout(layer1_out, 0.3, training=training)
#        
        residual2 = tf.layers.conv2d(layer1_out_drop, filters=64, kernel_size=4,
                                strides=4, padding='SAME',
                                activation=tf.nn.elu, name="residual2")

        conv21 = tf.layers.conv2d(layer1_out, filters=32, kernel_size=2,
                                strides=1, padding='SAME',
                                activation=tf.nn.elu, name="conv21")

        pool22 = tf.nn.max_pool(conv21, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        conv23 = tf.layers.conv2d(pool22, filters=64, kernel_size=4,
                                strides=1, padding='SAME',
                                activation=tf.nn.elu, name="pool22")

        pool24 = tf.nn.max_pool(conv23, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        layer2_out = tf.add(residual2, pool24)
 #      
       
        pool4_flat = tf.reshape(layer2_out, shape=[-1, 64 * 4 * 4])
        pool4_flat_drop = tf.layers.dropout(pool4_flat, 0.5, training=training)

        fc1 = tf.layers.dense(pool4_flat_drop, 32, activation=tf.nn.elu, name="fc1")

        fc1_drop = tf.layers.dropout(fc1, 0.5, training=training)

        logits_emotion = tf.layers.dense(fc1_drop, num_emotion_outputs, name="logits_emotion")
        Y_proba_emotion = tf.nn.softmax(logits_emotion, name="Y_proba_emotion")

        global_step = tf.Variable(0, trainable=False, name="global_step")
        learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate)

    with tf.name_scope("train_emotion"):
        xentropy_emotion = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_emotion, labels=y)
        loss_emotion = tf.reduce_mean(xentropy_emotion)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss_emotion_reg = loss_emotion + reg_constant * sum(reg_losses)
        optimizer_emotion = tf.train.AdamOptimizer(learning_rate=learning_rate) # beta1=0.8
        #optimizer_emotion = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
        training_op_emotion = optimizer_emotion.minimize(loss_emotion_reg, global_step=global_step)

    with tf.name_scope("eval_emotion"):
        correct_emotion = tf.nn.in_top_k(logits_emotion, y, 1)
        accuracy_emotion = tf.reduce_mean(tf.cast(correct_emotion, tf.float32))

    with tf.name_scope("init_and_save"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    with tf.Session() as train_emotion_sess:
        init.run()
        print("Training for emotion identification")
        for epoch in range(n_epochs):
            n_it = X_train.shape[0]//batch_size
            for it in range(n_it):
                X_batch = X_train[it*batch_size:(it+1)*batch_size,:,:,:]
                y_batch = y_train[it*batch_size:(it+1)*batch_size]
                train_emotion_sess.run(training_op_emotion, feed_dict={X: X_batch, y: y_batch, training: True})
            
            acc_batch = accuracy_emotion.eval(feed_dict={X: X_batch, y: y_batch})
            acc_train = accuracy_emotion.eval(feed_dict={X: X_train[1:200], y: y_train[1:200]})
            acc_test = accuracy_emotion.eval(feed_dict={X: X_test, y: y_test})
            print(epoch, "Last batch accuracy:", acc_batch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

            save_path_emotion = saver.save(train_emotion_sess, EMOTION_PREDICTION_MODEL_PATH)

def main():

    if os.path.isfile(COHN_KANADE_DATA_FILE_PATH):
        X_train, X_test, y_train, y_test = pickle.load(open(COHN_KANADE_DATA_FILE_PATH, 'rb'))
    else:
        X_train, X_test, y_train, y_test = prepare_emotion_data()
        print(len(X_train))
        print(len(X_test))
        X_train, y_train = over_sample(X_train, y_train)
        X_test, y_test = over_sample(X_test, y_test)
        pickle.dump([X_train, X_test, y_train, y_test], open(COHN_KANADE_DATA_FILE_PATH, 'wb'))
    
    print(len(X_train))
    emotion_decoder = create_emotion_decoder()

    
    
    train_emotion_model(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()