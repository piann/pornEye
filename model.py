# maching learning model for training, predicting, evaluating

from common import *
import tensorflow as tf
import numpy as np

class Model:


    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.learningRate = 0.0003
        self.isTraining = tf.placeholder(tf.bool)
        self.X = tf.placeholder(tf.float32, shape=[None,  RESIZED_HEIGHT * RESIZED_WIDTH  * 3])
        self.Y = tf.placeholder(tf.float32, shape=[None, 2])
        self.dropoutRate = tf.placeholder(tf.float32)
        logging.info("Model Init")
        self.build()

    def build(self):

        # input placeholder
        
        # input layer
        X_img = tf.reshape(self.X, [-1, RESIZED_HEIGHT, RESIZED_WIDTH, 3])
        
        # conv layer #1
        conv1 = tf.layers.conv2d(inputs=X_img,
        filters=64, kernel_size=[3,3], padding="SAME", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1,
        pool_size=[2,2], padding="SAME", strides=2)
        dropout1= tf.layers.dropout(inputs=pool1,
        rate=self.dropoutRate, training=self.isTraining)


        # conv layer #2
        conv2 = tf.layers.conv2d(inputs=dropout1, 
        filters=128, kernel_size=[3,3], padding="SAME", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2,
        pool_size=[2,2], padding="SAME", strides=2)
        dropout2= tf.layers.dropout(inputs=pool2,
        rate=self.dropoutRate, training=self.isTraining)

        # conv layer #3
        conv3 = tf.layers.conv2d(inputs=dropout2, 
        filters=256, kernel_size=[3,3], padding="SAME", activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3,
        pool_size=[2,2], padding="SAME", strides=2)
        dropout3= tf.layers.dropout(inputs=pool3,
        rate=self.dropoutRate, training=self.isTraining)

        # dense layer
        flat = tf.reshape(dropout3, [-1, 128*128*4])
        dense = tf.layers.dense(inputs=flat, units=700, activation=tf.nn.relu)
        dropout4 = tf.layers.dropout(inputs=dense,
        rate=self.dropoutRate, training=self.isTraining)

        self.logits = tf.layers.dense(inputs=dropout4, units=2)
        y_conv = tf.nn.softmax(self.logits)
        self.prediction = tf.argmax(y_conv,1)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits, labels=self.Y
            )
        )
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learningRate
        ).minimize(self.cost)

        self.correction = tf.equal(self.prediction, tf.argmax(self.Y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correction, tf.float32))


    def train(self, x_train, y_train, dropoutRate, isTraining=True):
        logging.debug(self.sess.run(self.prediction,feed_dict={self.X:x_train, self.isTraining:isTraining, self.dropoutRate:dropoutRate}))
        logging.debug(self.sess.run(tf.argmax(self.Y,1),feed_dict={self.X:x_train, self.Y:y_train, self.isTraining:isTraining, self.dropoutRate:dropoutRate}))
        logging.debug(self.sess.run(self.accuracy,feed_dict={self.X:x_train, self.Y:y_train, self.isTraining:isTraining, self.dropoutRate:dropoutRate}))
        
        return self.sess.run([self.cost, self.optimizer],
        feed_dict={self.X:x_train, self.Y:y_train, self.isTraining:isTraining, self.dropoutRate:dropoutRate}
        )
        

    def predict(self, x_test, isTraining=False):
        return self.sess.run(self.prediction, 
        feed_dict={self.X:x_test, self.isTraining:isTraining, self.dropoutRate:0.0}
        )
        
    def getAccuracy(self, x_test, y_test,isTraining=False):
        
        return self.sess.run(self.accuracy,
        feed_dict={self.X:x_test, self.Y:y_test, self.isTraining:isTraining, self.dropoutRate:0.0}
        )
    