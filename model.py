# maching learning model for training, predicting, evaluating

from common import *
import tensorflow as tf
import numpy as np

class Model:


    def __init__(self, sess):
        self.sess = sess
        self.learning_rate = 0.003

    def build(self):
        # set basic tensor
        self.isTraining = tf.placeholder(tf.bool)
        self.X = tf.placeholder(tf.float32, shape=[None,  RESIZED_HEIGHT * RESIZED_WIDTH  * 3])
        self.Y = tf.placeholder(tf.float32, shape=[None, 2])
        X_img = tf.reshape(self.X, [-1, RESIZED_HEIGHT, RESIZED_WIDTH, 3])
        
        with tf.name_scope("layer#1"):
            conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], padding="SAME", strides=2)
            dropout1 = tf.layers.dropout(inputs=pool1, rate=0.5, training=self.isTraining)

        with tf.name_scope("layer#2"):
            conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], padding="SAME", strides=2)
            dropout2 = tf.layers.dropout(inputs=pool2, rate=0.5, training=self.isTraining)

        with tf.name_scope("layer#3"):
            conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], padding="SAME", strides=2)
            dropout3 = tf.layers.dropout(inputs=pool3, rate=0.5, training=self.isTraining)

        with tf.name_scope("layer#4"):
            flat = tf.reshape(dropout3, [-1, 128 * 4 * 4])
            dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
            dropout4 = tf.layers.dropout(inputs=dense4, rate=0.5, training=self.isTraining)

        with tf.name_scope("output"):
            self.logits = tf.layers.dense(inputs=dropout4, units=2)
        

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.cost)

        correction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correction, tf.float32))


    
    def train(self, x_data, y_data, isTraining=True):
        return self.sess.run([self.cost, self.optimizer], 
        feed_dict={self.X: x_data, self.Y: y_data, self.isTraining: isTraining})

    def predict(self, x_data, isTraining=False):
        return self.sess.run(self.logits,
        feed_dict={self.X: x_data, self.isTraining: isTraining})

    def getAccuracy(self):
        pass

    