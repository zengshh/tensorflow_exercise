import tensorflow as tf
import numpy as np
import os

MODEL_PATH = "model/"
MODEL_NAME = "model.ckpt"

x = tf.Variable(tf.truncated_normal([2, 5]), 'x-input')
x1 = tf.Variable(tf.truncated_normal([2, 2]), 'x1-input')
with tf.variable_scope('layers') as scope:
    w = tf.get_variable('weights', [2, 3, 5, 3],
                        initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('biase', [1],
                        initializer=tf.constant_initializer(1.0))

#y = tf.matmul(x, w) + b     #add operation used broadcast.

init_var = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_var)
    saver.save(sess, MODEL_PATH+MODEL_NAME)
    #print("The result of y is\n ", y.eval())
    print ('weight :\n ', w.eval() , '\n biase : \n', b.eval())

