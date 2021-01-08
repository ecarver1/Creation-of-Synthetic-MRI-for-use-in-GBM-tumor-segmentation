#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import six
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import array_ops
import tensorflow.contrib.layers as layers

def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2, axis=[1,2])


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


def dis (G,reuse,alpha,name):
    with tf.variable_scope(name,reuse=reuse):
        net = slim.conv2d(G,64,5,2,padding='same',activation_fn=None) #2
        net = tf.maximum(alpha*net,net)
            
        net = slim.batch_norm(slim.conv2d(net,128,5,2,padding='same',activation_fn=None)) #4
        net = tf.maximum(alpha*net,net)
        net = slim.batch_norm(slim.conv2d(net,256,5,2,padding='same',activation_fn=None)) #8
        net = tf.maximum(alpha*net,net)
        net = slim.batch_norm(slim.conv2d(net,512,5,2,padding='same',activation_fn=None)) #16
        net = tf.maximum(alpha*net,net)
            
        net = slim.conv2d(net, 1, 3, 1, activation_fn =None) 
        #net = tf.reduce_mean(net,axis = [1,2],keepdims = True)
        #net = tf.squeeze(net, [1,2]) 
        #print(net.shape)
            
        output = tf.nn.sigmoid(net)
    return tf.identity(net,'logits'),tf.identity(output,'output')

def gen (G):
    net = G
    with tf.variable_scope("generator") as scope:
        with slim.arg_scope([slim.conv2d_transpose, slim.batch_norm],activation_fn=None):
            net = tf.layers.dense(net,4*4*512) #7*7
            net = tf.reshape(net,[-1,4,4,512])
            net = tf.nn.relu(slim.batch_norm(net))

            net = tf.nn.relu(slim.batch_norm(slim.conv2d_transpose(net, 256, 5, 2,padding='same')))
            net = tf.nn.relu(slim.batch_norm(slim.conv2d_transpose(net, 128, 5, 2,padding='same')))
            net = tf.nn.relu(slim.batch_norm(slim.conv2d_transpose(net, 64, 5, 2,padding='same')))
            net = slim.conv2d_transpose(net,1, 5, 2,padding='same')

            net = tf.nn.tanh(net)
    return tf.identity(net,'generation')


