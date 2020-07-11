# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018
# Modified by Rui Zeng 07/11/2020
import tensorflow as tf
from tf_util import *
from Common import ops

import pdb

class Model:
    def __init__(self, inputs, npts, gt, view_state, eval_value_gt, is_training):
        self.is_training = is_training
        self.num_coarse = 1024
        self.views = 33
        with tf.variable_scope('pcn', reuse=tf.AUTO_REUSE):
            self.features = self.create_encoder(inputs, npts, view_state)
        
        self.coarse= self.create_decoder(self.features)
        self.eval_value = self.eval_view(self.features, view_state)
        self.loss = self.create_loss(self.coarse, gt, self.eval_value, eval_value_gt)

    def create_encoder(self, inputs, npts, view_state):
        with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
            features = ops.feature_extraction(inputs, scope='feature_extraction', is_training=self.is_training, bn_decay=None)
            temp = point_maxpool(features, npts, keepdims=True)
            features_global = point_unpool(temp, npts)
            view_state = tf.expand_dims(view_state, 1)
            feature_viewstate = point_unpool(view_state, npts)

            features = tf.concat([features, features_global, feature_viewstate], axis=2)
            features = tf.reshape(features, (1, tf.shape(features)[1], 1, 561))
            features = ops.attention_unit(features, is_training=self.is_training)
            features = tf.squeeze(features, [2])

        with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE):
            features = mlp_conv(features, [1024, 1024])
            features = point_maxpool(features, npts)
        return features

    def create_decoder(self, features):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            coarse = mlp(features, [1024, 1024, self.num_coarse * 3])
            coarse = tf.reshape(coarse, [-1, self.num_coarse, 3])
        return coarse

    def eval_view(self, feature_part, view_state ):
        with tf.variable_scope('nbv', reuse=tf.AUTO_REUSE):
            view_eval = mlp(feature_part, [1024, 512, 256, self.views * 1])
            view_eval = tf.reshape(view_eval, [-1, self.views, 1]) 
            return view_eval

    def create_loss(self, coarse, gt, eval_value, eval_value_gt):
        train_vars = tf.trainable_variables(scope='pcn')
        lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in train_vars
                            if 'bias' not in v.name]) * 0.0001

        train_vars_nbv = tf.trainable_variables(scope='nbv')
        lossL2_nbv = tf.add_n([ tf.nn.l2_loss(v) for v in train_vars_nbv
                            if 'bias' not in v.name]) * 0.0001

        self.loss_eval = tf.nn.l2_loss(eval_value - eval_value_gt[:, :, :1])
        add_train_summary("train/eval_loss", self.loss_eval)
        self.loss_nbv = self.loss_eval + lossL2 + lossL2_nbv

        return self.loss_nbv
