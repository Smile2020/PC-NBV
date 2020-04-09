# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import tensorflow as tf
from tf_util import *
from Common import ops

import pdb

class Model:
    def __init__(self, inputs, npts, gt, view_state, eval_value_gt, is_training):
        self.is_training = is_training
        self.num_coarse = 1024
        self.grid_size = 4
        self.grid_scale = 0.05
        self.views = 33
        with tf.variable_scope('pcn', reuse=tf.AUTO_REUSE):
            self.features = self.create_encoder(inputs, npts, view_state)
        
        self.coarse= self.create_decoder(self.features)
        # self.eval_value = self.eval_view(self.features, self.coarse, view_state)
        self.eval_value = self.eval_view(self.features, view_state)
        self.loss, self.update = self.create_loss(self.coarse, gt, self.eval_value, eval_value_gt)
        self.outputs = self.coarse
        self.visualize_ops = [tf.split(inputs[0], npts, axis=0), self.coarse, self.coarse, gt]
        self.visualize_titles = ['input', 'coarse output', 'coarse output', 'ground truth']

    def create_encoder(self, inputs, npts, view_state):
        with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
            features = ops.feature_extraction(inputs, scope='feature_extraction', is_training=self.is_training, bn_decay=None)
            temp = point_maxpool(features, npts, keepdims=True)
            features_global = point_unpool(temp, npts)
            view_state = tf.expand_dims(view_state, 1)
            feature_viewstate = point_unpool(view_state, npts)

            # feature_viewstate = tf.tile(view_state, [1, tf.shape(features)[1], 1])
            features = tf.concat([features, features_global, feature_viewstate], axis=2) #(1, ?, 264 * 2)
            features = tf.reshape(features, (1, tf.shape(features)[1], 1, 561))
            features = ops.attention_unit(features, is_training=self.is_training) # (28, 1024, 1, 128)
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

    # def eval_view(self, feature_part, complete, view_state):
    #     with tf.variable_scope('nbv', reuse=tf.AUTO_REUSE):
    #         with tf.variable_scope('encoder_2', reuse=tf.AUTO_REUSE):
    #             feature_complete = mlp_conv(complete, [128, 256])
    #             feature_global = tf.reduce_max(feature_complete, axis=1, keepdims=True)
    #             feature_global = tf.tile(feature_global, [1, self.num_coarse, 1])
    #             feature_part = tf.expand_dims(feature_part, 1)
    #             feature_part = tf.tile(feature_part, [1, self.num_coarse, 1])
    #             # view_state = tf.expand_dims(view_state, 1)
    #             # feature_viewstate = tf.tile(view_state, [1, self.num_coarse, 1])
    #             feature = tf.concat([feature_complete, feature_global, feature_part], axis=2)
    #         with tf.variable_scope('encoder_3', reuse=tf.AUTO_REUSE):
    #             feature = tf.expand_dims(feature, 2)
    #             feature = ops.attention_unit(feature, is_training=self.is_training)
    #             feature = tf.squeeze(feature, [2])
    #             feature = mlp_conv(feature, [512, 1024])
    #             feat = tf.reduce_max(feature, axis=1, keepdims=False)
    #             feat = tf.concat([feat, view_state], axis=1)
    #             view_eval = mlp(feat, [1024, 1024, self.views * 1])
    #             view_eval = tf.reshape(view_eval, [-1, self.views, 1]) 

    def eval_view(self, feature_part, view_state ):
        with tf.variable_scope('nbv', reuse=tf.AUTO_REUSE):
            # feat = tf.concat([feature_part, view_state], axis=1)
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

        update_coarse = add_valid_summary('valid/coarse_loss', self.loss_eval)
        update_loss = add_valid_summary('valid/loss', self.loss_eval)

        return self.loss_nbv, [update_coarse, update_loss] 

    # def create_loss(self, coarse, gt, eval_value, eval_value_gt):
    #     gt_ds = gt[:, :coarse.shape[1], :]
    #     loss_coarse = earth_mover(coarse, gt_ds)
    #     add_train_summary('train/coarse_loss', loss_coarse)
    #     update_coarse = add_valid_summary('valid/coarse_loss', loss_coarse)

    #     loss_eval = tf.nn.l2_loss(eval_value - eval_value_gt[:, :, :1])
    #     add_train_summary("train/eval_loss", loss_eval)
    #     self.loss_eval = loss_eval

    #     train_vars = tf.trainable_variables(scope='pcn')
    #     lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in train_vars
    #                         if 'bias' not in v.name]) * 0.0001
    #     self.pcn_l2 = lossL2

    #     self.loss_pcn = loss_coarse + lossL2

    #     train_vars_nbv = tf.trainable_variables(scope='nbv')
    #     lossL2_nbv = tf.add_n([ tf.nn.l2_loss(v) for v in train_vars_nbv
    #                         if 'bias' not in v.name]) * 0.0001

    #     self.nbv_ls = lossL2_nbv
    #     self.loss_nbv = loss_eval + lossL2_nbv

    #     add_train_summary('train/loss', self.loss_pcn)
    #     update_loss = add_valid_summary('valid/loss', self.loss_pcn)

    #     return self.loss_pcn, [update_coarse, update_loss]
