# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import argparse
import importlib
import models
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from open3d import *
import os
import pdb
from scipy import stats

def plot_pcd(ax, pcd):
    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir='y', c=pcd[:, 0], s=0.5, cmap='Reds', vmin=-1, vmax=0.5)
    ax.set_axis_off()
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(-0.3, 0.3)


def infer(partial, view_state, model_type='pcn_nbv_regular', checkpoint='/home/zengrui/IROS/pcn/log/1_15_1_33views/model-300000', views=33, num_gt_points=1024):

    inputs = tf.placeholder(tf.float32, (1, None, 3))
    gt = tf.placeholder(tf.float32, (1, num_gt_points, 3))
    npts = tf.placeholder(tf.int32, (1,))
    view_state_pl = tf.placeholder(tf.float32, (1, views), 'view_state') # view space selected state
    eval_value_pl = tf.placeholder(tf.float32, (1, views, 3), 'eval_value') # surface cov, register cov, moving cost
    model_module = importlib.import_module('log.1_15_1_33views.%s' % model_type)

    model = model_module.Model(inputs, npts, gt, view_state_pl, eval_value_pl)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)

    feed_dict = {inputs: [partial], npts: [partial.shape[0]], view_state_pl: [view_state]}
    complete, eval_value = sess.run([model.outputs, model.eval_value], feed_dict=feed_dict)
    complete = complete[0]
    eval_value = eval_value[0]

    sess.close()

    return np.argmax(eval_value, axis = 0)
