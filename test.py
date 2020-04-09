import argparse
import datetime
import importlib
import models
import os
import tensorflow as tf
import time
from data_util_nbv import lmdb_dataflow, get_queued_data
from termcolor import colored
from tf_util import add_train_summary
from visu_util import plot_pcd_three_views
import pdb
from tensorpack import dataflow
from scipy import stats
import csv


def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
    global_step = tf.Variable(0, trainable=False, name='global_step')
    inputs_pl = tf.placeholder(tf.float32, (1, None, 3), 'inputs') # input point cloud
    npts_pl = tf.placeholder(tf.int32, (args.batch_size,), 'num_points') 
    gt_pl = tf.placeholder(tf.float32, (args.batch_size, args.num_gt_points, 3), 'ground_truths') # ground truth
    view_state_pl = tf.placeholder(tf.float32, (args.batch_size, args.views), 'view_state') # view space selected state
    eval_value_pl = tf.placeholder(tf.float32, (args.batch_size, args.views, 1), 'eval_value') # surface cov, 

    model_module = importlib.import_module('.%s' % args.model_type, 'models')
    model = model_module.Model(inputs_pl, npts_pl, gt_pl, view_state_pl, eval_value_pl, is_training = is_training_pl)

    train_summary = tf.summary.merge_all('train_summary')
    valid_summary = tf.summary.merge_all('valid_summary')

    df_valid, num_valid = lmdb_dataflow(
        args.lmdb_valid, args.batch_size, args.num_input_points, args.num_gt_points, is_training=False)
    valid_gen = df_valid.get_data()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver(max_to_keep=100)

    saver.restore(sess, args.checkpoint)

    total_time = 0
    train_start = time.time()

    print(colored('Testing...', 'grey', 'on_green'))
    num_eval_steps = num_valid // args.batch_size
    # num_eval_steps = 100
    valid_total_loss = 0
    valid_total_loss_eval = 0
    valid_total_time = 0
    valid_total_spearmanr = 0
    sess.run(tf.local_variables_initializer())
    for i in range(num_eval_steps):
        print('step ' + str(i))
        
        ids, inputs, npts, gt, view_state, eval_value = next(valid_gen)
        feed_dict = {inputs_pl: inputs, npts_pl: npts, gt_pl: gt, view_state_pl:view_state, 
            eval_value_pl:eval_value[:, :, :1], is_training_pl: False}
        start = time.time()
        valid_loss, valid_loss_eval, valid_eval_value_pre = sess.run([model.loss, model.loss_eval, model.eval_value], feed_dict=feed_dict)
        valid_total_time += time.time() - start
        valid_spearmanr_batch_total = 0
        for j in range(args.batch_size):
            valid_spearmanr_batch_total += stats.spearmanr(eval_value[j, :, 0], valid_eval_value_pre[j, :, 0])[0]
        valid_spearmanr = valid_spearmanr_batch_total / args.batch_size
        valid_total_loss += valid_loss
        valid_total_loss_eval += valid_loss_eval
        valid_total_spearmanr += valid_spearmanr
        
    summary = sess.run(valid_summary, feed_dict={is_training_pl: False})
    print(colored('loss %.8f loss_eval %.8f spearmanr %.8f - time per batch %.4f' %
                  (valid_total_loss / num_eval_steps, valid_total_loss_eval / num_eval_steps,
                     valid_total_spearmanr / num_eval_steps, valid_total_time / num_eval_steps),
                  'grey', 'on_green'))
    valid_total_time = 0

    print('Total time', datetime.timedelta(seconds=time.time() - train_start))
    sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_valid', default='/home/zengrui/IROS/pcn/NBV_data/shapenet_33_views/test.lmdb')
    parser.add_argument('--model_type', default='pc-nbv')
    parser.add_argument('--checkpoint', default='/home/zengrui/IROS/pcn/log_shapenet/2_10_3_self/model-290000')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_input_points', type=int, default=512)
    parser.add_argument('--num_gt_points', type=int, default=1024)
    parser.add_argument('--views', type=int, default=33)
    parser.add_argument('--gpu', default='')

    args = parser.parse_args()

    train(args)