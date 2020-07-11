import argparse
import datetime
import importlib
import models
import os
import tensorflow as tf
import time
from data_util_nbv import lmdb_dataflow, get_queued_data
from termcolor import colored
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
    eval_value_pl = tf.placeholder(tf.float32, (args.batch_size, args.views, 1), 'eval_value') # surface coverage

    model_module = importlib.import_module('.%s' % args.model_type, 'models')
    model = model_module.Model(inputs_pl, npts_pl, gt_pl, view_state_pl, eval_value_pl, is_training = is_training_pl)

    if args.lr_decay:
        learning_rate = tf.train.exponential_decay(args.base_lr, global_step,
                                                   args.lr_decay_steps, args.lr_decay_rate,
                                                   staircase=True, name='lr')
        learning_rate = tf.maximum(learning_rate, args.lr_clip)
    else:
        learning_rate = tf.constant(args.base_lr, name='lr')

    trainer = tf.train.AdamOptimizer(learning_rate)
    loss_final = model.loss_nbv

    train_op = trainer.minimize(loss_final, global_step)

    df_train, num_train = lmdb_dataflow(
        args.lmdb_train, args.batch_size, args.num_input_points, args.num_gt_points, is_training=True)
    train_gen = df_train.get_data()
    df_valid, num_valid = lmdb_dataflow(
        args.lmdb_valid, args.batch_size, args.num_input_points, args.num_gt_points, is_training=False)
    valid_gen = df_valid.get_data()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver(max_to_keep=100)

    sess.run(tf.global_variables_initializer())
    if os.path.exists(args.log_dir):
        delete_key = input(colored('%s exists. Delete? [y (or enter)/N]'
                                   % args.log_dir, 'white', 'on_red'))
        if delete_key == 'y' or delete_key == "":
            os.system('rm -rf %s/*' % args.log_dir)
            os.makedirs(os.path.join(args.log_dir, 'plots'))
    else:
        os.makedirs(os.path.join(args.log_dir, 'plots'))
    with open(os.path.join(args.log_dir, 'args.txt'), 'w') as log:
        for arg in sorted(vars(args)):
            log.write(arg + ': ' + str(getattr(args, arg)) + '\n')     # log of arguments
    os.system('cp models/%s.py %s' % (args.model_type, args.log_dir))  # backup of model definition

    csv_file = open(os.path.join(args.log_dir, 'loss.csv'), 'a+')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['step', 'train loss', 'train loss_eval', 'train spearmanr', 
                                 'valid loss', 'valid loss_eval', 'valid spearmanr'])

    total_time = 0
    train_start = time.time()
    init_step = sess.run(global_step)
    for step in range(init_step + 1, args.max_step + 1):
        epoch = step * args.batch_size // num_train + 1
        ids, inputs, npts, gt, view_state, eval_value = next(train_gen)
        start = time.time()
        feed_dict = {inputs_pl: inputs, npts_pl: npts, gt_pl: gt, view_state_pl:view_state, 
            eval_value_pl:eval_value[:, :, :1], is_training_pl: True}
        _, loss, loss_eval, eval_value_pre = sess.run([train_op, model.loss, model.loss_eval, model.eval_value], feed_dict=feed_dict)
        total_time += time.time() - start
        if step % args.steps_per_print == 0:

            spearmanr_total = 0
            for j in range(args.batch_size):
                spearmanr_total += stats.spearmanr(eval_value[j, :, 0], eval_value_pre[j, :, 0])[0]
            spearmanr = spearmanr_total / args.batch_size

            print('epoch %d  step %d  loss %.8f loss_eval %.8f spearmanr %.8f - time per batch %.4f' %
                  (epoch, step, loss, loss_eval, spearmanr, total_time / args.steps_per_print))
            total_time = 0
        if step % args.steps_per_eval == 0:
            print(colored('Testing...', 'grey', 'on_green'))
            num_eval_steps = num_valid // args.batch_size
            valid_total_loss = 0
            valid_total_loss_eval = 0
            valid_total_time = 0
            valid_total_spearmanr = 0
            sess.run(tf.local_variables_initializer())
            for i in range(num_eval_steps):
                start = time.time()
                ids, inputs, npts, gt, view_state, eval_value = next(valid_gen)

                feed_dict = {inputs_pl: inputs, npts_pl: npts, gt_pl: gt, view_state_pl:view_state, 
                    eval_value_pl:eval_value[:, :, :1], is_training_pl: False}
                valid_loss, valid_loss_eval, valid_eval_value_pre = sess.run([model.loss, model.loss_eval, model.eval_value], feed_dict=feed_dict)
            
                valid_spearmanr_batch_total = 0
                for j in range(args.batch_size):
                    valid_spearmanr_batch_total += stats.spearmanr(eval_value[j, :, 0], valid_eval_value_pre[j, :, 0])[0]
                valid_spearmanr = valid_spearmanr_batch_total / args.batch_size

                valid_total_loss += valid_loss
                valid_total_loss_eval += valid_loss_eval
                valid_total_spearmanr += valid_spearmanr
                valid_total_time += time.time() - start

            print(colored('epoch %d  step %d  loss %.8f loss_eval %.8f spearmanr %.8f - time per batch %.4f' %
                          (epoch, step, valid_total_loss / num_eval_steps, valid_total_loss_eval / num_eval_steps,
                             valid_total_spearmanr / num_eval_steps, valid_total_time / num_eval_steps),
                          'grey', 'on_green'))

            csv_writer.writerow([step, loss, loss_eval, spearmanr, 
                valid_total_loss / num_eval_steps, valid_total_loss_eval / num_eval_steps, valid_total_spearmanr / num_eval_steps,])

            valid_total_time = 0

        if step % args.steps_per_save == 0:
            saver.save(sess, os.path.join(args.log_dir, 'model'), step)
            print(colored('Model saved at %s' % args.log_dir, 'white', 'on_blue'))

    print('Total time', datetime.timedelta(seconds=time.time() - train_start))
    sess.close()
    csv_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_train', default='/home/zengrui/IROS/pcn/NBV_data/shapenet_33_views/train.lmdb')
    parser.add_argument('--lmdb_valid', default='/home/zengrui/IROS/pcn/NBV_data/shapenet_33_views/valid.lmdb')
    parser.add_argument('--log_dir', default='log/7_11_1')
    parser.add_argument('--model_type', default='pc-nbv')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_input_points', type=int, default=512)
    parser.add_argument('--num_gt_points', type=int, default=1024)
    parser.add_argument('--views', type=int, default=33)
    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--lr_decay', action='store_true')
    parser.add_argument('--lr_decay_steps', type=int, default=50000)
    parser.add_argument('--lr_decay_rate', type=float, default=0.7)
    parser.add_argument('--lr_clip', type=float, default=1e-6)
    parser.add_argument('--max_step', type=int, default=400000)
    parser.add_argument('--steps_per_print', type=int, default=100)
    parser.add_argument('--steps_per_eval', type=int, default=1000)
    parser.add_argument('--steps_per_save', type=int, default=5000)
    parser.add_argument('--gpu', default='2')

    args = parser.parse_args()

    train(args)