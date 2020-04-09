import os
import pdb
import numpy as np
import random
from infer_nbv_new import infer
from open3d import *
from data_util_nbv import *
import time
import importlib
import tensorflow as tf
from tf_ops.nn_distance import tf_nndistance 

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "4"

    # model_list_path = 'data/treedataset/test.txt'
    # viewspace_path = 'render/viewspace_shapenet_33.txt'
    # result_dir = '/home/zengrui/IROS/pcn/Simulation/Results/ShapeNet_2_12/2_20_Our_0.050/'
    # start_view_dir = '/home/zengrui/IROS/pcn/Simulation/shapenet_start/'
    # PC_cloud_dir = '/home/zengrui/IROS/pcn/PC_results/ShapeNetv1/test/pcd_0.050000'
    # model_list_path = '/home/zengrui/IROS/pcn/Simulation/Results/ShapeNet_2_12/test.txt'

    # viewspace_path = 'render/viewspace_shapenet_33.txt'
    # result_dir = '/home/zengrui/IROS/pcn/Simulation/Results/ABC/Our/'
    # start_view_dir = '/home/zengrui/IROS/pcn/Simulation/shapenet_start/'
    # PC_cloud_dir = '/home/zengrui/IROS/pcn/PC_results/ABC/test_norm/pcd'
    # model_list_path = '/home/zengrui/IROS/pcn/Simulation/Results/ABC/test.txt'

    # viewspace_path = 'render/viewspace_shapenet_33.txt'
    # result_dir = '/home/zengrui/IROS/pcn/Simulation/Results/ABC_50/Our/'
    # start_view_dir = '/home/zengrui/IROS/pcn/Simulation/shapenet_start/'
    # PC_cloud_dir = '/home/zengrui/IROS/pcn/PC_results/ABC/test_norm_1_50/pcd'
    # model_list_path = '/home/zengrui/IROS/pcn/Simulation/Results/ABC_50/test.txt'

    viewspace_path = '/home/zengrui/IROS/pcn/render/viewspace_shapenet_33.txt'
    result_dir = '/home/zengrui/IROS/pcn/Simulation/Results/ABC_patch/Our/'
    start_view_dir = '/home/zengrui/IROS/pcn/Simulation/shapenet_start/'
    PC_cloud_dir = '/home/zengrui/IROS/pcn/PC_results/ABC_patch/test_norm_0.8/pcd'
    model_list_path = '/home/zengrui/IROS/pcn/Simulation/Results/ABC_patch/test.txt'

    ex_num = 1
    scan_num = 10   

    viewspace = np.loadtxt(viewspace_path)
    with open(os.path.join(model_list_path)) as file:
        model_list = [line.strip() for line in file]

    # for calculating surface coverage and register
    part_tensor = tf.placeholder(tf.float32, (1, None, 3))
    gt_tensor = tf.placeholder(tf.float32, (1, None, 3))            
    # sess = tf.Session()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)            

    dist1, _, dist2, _ = tf_nndistance.nn_distance(part_tensor, gt_tensor)

    model_type='pc-nbv'
    checkpoint='/home/zengrui/IROS/pcn/log_shapenet/2_10_3_self/model-290000'
    views=33
    num_gt_points=1024

    is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
    global_step = tf.Variable(0, trainable=False, name='global_step')
    inputs_pl = tf.placeholder(tf.float32, (1, None, 3), 'inputs') # input point cloud
    npts_pl = tf.placeholder(tf.int32, (1,), 'num_points') 
    gt_pl = tf.placeholder(tf.float32, (1, num_gt_points, 3), 'ground_truths') # ground truth
    view_state_pl = tf.placeholder(tf.float32, (1, views), 'view_state') # view space selected state
    eval_value_pl = tf.placeholder(tf.float32, (1, views, 1), 'eval_value') # surface cov, 

    model_module = importlib.import_module('models.pc-nbv')            

    # model = model_module.Model(inputs, npts, gt, view_state_pl, eval_value_pl)          
    model = model_module.Model(inputs_pl, npts_pl, gt_pl, view_state_pl, eval_value_pl, is_training = is_training_pl)

    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)         

    time_count_num = 0
    time_all = 0

    for ex_i in range(ex_num):

        start_view = np.loadtxt(os.path.join(start_view_dir, "test_init_" + str(ex_i) + ".txt"))

        cur_ressult_dir = result_dir + str(ex_i)
        if not os.path.exists(cur_ressult_dir):
            os.makedirs(cur_ressult_dir)

        for i in range(len(model_list)):

            start_time = time.time()
            # model_id = str(int(model_list[i]))
            model_id = model_list[i]
            cur_view = int(start_view[i])
            view_state = np.zeros(viewspace.shape[0], dtype=np.int) 

            scan_pc = np.zeros((1, 3))
            scan_view_list = np.zeros(scan_num)

            for scan_id in range(scan_num):
                time_count_num += 1

                view_state[cur_view] = 2
                scan_view_list[scan_id] = cur_view  

                partial_path = os.path.join(PC_cloud_dir, model_id, str(cur_view) + ".pcd")    

                partial = open3d.io.read_point_cloud(partial_path)
                partial = np.array(partial.points)  

                # remove duplicate
                # batch_part_cur = partial[np.newaxis, :, :]
                # batch_acc = scan_pc[np.newaxis, :, :]
                # dist1_new = sess.run(dist1, feed_dict={part_tensor:batch_part_cur, gt_tensor:batch_acc})
                # dis_flag_new = dist1_new < 0.00005
                # pc_new = batch_part_cur[~dis_flag_new]  

                scan_pc = np.append(scan_pc, partial, axis=0)
                partial = resample_pcd(scan_pc, 1024)

                feed_dict = {inputs_pl: [partial], npts_pl: [partial.shape[0]],view_state_pl: [view_state], 
                     is_training_pl: False}

                # feed_dict = {inputs: [partial], npts: [partial.shape[0]], view_state_pl: [view_state], is_training_pl: False}
                complete, eval_value = sess.run([model.outputs, model.eval_value], feed_dict=feed_dict)
                complete = complete[0]
                eval_value = eval_value[0]          

                new_view = np.argmax(eval_value, axis = 0)

                # new_view = infer(partial=scan_pc, view_state=view_state)    

                view_state[cur_view] = 1
                cur_view = new_view[0]  

            np.savetxt(os.path.join(cur_ressult_dir, model_id + "_selected_views.txt"), scan_view_list)

            end_time = time.time()
            print("model:" + model_id + " spend time:" + str(end_time - start_time))
            time_all += end_time - start_time

    print("ave time per scan:" + str(time_all / time_count_num))



