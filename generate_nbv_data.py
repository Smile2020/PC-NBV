import os
import numpy as np
import tensorflow as tf
import scipy.io as sio
from open3d import *
import random
from tf_ops.nn_distance import tf_nndistance 
import time
import pdb

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    
    # view num
    view_num = 33

    # path
    data_type = 'train'
    ShapeNetv1_dir = '/home/zengrui/IROS/pcn/data/ShapeNetv1/'    
    pc_dir = "/home/zengrui/IROS/pcn/PC_results/ShapeNetv1/" + data_type + "/pcd"
    save_dir = "/home/zengrui/IROS/pcn/NBV_data/shapenet_33_views_640x480"
    model_dir = '/home/zengrui/IROS/pcn/data/ShapeNetv1/' + data_type

    # for calculating surface coverage and register
    part_tensor = tf.placeholder(tf.float32, (1, None, 3))
    gt_tensor = tf.placeholder(tf.float32, (1, None, 3))
    sess = tf.Session()
    dist1, _, dist2, _ = tf_nndistance.nn_distance(part_tensor, gt_tensor)

    class_list = os.listdir(model_dir)

    for class_id in class_list:

        model_list = os.listdir(os.path.join(ShapeNetv1_dir, data_type, class_id))

        for model in model_list:
            save_model_path = os.path.join(save_dir, model)
            if os.path.exists(save_model_path):
                print("skip " + save_model_path)
                continue

            # gt point cloud
            gt_points = sio.loadmat(os.path.join(ShapeNetv1_dir, data_type, class_id, model, 'model.mat'))
            gt_points = np.array(gt_points['points'])
            if not os.path.exists(os.path.join(save_dir, model)):
                os.makedirs(os.path.join(save_dir, model))

            np.savetxt(os.path.join(save_dir, model, "gt.xyz"), gt_points)    

            # every view's partial point cloud
            part_points_list = []
            
            for i in range(view_num):
                pcd_path = os.path.join(pc_dir, model, str(i) + ".pcd")
                if os.path.exists(pcd_path):
                    cur_pc = open3d.io.read_point_cloud(pcd_path)
                    cur_points = np.asarray(cur_pc.points)  
                else:
                    cur_points = np.zeros((1,3))

                part_points_list.append(cur_points)

            # reconstruct from different views 1 times
            selected_init_view = []
            for ex_index in range(1):   

                start = time.time() 

                cur_ex_dir = os.path.join(save_dir, model, str(ex_index))
                if not os.path.exists(cur_ex_dir):
                    os.makedirs(cur_ex_dir) 

                # init view state
                view_state = np.zeros(view_num, dtype=np.int) # 0 unselected, 1 selected, 2 cur
                # init start view
                while (True):
                    cur_view = random.randint(0, view_num - 1)
                    if not cur_view in selected_init_view:
                        selected_init_view.append(cur_view)
                        break   

                view_state[cur_view] = 1

                acc_pc_points = part_points_list[cur_view]  

                # accumulate points coverage
                batch_acc = acc_pc_points[np.newaxis, :, :]
                batch_gt = gt_points[np.newaxis, :, :]

                dist2_new = sess.run(dist2, feed_dict={part_tensor:batch_acc, gt_tensor:batch_gt})      
                dis_flag_new = dist2_new < 0.00005
                cover_sum = np.sum(dis_flag_new == True)
                cur_cov = cover_sum / dis_flag_new.shape[1]

                # max scan 10 times
                for scan_index in range(10):    

                    print("coverage:" + str(cur_cov) + " in scan round " + str(scan_index)) 

                    np.save(os.path.join(cur_ex_dir, str(scan_index) + "_viewstate.npy") ,view_state)
                    np.save(os.path.join(cur_ex_dir, str(scan_index) + "_acc_pc.npy"), acc_pc_points)
                    # np.savetxt(os.path.join(cur_ex_dir, str(scan_index) + "_acc_pc.xyz"), acc_pc_points)    

                    target_value = np.zeros((view_num, 1)) # surface coverage, register coverage, moving cost for each view         

                    max_view_index = 0
                    max_view_cov = 0
                    max_new_pc = np.zeros((1,3))

                    # # accumulate points coverage
                    batch_acc = acc_pc_points[np.newaxis, :, :]
                    batch_gt = gt_points[np.newaxis, :, :]
                    
                    # evaluate all the views
                    for i in range(view_num):   

                        # current evaluate view
                        batch_part_cur = part_points_list[i][np.newaxis, :, :]  

                        # new pc
                        dist1_new = sess.run(dist1, feed_dict={part_tensor:batch_part_cur, gt_tensor:batch_acc})
                        dis_flag_new = dist1_new < 0.00005  

                        pc_register = batch_part_cur[dis_flag_new]
                        pc_new = batch_part_cur[~dis_flag_new]

                        batch_new = pc_new[np.newaxis, :, :]    

                        # test new coverage
                        if batch_new.shape[1] != 0:
                            dist2_new = sess.run(dist2, feed_dict={part_tensor:batch_new, gt_tensor:batch_gt})      

                            dis_flag_new = dist2_new < 0.00005
                            cover_sum = np.sum(dis_flag_new == True)
                            cover_new = cover_sum / dis_flag_new.shape[1]
                        else:
                            cover_new = 0   

                        target_value[i, 0] = cover_new

                        if ( target_value[i, 0] > max_view_cov ):
                            max_view_index = i
                            max_view_cov = target_value[i, 0]
                            max_new_pc = pc_new 

                    np.save(os.path.join(cur_ex_dir, str(scan_index) + "_target_value.npy"), target_value)  

                    print("choose view:" + str(max_view_index) + " add coverage:" + str(max_view_cov))  

                    cur_view = max_view_index
                    cur_cov = cur_cov + target_value[max_view_index, 0]
                    view_state[cur_view] = 1
                    acc_pc_points = np.append(acc_pc_points, max_new_pc, axis=0)    

                    print('scan %s done, time=%.4f sec' % (scan_index, time.time() - start))


