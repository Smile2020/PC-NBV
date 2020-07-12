import argparse
import os
from io_util import read_pcd
from tensorpack import DataFlow, dataflow
import numpy as np
import scipy.io as sio
import pdb


class pcd_df(DataFlow):
    def __init__(self, class_list, ex_times, num_scans, NBV_dir, gt_dir, data_type):
        self.class_list = class_list
        self.num_scans = num_scans
        self.ex_times = ex_times
        self.NBV_dir = NBV_dir
        self.gt_dir = gt_dir
        self.data_type = data_type

    def size(self):
        if self.data_type == 'valid':
            return 4000
        elif self.data_type == 'train':
            return 40000
        elif self.data_type == 'test':
            return 4000
        elif self.data_type == 'test_novel':
            return 4000

    def get_data(self):
        ShapeNetv1_dir = '/home/zengrui/IROS/pcn/data/ShapeNetv1/'
        for class_id in self.class_list:
            model_list = os.listdir(os.path.join(ShapeNetv1_dir, self.data_type, class_id))
            for model_id in model_list:
                gt_points = sio.loadmat(os.path.join(self.gt_dir, class_id, model_id, 'model.mat'))
                gt_pc = np.array(gt_points['points']) # shape (16384, 3)
                for ex_index in range(self.ex_times):
                    for scan_index in range(self.num_scans):
                        view_state = np.load(os.path.join(self.NBV_dir, str(model_id), str(ex_index), str(scan_index) + "_viewstate.npy")) # shape (33) , 33 is view number
                        accumulate_pointcloud = np.load(os.path.join(self.NBV_dir, str(model_id), str(ex_index), str(scan_index) + "_acc_pc.npy")) # shape (point number, 3)
                        target_value = np.load(os.path.join(self.NBV_dir, str(model_id), str(ex_index), str(scan_index) + "_target_value.npy")) # shape (33, 1), 33 is view number
                        yield model_id, accumulate_pointcloud, gt_pc, view_state, target_value

if __name__ == '__main__':

    data_type = 'valid'
    class_list_path = '/home/zengrui/IROS/pcn/data/ShapeNetv1/' + data_type + '_class.txt'
    gt_dir = "/home/zengrui/IROS/pcn/data/ShapeNetv1/" + data_type
    output_path = "data/" + data_type + ".lmdb"
    NBV_dir = "/home/zengrui/IROS/pcn/NBV_data/shapenet_33_views"
    ex_times = 1
    num_scans = 10

    with open(os.path.join(class_list_path)) as file:
        class_list = [line.strip() for line in file]

    df = pcd_df(class_list, ex_times, num_scans, NBV_dir, gt_dir, data_type)
    if os.path.exists(output_path):
        os.system('rm %s' % output_path)

    dataflow.LMDBSerializer.save(df, output_path)
