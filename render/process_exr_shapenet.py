# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import Imath
import OpenEXR
import argparse
import array
import numpy as np
import os
from open3d import *
# from open3d.io import *
import matplotlib.pyplot as plt
import sys
import pdb

def read_exr(exr_path, height, width):
    file = OpenEXR.InputFile(exr_path)
    depth_arr = array.array('f', file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT)))
    depth = np.array(depth_arr).reshape((height, width))
    depth[depth < 0] = 0
    depth[np.isinf(depth)] = 0
    return depth


def depth2pcd(depth, intrinsics, pose):
    inv_K = np.linalg.inv(intrinsics)
    inv_K[2, 2] = -1
    depth = np.flipud(depth)
    y, x = np.where(depth > 0)
    # image coordinates -> camera coordinates
    points = np.dot(inv_K, np.stack([x, y, np.ones_like(x)] * depth[y, x], 0))
    # camera coordinates -> world coordinates
    points = np.dot(pose, np.concatenate([points, np.ones((1, points.shape[1]))], 0)).T[:, :3]
    return points


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--list_file', default='/home/zengrui/IROS/pcn/data/treedataset/test.txt')
    # parser.add_argument('--output_dir', default='/home/zengrui/IROS/pcn/PC_results/ShapeNetv1/valid')
    parser.add_argument('--num_scans', type=int, default=33)
    args = parser.parse_args()
    # data_type = 'valid'
    # ShapeNetv1_dir = '/home/zengrui/IROS/pcn/data/ShapeNetv1/'
    # model_dir = '/home/zengrui/IROS/pcn/data/ShapeNetv1/' + data_type
    # output_dir = '/home/zengrui/IROS/pcn/PC_results/ShapeNetv1/' + data_type

    # data_type = 'model'
    # ShapeNetv1_dir = '/home/zengrui/IROS/pcn/data/single'
    # model_dir = '/home/zengrui/IROS/pcn/data/single/' + data_type
    # output_dir = '/home/zengrui/IROS/pcn/PC_results/single/' + data_type

    # data_type = 'test_norm_1_50'
    # ShapeNetv1_dir = '/home/zengrui/IROS/pcn/data/ABC/10k'
    # model_dir = '/home/zengrui/IROS/pcn/data/ABC/10k/' + data_type
    # output_dir = '/home/zengrui/IROS/pcn/PC_results/ABC/' + data_type

    data_type = 'test_norm_0.8'
    ShapeNetv1_dir = '/home/zengrui/IROS/pcn/data/ABC/patch/10k'
    model_dir = '/home/zengrui/IROS/pcn/data/ABC/patch/10k/' + data_type
    output_dir = '/home/zengrui/IROS/pcn/PC_results/ABC_patch/' + data_type

    intrinsics = np.loadtxt(os.path.join(output_dir, 'intrinsics.txt'))
    width = int(intrinsics[0, 2] * 2)
    height = int(intrinsics[1, 2] * 2)

    # with open(os.path.join(class_list_path)) as file:
    #     class_list = [line.strip() for line in file]

    # for class_id in class_list:
    #     model_list_path = os.path.join(ShapeNetv1_dir, data_type, class_id, 'model_list.txt')
    #     with open(os.path.join(model_list_path)) as file:
    #         model_list = [line.strip() for line in file]

    #     for model_id in model_list:
    class_list = os.listdir(model_dir)

    for class_id in class_list:

        model_list = os.listdir(os.path.join(ShapeNetv1_dir, data_type, class_id))

        for model_id in model_list:
            depth_dir = os.path.join(output_dir, 'depth', model_id)
            pcd_dir = os.path.join(output_dir, 'pcd', model_id)
            pcd_list = os.listdir(pcd_dir)
            if (len(pcd_list) == 33):
                print('skip ' + pcd_dir)
                continue
            # if os.path.exists(os.path.join(depth_dir, '32.png')):
            #     print('skip ' + depth_dir)
            #     continue
            os.makedirs(depth_dir, exist_ok=True)
            os.makedirs(pcd_dir, exist_ok=True)
            for i in range(args.num_scans):
                exr_path = os.path.join(output_dir, 'exr', model_id, '%d.exr' % i)
                pose_path = os.path.join(output_dir, 'pose', model_id, '%d.txt' % i)   

                depth = read_exr(exr_path, height, width)
                depth_img = open3d.geometry.Image(np.uint16(depth * 100000))
                open3d.io.write_image(os.path.join(depth_dir, '%d.png' % i), depth_img) 

                pose = np.loadtxt(pose_path)
                points = depth2pcd(depth, intrinsics, pose)
                if (points.shape[0] == 0):
                    points = np.array([(1.0,1.0,1.0)])
                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(points)
                open3d.io.write_point_cloud(os.path.join(pcd_dir, '%d.pcd' % i), pcd)
