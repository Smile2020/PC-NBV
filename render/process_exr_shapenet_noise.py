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
    data_type = 'test'
    ShapeNetv1_dir = '/home/zengrui/IROS/pcn/data/ShapeNetv1/'
    # class_list_path = '/home/zengrui/IROS/pcn/data/ShapeNetv1/' + data_type + '_class.txt'
    model_dir = '/home/zengrui/IROS/pcn/data/ShapeNetv1/' + data_type
    output_dir = '/home/zengrui/IROS/pcn/PC_results/ShapeNetv1/' + data_type

    intrinsics = np.loadtxt(os.path.join(output_dir, 'intrinsics.txt'))
    width = int(intrinsics[0, 2] * 2)
    height = int(intrinsics[1, 2] * 2)

    class_list = os.listdir(model_dir)

    noise_scale = 0.05

    index = 0

    for class_id in class_list:

        model_list = os.listdir(os.path.join(ShapeNetv1_dir, data_type, class_id))

        for model_id in model_list:
            depth_dir = os.path.join(output_dir, 'depth_%f' % noise_scale, model_id)
            pcd_dir = os.path.join(output_dir, 'pcd_%f' % noise_scale, model_id)
            # if os.path.exists(os.path.join(pcd_dir, '32.pcd')):
            #     print('skip ' + pcd_dir)
            #     continue
            os.makedirs(depth_dir, exist_ok=True)
            os.makedirs(pcd_dir, exist_ok=True)

            for i in range(args.num_scans):
                exr_path = os.path.join(output_dir, 'exr', model_id, '%d.exr' % i)
                pose_path = os.path.join(output_dir, 'pose', model_id, '%d.txt' % i)   

                depth = read_exr(exr_path, height, width)
                flag = depth == 0
                noise = np.random.normal(scale = noise_scale, size=depth.shape)
                depth += noise
                depth[flag] = 0
                # flag = np.random.randint(0, 2, depth.shape)
                # depth = depth * flag

                depth_img = open3d.geometry.Image(np.uint16(depth * 100000))
                open3d.io.write_image(os.path.join(depth_dir, '%d.png' % i), depth_img) 

                pose = np.loadtxt(pose_path)
                points = depth2pcd(depth, intrinsics, pose)
                # points = np.array(points)
                # np.random.shuffle(points)
                # points = points[:int(points.shape[0] * 0.3), :]
                # noise = np.random.normal(scale = 0.0001, size=points.shape)
                # points = points + noise
                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(points)
                open3d.io.write_point_cloud(os.path.join(pcd_dir, '%d.pcd' % i), pcd)

            print (model_id)


