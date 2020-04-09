# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018
# Modified by Chenming Wu (wucm@uw.edu) 01/13/2019

import numpy as np
import os
import sys
import time
import pdb

# Usage: blender -b -P render_depth.py [ShapeNet directory] [model list] [output directory] [num scans per model]


def random_pose():
    angle_x = np.random.uniform() * 2 * np.pi
    angle_y = np.random.uniform() * 2 * np.pi
    angle_z = np.random.uniform() * 2 * np.pi
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(angle_x), -np.sin(angle_x)],
                    [0, np.sin(angle_x), np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                    [0, 1, 0],
                    [-np.sin(angle_y), 0, np.cos(angle_y)]])
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                    [np.sin(angle_z), np.cos(angle_z), 0],
                    [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    # Set camera pointing to the origin and 1 unit away from the origin
    t = np.expand_dims(R[:, 2], 1)
    pose = np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)
    return t, pose

def cal_pose(angle_x, angle_y, angle_z):
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(angle_x), -np.sin(angle_x)],
                    [0, np.sin(angle_x), np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                    [0, 1, 0],
                    [-np.sin(angle_y), 0, np.cos(angle_y)]])
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                    [np.sin(angle_z), np.cos(angle_z), 0],
                    [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))

    # Set camera pointing to the origin(0, 1, 0) and 1.5 unit away from the origin
    t = np.expand_dims(R[:, 2], 1) * 1.25
    pose = np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)
    return t, pose

if __name__ == '__main__': 
    
    t_list = []
    # for angle_x in [0, np.pi/2, np.pi, np.pi * 3/2]:
    #     for angle_y in [0, np.pi/2, np.pi, np.pi * 3/2]:
    #         for angle_z in [0, np.pi/2, np.pi, np.pi * 3/2]:
    #             t, _ = cal_pose(angle_x, angle_y, angle_z)
    
    # for i in range(48):
    #     t, _ = random_pose()
    #     t_list.append(t);

    for i in range(-90, 45, 30):
        angle_x = i * np.pi / 180
        # theta_y = (45 - i) * 2
        for j in range(0, 360, 45):
            angle_y = j * np.pi / 180
            t, _ = cal_pose(angle_x, angle_y, 0)
            t_list.append(t);

    viewspace = np.hstack(t_list).T
    np.savetxt("viewspace_shapenet_large.txt", viewspace)
