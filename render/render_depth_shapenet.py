# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018
# Modified by Rui Zeng 07/12/2020

import bpy
import mathutils
import numpy as np
import os
import sys
import time
import pdb
import argparse
# Usage: blender -b -P render_depth.py


def setup_blender(width, height, focal_length, output_dir):
    # camera
    camera = bpy.data.objects['Camera']
    camera.data.angle = np.arctan(width / 2 / focal_length) * 2
    # camera.data.clip_end = 1.2
    # camera.data.clip_start = 0.2

    # render layer
    scene = bpy.context.scene
    scene.render.filepath = 'buffer'
    scene.render.image_settings.color_depth = '16'
    scene.render.resolution_percentage = 100
    scene.render.resolution_x = width
    scene.render.resolution_y = height

    # compositor nodes
    scene.use_nodes = True
    tree = scene.node_tree
    rl = tree.nodes.new('CompositorNodeRLayers')
    output = tree.nodes.new('CompositorNodeOutputFile')
    output.base_path = ''
    output.format.file_format = 'OPEN_EXR'
    tree.links.new(rl.outputs['Depth'], output.inputs[0])

    # remove default cube
    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete()

    return scene, camera, output


if __name__ == '__main__':

    viewspace_path = 'viewspace_shapenet_33.txt'

    data_type = 'test_norm_0.8'
    ShapeNetv1_dir = '/home/zengrui/IROS/pcn/data/ABC/patch/10k'
    model_dir = '/home/zengrui/IROS/pcn/data/ABC/patch/10k/' + data_type
    output_dir = '/home/zengrui/IROS/pcn/PC_results/ABC_patch/' + data_type

    viewspace = np.loadtxt(viewspace_path)

    width = 640
    height = 480
    focal = 238 * 2

    scene, camera, output = setup_blender(width, height, focal, output_dir)
    intrinsics = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])

    open('blender.log', 'w+').close()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.savetxt(os.path.join(output_dir, 'intrinsics.txt'), intrinsics, '%f')

    class_list = os.listdir(model_dir)

    for class_id in class_list:
        model_list = os.listdir(os.path.join(ShapeNetv1_dir, data_type, class_id))

        for model_id in model_list:
            start = time.time()
            exr_dir = os.path.join(output_dir, 'exr', model_id)
            pose_dir = os.path.join(output_dir, 'pose', model_id)
            if os.path.exists(os.path.join(exr_dir, '32.exr')):
                print("skip " + exr_dir)
                continue

            os.makedirs(exr_dir, exist_ok=True)
            os.makedirs(pose_dir, exist_ok=True)   

            # Redirect output to log file
            old_os_out = os.dup(1)
            os.close(1)
            os.open('blender.log', os.O_WRONLY) 

            # Import mesh model
            model_path = os.path.join(ShapeNetv1_dir, data_type, class_id, model_id, 'model.obj')
            bpy.ops.import_scene.obj(filepath=model_path)   

            # Rotate model by 90 degrees around x-axis (z-up => y-up) to match ShapeNet's coordinates
            bpy.ops.transform.rotate(value=-np.pi / 2, axis=(1, 0, 0))  

            # Render
            for i in range(viewspace.shape[0]):
                scene.frame_set(i)
                cam_pose = mathutils.Vector((viewspace[i][0], viewspace[i][1], viewspace[i][2]))
                center_pose = mathutils.Vector((0, 0, 0))
                direct = center_pose - cam_pose
                rot_quat = direct.to_track_quat('-Z', 'Y')
                camera.rotation_euler = rot_quat.to_euler()
                camera.location = cam_pose
                pose_matrix = camera.matrix_world
                output.file_slots[0].path = os.path.join(exr_dir, '#.exr')
                bpy.ops.render.render(write_still=True)
                np.savetxt(os.path.join(pose_dir, '%d.txt' % i), pose_matrix, '%f') 

            # Clean up
            bpy.ops.object.delete()
            for m in bpy.data.meshes:
                bpy.data.meshes.remove(m)
            for m in bpy.data.materials:
                m.user_clear()
                bpy.data.materials.remove(m)    

            # Show time
            os.close(1)
            os.dup(old_os_out)
            os.close(old_os_out)
            print('%s done, time=%.4f sec' % (model_id, time.time() - start))