"""
this file will show kitti lidar point cloud data
in sequence continuous
"""
import numpy as np
import open3d
from open3d import *
import os
import glob
from mayavi import mlab
import pcl
import vtk


kitti_seq_dir = '/media/jintain/sg/permanent/datasets/KITTI/videos/2011_09_26/2011_09_26_drive_0009_sync'
image_02_dir = os.path.join(kitti_seq_dir, 'image_02/data')
velo_dir = os.path.join(kitti_seq_dir, 'velodyne_points/data')

all_idx = [i.split('.')[0] for i in os.listdir(velo_dir)]
all_images = [os.path.join(image_02_dir, '{}.png'.format(i)) for i in all_idx]
all_velos = [os.path.join(velo_dir, '{}.bin'.format(i)) for i in all_idx]

assert len(all_images) == len(all_velos), \
    'images and velos are not equal. {} vs {}'.format(len(all_images), len(all_velos))


def load_pc(pc_f):
    pc = np.fromfile(pc_f, dtype=np.float32).reshape(-1, 4)
    return pc


def show_pc():
    img_f = all_images[0]
    velo_f = all_velos[0]

    points = load_pc(velo_f)

    # fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
    #                   fgcolor=None, engine=None, size=(500, 500))
    # mlab.points3d(points[:, 0], points[:, 1], points[:, 3], mode='sphere',
    #               colormap='gnuplot', scale_factor=0.1, figure=fig)
    # mlab.show()






if __name__ == '__main__':
    show_pc()
