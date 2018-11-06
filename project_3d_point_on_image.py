"""
this file project 3d point on image
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# label_path = '/home/husky/data/kitti_object/data_object_image_2/training/label_2/'
# image_path = '/home/husky/data/kitti_object/data_object_image_2/training/image_2/'
# calib_path = '/home/husky/data/kitti_object/data_object_calib/training/calib/'

image_dir = '/media/jintain/sg/permanent/datasets/KITTI/tiny_kitti/image_2'
box2d_dir = '/media/jintain/sg/permanent/datasets/KITTI/tiny_kitti/box_2d/'
box3d_dir = '/media/jintain/sg/permanent/datasets/KITTI/tiny_kitti/box_3d/'

label_dir = '/media/jintain/sg/permanent/datasets/KITTI/tiny_kitti/box_2d/'
calib_dir = '/media/jintain/sg/permanent/datasets/KITTI/tiny_kitti/calib_02/'
predi_dir = '/media/jintain/sg/permanent/datasets/KITTI/tiny_kitti/predict_02/'


dataset = [name.split('.')[0] for name in sorted(os.listdir(label_dir))]
all_image = sorted(os.listdir(image_dir))
# np.random.shuffle(all_image)

colors = {
    'green': (0, 255, 0),
    'pink': (255, 0, 255)
}


def get_calibration_cam_to_image(cab_f):
    for line in open(cab_f):
        if 'P2:' in line:
            cam_to_img = line.strip().split(' ')
            cam_to_img = np.asarray([float(number) for number in cam_to_img[1:]])
            cam_to_img = np.reshape(cam_to_img, (3, 4))
    return cam_to_img


def plot_3d_bbox(img, label_info, is_gt=True):
    print('current label info: ', label_info)
    alpha = label_info['alpha']
    # theta_ray = label_info['theta_ray']
    box_3d = []
    center = label_info['location']
    dims = label_info['dimension']
    cam_to_img = label_info['calib']
    if is_gt:
        rot_y = label_info['rot_y']
    else:
        rot_y = alpha / 180 * np.pi + np.arctan(center[0] / center[2])
        # import pdb; pdb.set_trace()

    for i in [1, -1]:
        for j in [1, -1]:
            for k in [0, 1]:
                point = np.copy(center)
                point[0] = center[0] + i * dims[1] / 2 * np.cos(-rot_y + np.pi / 2) + (j * i) * dims[2] / 2 * np.cos(
                    -rot_y)
                point[2] = center[2] + i * dims[1] / 2 * np.sin(-rot_y + np.pi / 2) + (j * i) * dims[2] / 2 * np.sin(
                    -rot_y)
                point[1] = center[1] - k * dims[0]

                point = np.append(point, 1)
                point = np.dot(cam_to_img, point)
                point = point[:2] / point[2]
                point = point.astype(np.int16)
                box_3d.append(point)

    front_mark = []
    for i in range(4):
        point_1_ = box_3d[2 * i]
        point_2_ = box_3d[2 * i + 1]
        cv2.line(img, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), colors['pink'], 1)
    #     if i == 0 or i == 3:
    #         front_mark.append((point_1_[0], point_1_[1]))
    #         front_mark.append((point_2_[0], point_2_[1]))

    # cv2.line(img, front_mark[0], front_mark[-1], (255, 0, 0), 1)
    # cv2.line(img, front_mark[1], front_mark[2], (255, 0, 0), 1)

    for i in range(8):
        point_1_ = box_3d[i]
        point_2_ = box_3d[(i + 2) % 8]
        cv2.line(img, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), colors['pink'], 1)

    return img


def main():
    for f in all_image:
        image_file = os.path.join(image_dir, f)
        calib_file = os.path.join(calib_dir, f.replace('png', 'txt'))
        predi_file = os.path.join(predi_dir, f.replace('png', 'txt'))
        label_file = os.path.join(label_dir, f.replace('png', 'txt'))
        box3d_file = os.path.join(box3d_dir, f.replace('png', 'txt'))

        image = cv2.imread(image_file, cv2.COLOR_BGR2RGB)

        with open(label_file, 'r') as f:

            label_info = dict()
            for l in f.readlines():
                l = l.strip().split(' ')

                # this angle need a preprocess
                label_info['alpha'] = float(l[3])
                label_info['location'] = np.asarray(l[11: 14], dtype=float)
                label_info['dimension'] = np.asarray(l[8: 11], dtype=float)
                label_info['calib'] = get_calibration_cam_to_image(calib_file)
                label_info['rot_y'] = float(l[14])
                print(l[4: 7])
                label_info['box'] = np.asarray(l[4: 7], dtype=float)

                image = plot_3d_bbox(image, label_info)

        # d = 30
        # box_center = [(label_info['box'][0] + label_info['box'][2]) / 2,
        #               (label_info['box'][1] + label_info['box'][3]) / 2]
        # another_point = [int(box_center[0] + d * np.cos(label_info['alpha'])),
        #                  int(box_center[1] + d * np.sin(label_info['alpha']))]
        # cv2.arrowedLine(image, tuple(box_center), tuple(another_point), (255, 0, 0), 2)

        cv2.imshow('rr', image)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()



