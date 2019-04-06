import numpy as np
import cv2
from scipy.signal import gaussian
import json
from matplotlib.image import imsave

import os.path as osp
import json
import os

from manage_data.utils import cnt_overlaps

F_SZ = 15
SIGMA = 15

VALID_GT_MODES = ['same']

def gauss_ker(sigma, shape):
    gaussian_kernel = np.outer(gaussian(shape[0], std = sigma), gaussian(shape[1], std = sigma))
    gaussian_kernel /= gaussian_kernel.sum()
    return gaussian_kernel

def get_density_map_gaussian(img_shape, points, mode = 'same'):
    """
        Creates a density map with img_shape and gaussians over points

        Inputs:
        - img_shape: tuple of the heigth and width of the ouput density map
        - points: positions for the head of people
        - mode: ["same", "k-nearest"] if "same" is used all the gaussian kernels has the same kernel size, else and k-nearest kernel is used.

        Ouputs:
        - density_map of shape img_shape
    """
    img_density = np.zeros(img_shape)
    h, w = img_shape
    for ind, point in enumerate(points):
        kernel_size_y, kernel_size_x = F_SZ, F_SZ
        SIGMA = 4
        H = gauss_ker(SIGMA, [kernel_size_y, kernel_size_x])
        x = min(w,max(1,(int)(abs(point[1]))))
        y = min(h,max(1,(int)(abs(point[0]))))
        if x > w or y > h:
            continue
        x1 = x - (int)(np.floor(kernel_size_x/2))
        y1 = y - (int)(np.floor(kernel_size_y/2))
        x2 = x + (int)(np.floor(kernel_size_x/2))
        y2 = y + (int)(np.floor(kernel_size_y/2))
        dfx1 = 0
        dfy1 = 0
        dfx2 = 0
        dfy2 = 0
        change_H = False
        if x1 < 1:
            dfx1 = abs(x1) + 1
            x1 = 1
            change_H = True
        if y1 < 1:
            dfy1 = abs(y1) + 1;
            y1 = 1;
            change_H = True
        if x2 > w:
            dfx2 = x2 - w
            x2 = w
            change_H = True
        if y2 > h:
            dfy2 = y2 - h
            y2 = h
            change_H = True
        x1h = 1+dfx1
        y1h = 1+dfy1
        x2h = kernel_size_x - dfx2
        y2h = kernel_size_x - dfy2
        x1 = (int)(x1)
        x2 = (int)(x2)
        y1 = (int)(y1)
        y2 = (int)(y2)
        if change_H or y2 - y1 != kernel_size_y or x2 - x1 != kernel_size_x:
            H = gauss_ker(SIGMA, [y2 - y1, x2 - x1])
        img_density[y1:y2, x1:x2] += H
    return img_density

def create_density_map(imgs_path, labels_path, density_maps_path, mode = 'same'):
    """
    Generates density maps files (.npy) inside directory density_maps_path

    input:

    imgs_path: directory with original images (.jpg or .png)
    labels_path: directory with data labels (.json)
    density_maps_path: directory where generated density maps (.npy) files are stored
    mode: method used for generation of ground thuth images
    """
    if not mode in VALID_GT_MODES:
        raise RuntimeError("'{}' is invalid mode for grounth thruth generation. Valid modes are: {}".format(self.ori_dir_img, ', '.join(VALID_GT_MODES)))
    file_names = os.listdir(imgs_path)
    file_names.sort()
    print("Creating density maps for '{}', {} images will be processed".format(imgs_path, len(file_names)))

    for file_name in file_names:
        file_extention = file_name.split('.')[-1]
        file_id = file_name[:len(file_name) - len(file_extention)]
        if file_extention != 'png' and file_extention != 'jpg':
            continue
        file_path = osp.join(imgs_path, file_name)
        label_path = osp.join(labels_path, file_id + 'json')
        density_map_path = osp.join(density_maps_path, file_id + 'npy')
        with open(label_path) as data_file:
            labels = json.load(data_file)
        points = []
        for p in labels:
            points.append([p['y'], p['x']])
        img = cv2.imread(file_path)
        img_den = get_density_map_gaussian(img.shape[:2], points, mode = mode)
        np.save(density_map_path, img_den)



