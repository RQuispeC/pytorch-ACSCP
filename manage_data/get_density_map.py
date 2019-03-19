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

VALID_GT_MODES = ['same', 'k-nearest', 'face']

def gauss_ker(sigma, shape):
    gaussian_kernel = np.outer(gaussian(shape[0], std = sigma), gaussian(shape[1], std = sigma))
    gaussian_kernel /= gaussian_kernel.sum()
    return gaussian_kernel

def k_nearest(points, top = 3):
    """
    Computes kernel size for gaussians based on top closest points

    points has format (y, x): (heigth, width)
    """
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters = len(points))
    if len(points) == 0:
        return []
    model.fit(points)
    dist_mat = model.transform(points)
    mean_dists = []
    for p in dist_mat:
        indexes = np.argsort(p)
        indexes = indexes[1: top+1] #ignore point itself
        dists = p[indexes]
        mean_dists.append(np.mean(dists))
    mean_dists = np.array(mean_dists)
    kernel_sizes = mean_dists * 3 
    kernel_sizes = kernel_sizes.astype(int)
    return kernel_sizes

def interpolate_scale(img_shape, points, bb_faces, minimum_size = 5):
    """
    Computes kernel size for gaussians based on top closest detected faces

    points has format (y, x): (heigth, width)
    """
    if len(points) == 0:
        return []
    if len(bb_faces) < 5:
        kernel_sizes = np.vstack(([minimum_size]*len(points), [minimum_size]*len(points)))
        kernel_sizes = np.transpose(kernel_sizes)
        return kernel_sizes
    
    bb_centers = np.vstack(((bb_faces[:, 1] + bb_faces[:, 3])//2, (bb_faces[:, 0] + bb_faces[:, 2])//2))
    bb_lengths = np.vstack(((bb_faces[:, 3] - bb_faces[:, 1]), (bb_faces[:, 2] - bb_faces[:, 0])))
    bb_centers = np.transpose(bb_centers)
    bb_lengths = np.transpose(bb_lengths)
    points = np.array(points)
    new_bbs = []
    #compute initial bb sizes
    for p in points:
        dist = (bb_centers - p)
        dist = np.sqrt(dist[:, 0]**2 + dist[:, 1]**2)
        weight = (1./dist)
        lenght_y = np.sum(weight*bb_lengths[:, 0])/np.sum(weight)
        lenght_x = np.sum(weight*bb_lengths[:, 1])/np.sum(weight)
        new_bb = [max(0, p[1] - lenght_x/2), max(0, p[0] - lenght_y/2), min(img_shape[1], p[1] + lenght_x/2), min(img_shape[0], p[0] + lenght_y/2)]
        new_bb = [int(new_bb[0]), int(new_bb[1]), int(new_bb[2]), int(new_bb[3])]
        new_bbs.append(new_bb)
    
    boxes_overlap, _ = cnt_overlaps(new_bbs)
    boxes_overlap = np.array(boxes_overlap)

    will_update = (boxes_overlap >= 7)#threhold over number of overlaps

    kernel_sizes = []
    #refine bb estimation
    for ind in range(len(points)):
        p = points[ind]
        if will_update[ind]: #use minimum size
            new_bb = [max(0, p[1] - minimum_size/2), max(0, p[0] - minimum_size/2), min(img_shape[1], p[1] + minimum_size/2), min(img_shape[0], p[0] + minimum_size/2)]
            new_bb = [int(new_bb[0]), int(new_bb[1]), int(new_bb[2]), int(new_bb[3])]
            new_bbs[ind] = new_bb
            kernel_sizes.append([minimum_size, minimum_size])
        else: #interpolate scale
            dist = (bb_centers - p)
            dist = np.sqrt(dist[:, 0]**2 + dist[:, 1]**2)
            weight = (1./dist)**10
            lenght_y = np.sum(weight*bb_lengths[:, 0])/np.sum(weight)
            lenght_x = np.sum(weight*bb_lengths[:, 1])/np.sum(weight)
            new_bb = [max(0, p[1] - lenght_x/2), max(0, p[0] - lenght_y/2), min(img_shape[1], p[1] + lenght_x/2), min(img_shape[0], p[0] + lenght_y/2)]
            new_bb = [int(new_bb[0]), int(new_bb[1]), int(new_bb[2]), int(new_bb[3])]
            new_bbs[ind] = new_bb
            kernel_sizes.append([int(lenght_y), int(lenght_x)])

    kernel_sizes = np.array(kernel_sizes)
    new_bbs = np.array(new_bbs)
    return kernel_sizes

def get_density_map_gaussian(img_shape, points, mode = 'same', bb_faces = []):
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
    if mode == 'k-nearest':
        kernel_sizes = k_nearest(points)
    if mode == 'face':
        kernel_sizes = interpolate_scale(img_shape, points, bb_faces)
    for ind, point in enumerate(points):
        if mode == 'same':
            kernel_size_y, kernel_size_x = F_SZ, F_SZ
            SIGMA = 4
        elif mode == 'k-nearest':
            kernel_size_y, kernel_size_x = kernel_sizes[ind], kernel_sizes[ind]
            SIGMA = 15
        elif mode == 'face':
            kernel_size_y, kernel_size_x = kernel_sizes[ind]
            SIGMA = 8
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

def create_density_map(imgs_path, labels_path, density_maps_path, det_faces_path, mode = 'same'):
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
    if mode == 'face' and not osp.exists(det_faces_path):
        raise RuntimeError("'' doesn't exists, can't use 'face' mode for ground truth creation".format(det_faces_path))
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
        if mode == 'face':
            face_path = osp.join(det_faces_path, file_id + 'npy')
            tiny_faces = np.load(face_path)
            img_den = get_density_map_gaussian(img.shape[:2], points, mode = mode, bb_faces = tiny_faces)
        else:
            img_den = get_density_map_gaussian(img.shape[:2], points, mode = mode)
        np.save(density_map_path, img_den)



