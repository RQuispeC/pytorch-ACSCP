from __future__ import print_function, absolute_import
import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import h5py
from scipy.misc import imsave
import scipy.io as sio

from manage_data.get_density_map import create_density_map
from manage_data.utils import mkdir_if_missing, copy_to_directory
from manage_data.data_augmentation import augment

"""class that has a train and test unit"""
class train_test_unit(object):
    train_dir_img = ""
    train_dir_den = "" 
    test_dir_img = ""
    test_dir_den = ""
    metadata = dict()
    def __init__(self, _train_dir_img, _train_dir_den, _test_dir_img, _test_dir_den, kwargs):
        self.train_dir_img = _train_dir_img
        self.train_dir_den = _train_dir_den
        self.test_dir_img = _test_dir_img
        self.test_dir_den = _test_dir_den
        self._check_before_run()
        self.metadata = kwargs

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.train_dir_img):
            raise RuntimeError("'{}' is not available".format(self.train_dir_img))
        if not osp.exists(self.train_dir_den):
            raise RuntimeError("'{}' is not available".format(self.train_dir_den))
        if not osp.exists(self.test_dir_img):
            raise RuntimeError("'{}' is not available".format(self.test_dir_img))
        if not osp.exists(self.test_dir_den):
            raise RuntimeError("'{}' is not available".format(self.test_dir_den))

    def to_string(self):
        return "_".join([ str(key) + "_" + str(value) for key, value in sorted(self.metadata.items()) if key != 'name'])
    
"""Dataset classes"""

"""Crowd counting dataset"""

class UCF_CC_50(object):
    root = 'data/ucf_cc_50/'
    ori_dir = osp.join(root, 'UCF_CC_50')
    ori_dir_lab = osp.join(ori_dir, 'labels')
    ori_dir_img = osp.join(ori_dir, 'images')

    ori_dir_den = osp.join(ori_dir, 'density_maps') #.npy files of density maps matrices
    augmented_dir = ""
    train_test_set = []
    signature_args = ['people_thr', 'gt_mode']
    metadata = dict()
    train_test_size = 5
    
    def __init__(self, force_create_den_maps = False, force_augmentation = False, **kwargs):
        self._check_before_run()
        self.metadata = kwargs
        self._create_original_density_maps(force_create_den_maps)
        self._create_train_test(force_augmentation, kwargs)

    def _create_original_density_maps(self, force_create_den_maps):
        if not osp.exists(self.ori_dir_den):
            os.makedirs(self.ori_dir_den)
        elif not force_create_den_maps:
            return
        create_density_map(self.ori_dir_img, self.ori_dir_lab, self.ori_dir_den, mode = self.metadata['gt_mode'])

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.ori_dir):
            raise RuntimeError("'{}' is not available".format(self.ori_dir))
        if not osp.exists(self.ori_dir_img):
            raise RuntimeError("'{}' is not available".format(self.ori_dir_img))
        if not osp.exists(self.ori_dir_lab):
            raise RuntimeError("'{}' is not available".format(self.ori_dir_lab))

    def signature(self):
        return "_".join(["{}_{}".format(sign_elem, self.metadata[sign_elem]) for sign_elem in self.signature_args])

    def _create_train_test(self, force_augmentation, kwargs):
        slide_window_params = {'displace' : kwargs['displace'], 'size_x' : kwargs['size_x'], 'size_y' : kwargs['size_y'], 'people_thr' : kwargs['people_thr']}
        noise_params = {'augment_noise' : kwargs['augment_noise']}
        light_params = {'augment_light' : kwargs['augment_light'], 'bright' : kwargs['bright'], 'contrast' : kwargs['contrast']}

        file_names = os.listdir(self.ori_dir_img)
        file_names.sort()
        img_names = []
        img_ids = []
        for file_name in file_names:
            file_extention = file_name.split('.')[-1]
            file_id = file_name[:len(file_name) - len(file_extention)]
            if file_extention != 'png' and file_extention != 'jpg':
                continue
            img_names.append(file_name)
            img_ids.append(file_id)
        if len(img_names) != 50:
            raise RuntimeError("UCF_CC_50 dataset expects 50 images, {} found".format(len(img_names)))

        self.augmented_dir = osp.join(self.root, self.signature())
        augment_data = False
        if osp.exists(self.augmented_dir):  
            print("'{}' already exists".format(self.augmented_dir))
            if force_augmentation:
                augment_data = True
                print("augmenting data anyway")
            else:
                augment_data = False
                print("will not augmenting data")
        else:
            augment_data = True
            os.makedirs(self.augmented_dir)

        #using 5 fold cross validation protocol
        for fold in range(5):
            fold_dir = osp.join(self.augmented_dir, 'fold{}'.format(fold + 1))
            aug_train_dir_img = osp.join(fold_dir, 'train_img')
            aug_train_dir_den = osp.join(fold_dir, 'train_den')
            aug_train_dir_lab = osp.join(fold_dir, 'train_lab')
            fold_test_dir_img = osp.join(fold_dir, 'test_img')
            fold_test_dir_den = osp.join(fold_dir, 'test_den')
            fold_test_dir_lab = osp.join(fold_dir, 'test_lab')

            mkdir_if_missing(aug_train_dir_img)
            mkdir_if_missing(aug_train_dir_den)
            mkdir_if_missing(aug_train_dir_lab)
            mkdir_if_missing(fold_test_dir_img)
            mkdir_if_missing(fold_test_dir_den)
            mkdir_if_missing(fold_test_dir_lab)
            
            kwargs['name'] = 'ucf-fold{}'.format(fold + 1)
            train_test = train_test_unit(aug_train_dir_img, aug_train_dir_den, fold_test_dir_img, fold_test_dir_den, kwargs.copy())
            self.train_test_set.append(train_test)

            if augment_data:
                test_img = img_names[fold * 10: (fold + 1) * 10]
                test_ids = img_ids[fold * 10: (fold + 1) * 10]
                test_den_paths = [osp.join(self.ori_dir_den, img_id + 'npy') for img_id in test_ids]
                test_lab_paths = [osp.join(self.ori_dir_lab, img_id + 'json') for img_id in test_ids]
                test_img_paths = [osp.join(self.ori_dir_img, img) for img in test_img]

                train_img = sorted(list(set(img_names) - set(test_img)))
                train_ids = sorted(list(set(img_ids) - set(test_ids)))
                train_den_paths = [osp.join(self.ori_dir_den, img_id + 'npy') for img_id in train_ids]
                train_lab_paths = [osp.join(self.ori_dir_lab, img_id + 'json') for img_id in train_ids]
                train_img_paths = [osp.join(self.ori_dir_img, img) for img in train_img]

                #augment train data
                print("Augmenting {}".format(kwargs['name']))
                augment(train_img_paths, train_lab_paths, train_den_paths, aug_train_dir_img, aug_train_dir_lab, aug_train_dir_den, slide_window_params, noise_params, light_params)
                copy_to_directory(test_den_paths, fold_test_dir_den)
                copy_to_directory(test_lab_paths, fold_test_dir_lab)
                copy_to_directory(test_img_paths, fold_test_dir_img)

class ShanghaiTech(object):
    root = 'data/ShanghaiTech/'
    ori_dir_partA = osp.join(root, 'part_A')
    ori_dir_partA_train = osp.join(ori_dir_partA, 'train_data')
    ori_dir_partA_train_mat = osp.join(ori_dir_partA_train, 'ground-truth')
    ori_dir_partA_train_img = osp.join(ori_dir_partA_train, 'images')
    ori_dir_partA_test = osp.join(ori_dir_partA, 'test_data')
    ori_dir_partA_test_mat = osp.join(ori_dir_partA_test, 'ground-truth')
    ori_dir_partA_test_img = osp.join(ori_dir_partA_test, 'images')

    ori_dir_partB = osp.join(root, 'part_B')
    ori_dir_partB_train = osp.join(ori_dir_partB, 'train_data')
    ori_dir_partB_train_mat = osp.join(ori_dir_partB_train, 'ground-truth')
    ori_dir_partB_train_img = osp.join(ori_dir_partB_train, 'images')
    ori_dir_partB_test = osp.join(ori_dir_partB, 'test_data')
    ori_dir_partB_test_mat = osp.join(ori_dir_partB_test, 'ground-truth')
    ori_dir_partB_test_img = osp.join(ori_dir_partB_test, 'images')

    #to be computed
    ori_dir_partA_train_lab = osp.join(ori_dir_partA_train, 'labels')
    ori_dir_partA_train_den = osp.join(ori_dir_partA_train, 'density_maps')
    ori_dir_partA_test_lab = osp.join(ori_dir_partA_test, 'labels')
    ori_dir_partA_test_den = osp.join(ori_dir_partA_test, 'density_maps')

    ori_dir_partB_train_lab = osp.join(ori_dir_partB_train, 'labels')
    ori_dir_partB_train_den = osp.join(ori_dir_partB_train, 'density_maps')
    ori_dir_partB_test_lab = osp.join(ori_dir_partB_test, 'labels')
    ori_dir_partB_test_den = osp.join(ori_dir_partB_test, 'density_maps')

    augmented_dir_partA = ""
    augmented_dir_partB = ""
    train_test_set = []
    signature_args = ['people_thr', 'gt_mode']
    metadata = dict()
    train_test_size = 2

    def __init__(self, force_create_den_maps = False, force_augmentation = False, **kwargs):
        self._check_before_run()
        self.metadata = kwargs
        self._create_labels()
        self._create_original_density_maps(force_create_den_maps)
        self._create_train_test(force_augmentation, kwargs)

    def _create_original_density_maps(self, force_create_den_maps):
        all_density_dirs_exist = osp.exists(self.ori_dir_partA_train_den)
        all_density_dirs_exist = all_density_dirs_exist and osp.exists(self.ori_dir_partA_test_den)
        all_density_dirs_exist = all_density_dirs_exist and osp.exists(self.ori_dir_partB_train_den)
        all_density_dirs_exist = all_density_dirs_exist and osp.exists(self.ori_dir_partB_test_den)
        if not all_density_dirs_exist:
            mkdir_if_missing(self.ori_dir_partA_train_den)
            mkdir_if_missing(self.ori_dir_partA_test_den)
            mkdir_if_missing(self.ori_dir_partB_train_den)
            mkdir_if_missing(self.ori_dir_partB_test_den)
        elif not force_create_den_maps:
            return
        create_density_map(self.ori_dir_partA_train_img, self.ori_dir_partA_train_lab, self.ori_dir_partA_train_den, mode = self.metadata['gt_mode'])
        create_density_map(self.ori_dir_partA_test_img, self.ori_dir_partA_test_lab, self.ori_dir_partA_test_den, mode = self.metadata['gt_mode'])
        create_density_map(self.ori_dir_partB_train_img, self.ori_dir_partB_train_lab, self.ori_dir_partB_train_den, mode = self.metadata['gt_mode'])
        create_density_map(self.ori_dir_partB_test_img, self.ori_dir_partB_test_lab, self.ori_dir_partB_test_den, mode = self.metadata['gt_mode'])

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.ori_dir_partA):
            raise RuntimeError("'{}' is not available".format(self.ori_dir_partA))
        if not osp.exists(self.ori_dir_partB):
            raise RuntimeError("'{}' is not available".format(self.ori_dir_partB))
        if not osp.exists(self.ori_dir_partA_train):
            raise RuntimeError("'{}' is not available".format(self.ori_dir_partA_train))
        if not osp.exists(self.ori_dir_partB_train):
            raise RuntimeError("'{}' is not available".format(self.ori_dir_partB_train))
        if not osp.exists(self.ori_dir_partA_test):
            raise RuntimeError("'{}' is not available".format(self.ori_dir_partA_test))
        if not osp.exists(self.ori_dir_partB_test):
            raise RuntimeError("'{}' is not available".format(self.ori_dir_partB_test))
        if not osp.exists(self.ori_dir_partA_train_img):
            raise RuntimeError("'{}' is not available".format(self.ori_dir_partA_train_img))
        if not osp.exists(self.ori_dir_partA_train_mat):
            raise RuntimeError("'{}' is not available".format(self.ori_dir_partA_train_mat))
        if not osp.exists(self.ori_dir_partB_train_img):
            raise RuntimeError("'{}' is not available".format(self.ori_dir_partB_train_img))
        if not osp.exists(self.ori_dir_partB_train_mat):
            raise RuntimeError("'{}' is not available".format(self.ori_dir_partB_train_mat))
        if not osp.exists(self.ori_dir_partA_test_img):
            raise RuntimeError("'{}' is not available".format(self.ori_dir_partA_test_img))
        if not osp.exists(self.ori_dir_partA_test_mat):
            raise RuntimeError("'{}' is not available".format(self.ori_dir_partA_test_mat))
        if not osp.exists(self.ori_dir_partB_test_img):
            raise RuntimeError("'{}' is not available".format(self.ori_dir_partB_test_img))
        if not osp.exists(self.ori_dir_partB_test_mat):
            raise RuntimeError("'{}' is not available".format(self.ori_dir_partB_test_mat))

    def _json_to_string(self, array):
        """
        converts json to string specifically for shanghai tech dataset
        """
        if len(array)==0:
            return '[]'
        line = '['
        for i in range(len(array)):
            line += '{\"x\":'+str(array[i][0])+',\"y\":'+str(array[i][1])+'},'
        return line[0:len(line)-1]+']'
    
    def _convert_mat_to_json(self, in_dir, out_dir):
        """
        converts every .mat file in in_dir to a .json equivalent in out_dir
        """
        print("converting mat to json from {}".format(in_dir))
        file_names = os.listdir(in_dir)
        for mat_file in file_names:
            mat_file_path = osp.join(in_dir, mat_file)
            file_extention = mat_file.split('.')[-1]
            file_id = mat_file[3:len(mat_file) - len(file_extention)]
            json_file_path = osp.join(out_dir, file_id + 'json')
            labels = sio.loadmat(mat_file_path)
            labels = labels['image_info'][0][0][0][0][0]
            labels = str(self._json_to_string(labels))
            with open(json_file_path, 'w') as outfile:
                outfile.write(labels)

    def _create_labels(self):
        mkdir_if_missing(self.ori_dir_partA_train_lab)
        mkdir_if_missing(self.ori_dir_partA_test_lab)
        mkdir_if_missing(self.ori_dir_partB_train_lab)
        mkdir_if_missing(self.ori_dir_partB_test_lab)
        #check if number os files is equal
        if len(os.listdir(self.ori_dir_partA_train_mat)) != len(os.listdir(self.ori_dir_partA_train_lab)):
            self._convert_mat_to_json(self.ori_dir_partA_train_mat, self.ori_dir_partA_train_lab)
        if len(os.listdir(self.ori_dir_partA_test_mat)) != len(os.listdir(self.ori_dir_partA_test_lab)):
            self._convert_mat_to_json(self.ori_dir_partA_test_mat, self.ori_dir_partA_test_lab)
        if len(os.listdir(self.ori_dir_partB_train_mat)) != len(os.listdir(self.ori_dir_partB_train_lab)):
            self._convert_mat_to_json(self.ori_dir_partB_train_mat, self.ori_dir_partB_train_lab)
        if len(os.listdir(self.ori_dir_partB_test_mat)) != len(os.listdir(self.ori_dir_partB_test_lab)):
           self._convert_mat_to_json(self.ori_dir_partB_test_mat, self.ori_dir_partB_test_lab)

    def signature(self):
        return "_".join(["{}_{}".format(sign_elem, self.metadata[sign_elem]) for sign_elem in self.signature_args])

    def _create_train_test(self, force_augmentation, kwargs):
        slide_window_params = {'displace' : kwargs['displace'], 'size_x' : kwargs['size_x'], 'size_y' : kwargs['size_y'], 'people_thr' : kwargs['people_thr']}
        noise_params = {'augment_noise' : kwargs['augment_noise']}
        light_params = {'augment_light' : kwargs['augment_light'], 'bright' : kwargs['bright'], 'contrast' : kwargs['contrast']}

        #shanghaiTech part A
        self.augmented_dir_partA = osp.join(self.ori_dir_partA, self.signature())
        augment_data_A = False
        if osp.exists(self.augmented_dir_partA):
            print("'{}' already exists".format(self.augmented_dir_partA))
            if force_augmentation:
                augment_data_A = True
                print("augmenting data anyway")
            else:
                augment_data_A = False
                print("will not augmenting data")
        else:
            augment_data_A = True
            os.makedirs(self.augmented_dir_partA)

        aug_dir_partA_img = osp.join(self.augmented_dir_partA, "train_img")
        aug_dir_partA_den = osp.join(self.augmented_dir_partA, "train_den")
        aug_dir_partA_lab = osp.join(self.augmented_dir_partA, "train_lab")
        mkdir_if_missing(aug_dir_partA_img)
        mkdir_if_missing(aug_dir_partA_den)
        mkdir_if_missing(aug_dir_partA_lab)

        kwargs['name'] = 'shanghai-partA'
        part_A_train_test = train_test_unit(aug_dir_partA_img, aug_dir_partA_den, self.ori_dir_partA_test_img, self.ori_dir_partA_test_den, kwargs.copy())
        self.train_test_set.append(part_A_train_test)

        if augment_data_A:
            ori_img_paths = [osp.join(self.ori_dir_partA_train_img, file_name) for file_name in sorted(os.listdir(self.ori_dir_partA_train_img))]
            ori_lab_paths = [osp.join(self.ori_dir_partA_train_lab, file_name) for file_name in sorted(os.listdir(self.ori_dir_partA_train_lab))]
            ori_den_paths = [osp.join(self.ori_dir_partA_train_den, file_name) for file_name in sorted(os.listdir(self.ori_dir_partA_train_den))]
            augment(ori_img_paths, ori_lab_paths, ori_den_paths, aug_dir_partA_img, aug_dir_partA_lab, aug_dir_partA_den, slide_window_params, noise_params, light_params)

        #shanghaiTech part B
        self.augmented_dir_partB = osp.join(self.ori_dir_partB, self.signature())
        augment_data_B = False
        if osp.exists(self.augmented_dir_partB):
            print("'{}' already exists".format(self.augmented_dir_partB))
            if force_augmentation:
                augment_data_B = True
                print("augmenting data anyway")
            else:
                augment_data_B = False
                print("will not augmenting data")
        else:
            augment_data_B = True
            os.makedirs(self.augmented_dir_partB)

        aug_dir_partB_img = osp.join(self.augmented_dir_partB, "train_img")
        aug_dir_partB_den = osp.join(self.augmented_dir_partB, "train_den")
        aug_dir_partB_lab = osp.join(self.augmented_dir_partB, "train_lab")
        mkdir_if_missing(aug_dir_partB_img)
        mkdir_if_missing(aug_dir_partB_den)
        mkdir_if_missing(aug_dir_partB_lab)

        kwargs['name'] = 'shanghai-partB'
        part_B_train_test = train_test_unit(aug_dir_partB_img, aug_dir_partB_den, self.ori_dir_partB_test_img, self.ori_dir_partB_test_den, kwargs.copy())
        self.train_test_set.append(part_B_train_test)

        if augment_data_B:
            ori_img_paths = [osp.join(self.ori_dir_partB_train_img, file_name) for file_name in sorted(os.listdir(self.ori_dir_partB_train_img))]
            ori_lab_paths = [osp.join(self.ori_dir_partB_train_lab, file_name) for file_name in sorted(os.listdir(self.ori_dir_partB_train_lab))]
            ori_den_paths = [osp.join(self.ori_dir_partB_train_den, file_name) for file_name in sorted(os.listdir(self.ori_dir_partB_train_den))]
            augment(ori_img_paths, ori_lab_paths, ori_den_paths, aug_dir_partB_img, aug_dir_partB_lab, aug_dir_partB_den, slide_window_params, noise_params, light_params)

"""Create dataset"""

__factory = {
    'ucf-cc-50': UCF_CC_50,
    'shanghai-tech': ShanghaiTech
}

def get_names():
    return __factory.keys()

def init_dataset(name, force_create_den_maps = False, force_augmentation = False, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](force_create_den_maps, force_augmentation, **kwargs)

if __name__ == '__main__':
    pass
