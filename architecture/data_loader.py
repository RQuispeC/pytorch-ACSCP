import numpy as np
import cv2
import os
import random
import pandas as pd


class ImageDataLoader():
    def __init__(self, data_path, gt_path, shuffle=False, batch_size = 1, test_loader = False, img_width = 256, img_height = 256, test_overlap = False):
        self.data_path = data_path
        self.gt_path = gt_path
        self.batch_size = batch_size
        self.test_loader = test_loader
        self.img_width = img_width
        self.img_height = img_height
        self.test_overlap = test_overlap
        self.data_files = [filename for filename in os.listdir(data_path) \
                           if os.path.isfile(os.path.join(data_path,filename)) and os.path.splitext(filename)[1] == '.jpg']
        self.data_files.sort()
        self.shuffle = shuffle
        if shuffle:
            random.seed(2468)
        self.num_samples = len(self.data_files)
        self.blob_list = {}
        self.id_list = np.arange(0,self.num_samples)
        
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.data_files)
        files = np.array(self.data_files)
        id_list = np.array(self.id_list)
       
        for ind in range(0, len(id_list), self.batch_size):
            idx = id_list[ind: ind + self.batch_size]
            fnames = files[idx]
            imgs = []
            dens = []
            dens_small = []
            for fname in fnames:
                if not os.path.isfile(os.path.join(self.data_path,fname)):
                    print("Error: file '{}' doen't exists".format(os.path.join(self.data_path,fname)))
                img = cv2.imread(os.path.join(self.data_path,fname),0)
                img = img.astype(np.float32, copy=False)
                img = img.reshape((1,img.shape[0],img.shape[1]))

                den = np.load(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.npy'))
                den  = den.astype(np.float32, copy=False)
                den = den.reshape((1, den.shape[0], den.shape[1]))

                if self.test_loader: #loader is for testing, then we divide the image in chunks of size (img_height, img_width)
                    _, h, w = img.shape
                    orig_shape = (h, w)
                    # compute padding
                    if self.test_overlap:
                        padding_h = self.img_height - max(h % self.img_height, (h - self.img_height//2) % self.img_height)
                        padding_w = self.img_width - max(w % self.img_width, (w - self.img_width//2) % self.img_width)
                    else:
                        padding_h = self.img_height - (h % self.img_height)
                        padding_w = self.img_width - (w % self.img_width)

                    # add padding
                    img = np.concatenate((img, np.zeros((img.shape[0], padding_h, img.shape[2]))), axis =1)
                    den = np.concatenate((den, np.zeros((img.shape[0], padding_h, img.shape[2]))), axis =1)
                    img = np.concatenate((img, np.zeros((img.shape[0], img.shape[1], padding_w))), axis =2)
                    den = np.concatenate((den, np.zeros((img.shape[0], img.shape[1], padding_w))), axis =2)
                    assert img.shape[1] % 2 == 0 and img.shape[2] % 2 == 0, "Inputs images must have even dimensions, found {}".format(img.shape)
                    
                    # create batch for test
                    _, h, w = img.shape
                    new_shape = (h, w)
                    disp_height = self.img_height // 2 if self.test_overlap else self.img_height
                    disp_width = self.img_width // 2 if self.test_overlap else self.img_width
                    for i in range(0, h - self.img_height + 1, disp_height):
                        for j in range(0, w - self.img_width + 1, disp_width):
                            chunk_img = img[0, i:i + self.img_height, j:j + self.img_width]
                            chunk_den = den[0, i:i + self.img_height, j:j + self.img_width]
                            chunk_img = chunk_img.reshape((1, chunk_img.shape[0], chunk_img.shape[1]))
                            chunk_den = chunk_den.reshape((1, chunk_den.shape[0], chunk_den.shape[1]))
                            imgs.append(chunk_img)
                            dens.append(chunk_den)
                else:
                    imgs.append(img)
                    dens.append(den)
            blob = {}
            blob['data']=np.array(imgs)
            blob['gt_density']=np.array(dens)
            blob['fname'] = np.array(fnames)
            blob['idx'] = np.array(idx)
            if self.test_loader:
                blob['orig_shape'] = np.array(orig_shape)
                blob['new_shape'] = np.array(new_shape)
            yield blob
            
    def get_num_samples(self):
        return self.num_samples
    
    def recontruct_test(self, img_batch, den_batch, orig_shape, new_shape):
        disp_height = self.img_height // 2 if self.test_overlap else self.img_height
        disp_width = self.img_width // 2 if self.test_overlap else self.img_width
        img = np.zeros(new_shape)
        cnt = np.zeros(new_shape)
        den = np.zeros(new_shape)
        ind = 0
        for i in range(0, new_shape[0] - self.img_height + 1, disp_height):
            for j in range(0, new_shape[1] - self.img_width + 1, disp_width):
                img[i:i + self.img_height, j:j + self.img_width] = img_batch[ind, 0]
                den[i:i + self.img_height, j:j + self.img_width] += den_batch[ind, 0]
                cnt[i:i + self.img_height, j:j + self.img_width] += 1
                ind += 1
        den /= cnt
        
        #crop to original shape
        img = img[:orig_shape[0], :orig_shape[1]].reshape((1, 1, orig_shape[0], orig_shape[1]))
        den = den[:orig_shape[0], :orig_shape[1]].reshape((1, 1, orig_shape[0], orig_shape[1]))
        return img, den