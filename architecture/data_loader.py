import numpy as np
import cv2
import os
import random
import pandas as pd


class ImageDataLoader():
    def __init__(self, data_path, gt_path, shuffle=False, gt_downsample=False, pre_load=False, use_clahe = False, batch_size = 1, multiple_size = False):
        #pre_load: if true, all training and validation images are loaded into CPU RAM for faster processing.
        #          This avoids frequent file reads. Use this only for small datasets.
        self.data_path = data_path
        self.gt_path = gt_path
        self.gt_downsample = gt_downsample
        self.batch_size = batch_size
        self.pre_load = pre_load
        self.data_files = [filename for filename in os.listdir(data_path) \
                           if os.path.isfile(os.path.join(data_path,filename)) and os.path.splitext(filename)[1] == '.jpg']
        self.data_files.sort()
        self.use_clahe = use_clahe
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.shuffle = shuffle
        self.multiple_size = multiple_size
        if shuffle:
            random.seed(2468)
        self.num_samples = len(self.data_files)
        self.blob_list = {}        
        self.id_list = np.arange(0,self.num_samples)
        if self.pre_load:
            print ('Pre-loading the data. This may take a while...')
            idx = 0
            for fname in self.data_files:
                if os.path.splitext(fname)[1] == ".json":
                    continue
                img = cv2.imread(os.path.join(self.data_path,fname),0)
                if self.use_clahe:
                    img = self.clahe.apply(img)
                img = img.astype(np.float32, copy=False)
                ht = img.shape[0]
                wd = img.shape[1]
                img = img.reshape((1,1,img.shape[0],img.shape[1]))
                #den = pd.read_csv(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).as_matrix()
                den = np.load(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.npy'))
                den = den.astype(np.float32, copy=False)
                if self.gt_downsample or self.multiple_size:
                    ht = img.shape[1]
                    wd = img.shape[2]
                    wd_1 = (int)(wd/4)
                    ht_1 = (int)(ht/4)
                    den_small = cv2.resize(den,(wd_1,ht_1))
                    den_small = den_small * ((wd*ht)/(wd_1*ht_1))
                    if self.multiple_size:
                        den_small = den_small.reshape((1, den_small.shape[0], den_small.shape[1]))
                    else:
                        den = den_small

                blob = {}
                blob['data']=img
                blob['gt_density']=den
                blob['fname'] = fname
                blob['idx'] = idx
                if self.multiple_size:
                    blob['gt_density_small'] = np.array(den_small)
                self.blob_list[idx] = blob
                idx = idx+1
                if idx % 100 == 0:                    
                    print ('Loaded ', idx, '/', self.num_samples, 'files')
               
            print ('Completed Loading ', idx, 'files')
        
        
    def __iter__(self):
        if self.shuffle:
            if self.pre_load:
                random.shuffle(self.id_list)
            else:
                random.shuffle(self.data_files)
        files = np.array(self.data_files)
        id_list = np.array(self.id_list)
       
        for ind in range(0, len(id_list), self.batch_size):
            idx = id_list[ind: ind + self.batch_size]
            if self.pre_load:
                blob = self.blob_list[idx]
                blob['idx'] = idx
            else:
                fnames = files[idx]
                imgs = []
                dens = []
                dens_small = []
                for fname in fnames:
                    if not os.path.isfile(os.path.join(self.data_path,fname)):
                        print("Error: file '{}' doen't exists".format(os.path.join(self.data_path,fname)))
                    img = cv2.imread(os.path.join(self.data_path,fname),0)
                    if self.use_clahe:
                        img = self.clahe.apply(img)
                    img = img.astype(np.float32, copy=False)
                    img = img.reshape((1,img.shape[0],img.shape[1]))
                    den = np.load(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.npy'))
                    den  = den.astype(np.float32, copy=False)
                    if self.gt_downsample or self.multiple_size:
                        ht = img.shape[1]
                        wd = img.shape[2]
                        wd_1 = (int)(wd/4)
                        ht_1 = (int)(ht/4)
                        den_small = cv2.resize(den,(wd_1,ht_1))
                        den_small = den_small * ((wd*ht)/(wd_1*ht_1))
                        if self.multiple_size:
                            den_small = den_small.reshape((1, den_small.shape[0], den_small.shape[1]))
                            dens_small.append(den_small)
                        else:
                            den = den_small                
                    den = den.reshape((1, den.shape[0], den.shape[1]))
                    imgs.append(img)
                    dens.append(den)

                blob = {}
                blob['data']=np.array(imgs)
                blob['gt_density']=np.array(dens)
                blob['fname'] = np.array(fnames)
                blob['idx'] = np.array(idx)
                if self.multiple_size:
                    blob['gt_density_small'] = np.array(dens_small)
            yield blob
            
    def get_num_samples(self):
        return self.num_samples
                
        
            
        
