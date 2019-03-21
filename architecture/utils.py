import cv2
import numpy as np
import os

import torch
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')
import matplotlib.pylab as plt
import gc

MAXIMUM_CNT = 2.7675038540969217

def save_results(img, gt_density_map, et_density_map, idx, output_dir):
    idx = idx[0]
    img = img[0, 0]
    gt_density_map = np.array(gt_density_map[0, 0])
    et_density_map = et_density_map[0, 0]
    gt_count = np.sum(gt_density_map)
    et_count = np.sum(et_density_map)
    maxi = gt_density_map.max()
    if maxi != 0:
        gt_density_map = gt_density_map*(255. / maxi)
        et_density_map = et_density_map*(255. / maxi)
    #print("min, max GT - ET", gt_density_map.max(), gt_density_map.min(), et_density_map.max(), et_density_map.min())

    if gt_density_map.shape[1] != img.shape[1]:
        gt_density_map = cv2.resize(gt_density_map, (img.shape[1], img.shape[0]))
        et_density_map = cv2.resize(et_density_map, (img.shape[1], img.shape[0]))
    
    fig = plt.figure(figsize = (30, 20))
    a = fig.add_subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    a.set_title('input')
    plt.axis('off')
    a = fig.add_subplot(1, 3, 2)
    plt.imshow(gt_density_map)
    a.set_title('ground thruth {:.2f}'.format(gt_count))
    plt.axis('off')
    a = fig.add_subplot(1, 3, 3)
    plt.imshow(et_density_map)
    a.set_title('estimated {:.2f}'.format(et_count))
    plt.axis('off')
    
    img_file_name = os.path.join(output_dir, str(idx) + ".jpg")
    fig.savefig(img_file_name, bbox_inches='tight')
    fig.clf()
    plt.close()
    del a
    gc.collect()

def save_density_map(density_map,output_dir, fname='results.png'):
    density_map = 255*density_map/np.max(density_map)
    density_map= density_map[0][0]
    cv2.imwrite(os.path.join(output_dir,fname),density_map)
    
def display_results(input_img, gt_data,density_map):
    input_img = input_img[0][0]
    gt_data = 255*gt_data/np.max(gt_data)
    density_map = 255*density_map/np.max(density_map)
    gt_data = gt_data[0][0]
    density_map= density_map[0][0]
    if density_map.shape[1] != input_img.shape[1]:
         input_img = cv2.resize(input_img, (density_map.shape[1],density_map.shape[0]))
    result_img = np.hstack((input_img,gt_data,density_map))
    result_img  = result_img.astype(np.uint8, copy=False)
    cv2.imshow('Result', result_img)
    cv2.waitKey(0)
