"""
set of functions used for debbuging during development
"""

import numpy as np
import cv2
import json

import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')
import matplotlib.pylab as plt
import gc

def create_points(shape):
  points = []
  for i in range(shape[0]):
    for j in range(shape[1]):
      points.append([i, j])
  return points

def max_value_gt():
  shape = (15, 15)
  points = create_points(shape)
  img = f(shape, points)
  print(np.max(img))

def plot_img_gt():
  base_dir  = "/home/quispe/Documents/crowd-counting/code/"

  #data_dir = "data/ucf_cc_50/people_thr_20_gt_mode_face/fold1/"
  #img = cv2.imread(base_dir + data_dir + "train_img/0000234.jpg", 0)
  #lab = json.load(open(base_dir + data_dir + "train_lab/0000234.json"))
  #gt  = np.load(base_dir + data_dir + "train_den/0000234.npy")

  #data_dir = "data/ucf_cc_50/UCF_CC_50/"
  #img = cv2.imread(base_dir + data_dir + "images/02.jpg", 0)
  #lab = json.load(open(base_dir + data_dir + "labels/02.json"))
  #gt  = np.load(base_dir + data_dir + "density_maps/02.npy")
  
  data_dir = "data/ShanghaiTech/part_A/train_data/"
  img = cv2.imread(base_dir + data_dir + "images/IMG_74.jpg", 0)
  lab = json.load(open(base_dir + data_dir + "labels/IMG_74.json"))
  gt  = np.load(base_dir + data_dir + "density_maps/IMG_74.npy")
  

  gt_cnt = np.sum(gt)
  gt = gt*255.0

  fig = plt.figure(figsize = (30, 20))
  a = fig.add_subplot(1, 2, 1)
  plt.imshow(img, cmap='gray')
  a.set_title('input')
  plt.axis('off')
  
  a = fig.add_subplot(1, 2, 2)
  plt.imshow(gt)
  a.set_title('sum {:.2f} -- ground thruth {:.0f}'.format(gt_cnt, len(lab)))
  plt.axis('off')

  fig.savefig("tmp.jpg", bbox_inches='tight')
  fig.clf()
  plt.close()
  del a
  gc.collect()

def plot_maps(origin_dir, output_dir):
  files = os.listdir(origin_dir)
  for file in files:
    if file.split('.')[-1] != 'npy':
      continue
    file_name = os.path.join(origin_dir, file)
    print(file_name)
    image = np.load(file_name)
    image = image / np.max(image) * 255
    file_out = os.path.join(output_dir, file.split('.')[0] + '.jpg')
    print(file_out)

    fig = plt.figure()
    a = fig.add_subplot(1, 1, 1)
    plt.imshow(image)
    plt.axis('off')

    fig.savefig(file_out, bbox_inches='tight')
    fig.clf()
    plt.close()
    del a
    gc.collect()

def plot_loss(log_file, out_file, out_options = '1'):
  f = open(log_file, "r")
  loss = []
  mae = []
  mse = []
  for line in f:
    if line.startswith("Epoch:"):
      for item in line.split(","):
        item = item.strip()
        num = float(item.split()[-1])
        if item.startswith("MAE"):
          mae.append(num)
        elif item.startswith("MSE"):
          mse.append(num)
        elif item.startswith("loss:"):
          loss.append(num) 
  assert len(loss) == len(mse) and len(mae) == len(mse), "Error in vector sizes mae: {}, mse: {}, loss: {}".format(len(mae), len(mse), len(loss))
  epoch = np.arange(len(loss))
  if out_options == '1' or out_options == '2':
    plt.plot(epoch, loss, label = 'loss')
  if out_options == '1' or out_options == '3':
    plt.plot(epoch, mae, label = 'mae')
    plt.plot(epoch, mse, label = 'mse')
  plt.xlabel("epoch")
  plt.legend(loc='upper left')
  plt.savefig(out_file)
  plt.clf()
  plt.close()
  gc.collect()

if __name__ == "__main__":
  #origin_dir = '/workspace/quispe/ucf_cc_50/UCF_CC_50/density_maps/'
  #output_dir = '/home/quispe/public_html/files/ucf_face_det_v3/'
  #plot_maps(origin_dir, output_dir)

  #plot_img_gt()

  log_file = 'log/ACSCP/ucf-cc-50_people_thr_20_gt_mode_same/ucf-fold1/log_train.txt'
  out_file = 'log/ACSCP/ucf-cc-50_people_thr_20_gt_mode_same/ucf-fold1/log_train_loss_mae_mse.png'
  plot_loss(log_file, out_file, out_options = '1')
