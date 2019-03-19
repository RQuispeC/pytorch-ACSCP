import numpy as np
import cv2
import json, codecs
from random import *
from random import randint
import scipy.io as sio
import os
import os.path as osp
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import sys

from manage_data.utils import join_json, resize

def is_valid(X, Y, img):
	return X>=0 and X<img.shape[1] and Y>=0 and Y<img.shape[0]

def json_to_string(labels):
	if len(labels)==0:
		return '[]'
	line = '['
	for i in range(len(labels)):
		line += '{\"x\":'+str(labels[i]['x'])+',\"y\":'+str(labels[i]['y'])+'},'
	return line[0:len(line)-1]+']'

def count_people(heads,labels,position, x_low, x_upper , y_low, y_upper):
	cont =0 
	new_labels = []
	for x in range(x_low, x_upper):
		for y in range(y_low, y_upper):
			if heads[y, x] == 1:
				cont += 1
				auxLabel = labels[int(position[y, x])].copy()
				auxLabel['y'] -= y_low
				auxLabel['x'] -= x_low
				new_labels.append(auxLabel)
	return cont,new_labels

def add_noise(image, noise_type = 's&p'):
	if noise_type == "gauss":
		row,col= image.shape
		mean = 0
		var = 0.1
		sigma = var**0.5
		gauss = np.random.normal(mean,sigma,(row,col))
		gauss = gauss.reshape(row,col)
		noisy = image + gauss
		return noisy
	elif noise_type == "s&p":
		row,col = image.shape
		s_vs_p = 0.5
		amount = 0.04
		out = np.copy(image)
		# Salt mode
		num_salt = np.ceil(amount * image.size * s_vs_p)
		coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
		out[coords] = 1

		# Pepper mode
		num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
		coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
		out[coords] = 0
		return out
	else:
		raise RuntimeError("'{}' is not a valid noise type".format(noise_type))

def noise_augmentation(out_img_dir, out_lab_dir, out_den_dir, img_id):
	file_names = os.listdir(out_img_dir)
	for file_name in file_names:
		file_extension = file_name.split('.')[-1]
		if file_extension == 'jpg':
			img_id += 1
			in_img_path = osp.join(out_img_dir, file_name)
			in_lab_path = osp.join(out_lab_dir, file_name[:len(file_name) - len(file_extension)] + 'json')
			in_den_path = osp.join(out_den_dir, file_name[:len(file_name) - len(file_extension)] + 'npy')
			img = cv2.imread(in_img_path, 0)
			new_den = np.load(in_den_path)
			label = json.load(open(in_lab_path))
			if img_id % 2 == 0:
				img = add_noise(img, 'gauss')
			else:
				img = add_noise(img, 's&p')
			out_img_path = osp.join(out_img_dir, str(img_id).zfill(7) + '.jpg')
			out_lab_path = osp.join(out_lab_dir, str(img_id).zfill(7) + '.json')
			out_den_path = osp.join(out_den_dir, str(img_id).zfill(7) + '.npy')
			np.save(out_den_path, new_den)
			label = json_to_string(label)
			cv2.imwrite(out_img_path, img)
			with open(out_lab_path, 'w') as outfile:
				outfile.write(label)
	return img_id

def bright_contrast_augmentation(out_img_dir, out_lab_dir, out_den_dir, bright = 10, contrast = -50.0, img_id = 1):
	file_names = os.listdir(out_img_dir)
	alpha = 1.25
	beta = contrast
	for file_name in file_names:
		file_extension = file_name.split('.')[-1]
		if file_extension == 'jpg':
			img_id += 1
			in_img_path = osp.join(out_img_dir, file_name)
			in_lab_path = osp.join(out_lab_dir, file_name[:len(file_name) - len(file_extension)] + 'json')
			in_den_path = osp.join(out_den_dir, file_name[:len(file_name) - len(file_extension)] + 'npy')
			img = cv2.imread(in_img_path, 0)
			new_den = np.load(in_den_path)
			label = json.load(open(in_lab_path))
			if img_id % 2 == 0:
				img = img + bright
			else:
				img = img * alpha + beta
			out_img_path = osp.join(out_img_dir, str(img_id).zfill(7) + '.jpg')
			out_lab_path = osp.join(out_lab_dir, str(img_id).zfill(7) + '.json')
			out_den_path = osp.join(out_den_dir, str(img_id).zfill(7) + '.npy')
			np.save(out_den_path, new_den)
			cv2.imwrite(out_img_path, img)
			label = json_to_string(label)
			with open(out_lab_path, 'w') as outfile:
				outfile.write(label)
	return img_id

def sliding_window(out_img_dir, out_lab_dir, out_den_dir, img_id, img, labels, den, displace = 70, size_x=200, size_y=300, people_thr = 20):
	"""
	first config for data augmentation : displace =70, size_x = 200, size_y = 300, numberHeads =200
	second config for data augmentation : displace = 70, size_x = 250, size_y = 300, numberHeads =200 
	"""
	hasHeads = np.zeros(img.shape[:2])
	position = np.zeros(img.shape[:2])

	for i in range(len(labels)):
		if is_valid(int(labels[i]['x']), int(labels[i]['y']), img):
			hasHeads[int(labels[i]['y']), int(labels[i]['x'])] = 1
			position[int(labels[i]['y']), int(labels[i]['x'])] = i
		else:
			print("Warning: image '{}' has shape {}, and label is in position ({}, {})".format(img_id, img.shape, int(labels[i]['y']), int(labels[i]['x'])))

	for i in range(0, img.shape[0], displace): 
		for j in range(0, img.shape[1], displace):
			if is_valid(j + size_x, i + size_y, img):
				new_img = img[i:i+size_y, j:j+size_x]
				new_den = den[i:i+size_y, j:j+size_x]
				numberHeads_, new_labels = count_people(hasHeads, labels, position, j, j+size_x , i, i+size_y)
				if numberHeads_ >= people_thr:
					img_id += 1
					out_img_path = osp.join(out_img_dir, str(img_id).zfill(7) + '.jpg')
					out_lab_path = osp.join(out_lab_dir, str(img_id).zfill(7) + '.json')
					out_den_path = osp.join(out_den_dir, str(img_id).zfill(7) + '.npy')
					cv2.imwrite(out_img_path, new_img)
					np.save(out_den_path, new_den)
					data = str(json_to_string(new_labels))
					with open(out_lab_path, 'w') as outfile:
						outfile.write(data)
	return img_id

def augment(img_paths, label_paths, den_paths, out_img_dir, out_lab_dir, out_den_dir, slide_window_params, noise_params, light_params, add_original = False):
	print("Augmenting data, results will be stored in '{}'".format(out_img_dir))
	aug_img_id = 0
	for img_path, label_path, den_path in zip(img_paths, label_paths, den_paths):
		img = cv2.imread(img_path)
		den = np.load(den_path)
		label = json.load(open(label_path))
		
		#sliding window for data data augmentation
		aug_img_id = sliding_window(out_img_dir, out_lab_dir, out_den_dir, aug_img_id, img, label, den
		, displace = slide_window_params['displace']
		, size_x = slide_window_params['size_x']
		, size_y = slide_window_params['size_y']
		, people_thr = slide_window_params['people_thr'])

		if add_original:
			#add original to augmented set
			aug_img_id += 1
			out_img_path = osp.join(out_img_dir, str(aug_img_id).zfill(7) + '.jpg')
			out_lab_path = osp.join(out_lab_dir, str(aug_img_id).zfill(7) + '.json')
			out_den_path = osp.join(out_den_dir, str(aug_img_id).zfill(7) + '.npy')
			cv2.imwrite(out_img_path, img)
			npy.save(out_den_path, den)
			label = json_to_string(label)
			with open(out_lab_path, 'w') as outfile:
				outfile.write(label)

	#if slide_window_params['joint_patches']:
	#	join_patches(out_img_dir, out_lab_dir, slide_window_params['numberPatch'])
	if noise_params['augment_noise']:
		aug_img_id = noise_augmentation(out_img_dir, out_lab_dir, out_den_dir, aug_img_id)
	if light_params['augment_light']:
		aug_img_id = bright_contrast_augmentation(out_img_dir, out_lab_dir, out_den_dir, light_params['bright'], light_params['contrast'], aug_img_id)
	print("{} images created after augmentation".format(aug_img_id))


	


	
