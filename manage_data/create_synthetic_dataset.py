"""
THIS SCRIPT HAS NOT BEEN TESTED AND MAY HAVE ERRORS
original author: @darwin
"""

import numpy as np
import cv2
import json, codecs
from random import *
from random import randint
import scipy.io as sio
import os
import os.path as osp
import matplotlib.pyplot as plt
import sys

from manage_data.utils import join_json, resize

people = []
background = []

labelPeople = []
labelBackground = []

dataset = []

def readImages(fileName = 'input/FAKE/elements/', scale =10):
    # read figures of people and background
	file_names = os.listdir(fileName)
	file_names.sort()
	for file_name in file_names:
		if file_name[len(file_name) - 3:] != 'png' and file_name[len(file_name) - 3:] != 'jpg':
			continue
		image_name = fileName + file_name 
		img = cv2.imread(image_name,0)
		if file_name[0]=='p':
			img = cv2.resize(img, (img.shape[1]//scale, img.shape[0]//scale))
			people.append(img)
		else :
			background.append(img)
    
def readJSON(fileName= '/JSON/', scale =10):
    # read labels(.json) of people and background
	file_names = os.listdir(fileName)
	file_names.sort()
	for file_name in file_names:
		if file_name[len(file_name) - 4:] != 'json':
			continue
		label_name = fileName + file_name
		if file_name[0]=='p':
			with open(label_name) as data_file:
				data = json.load(data_file)
				labelPeople.append(resize(data, scale))
		else :
			with open(label_name) as data_file:
				data = json.load(data_file)
				labelBackground.append(data)

def isValid(Y, X, img):
	return X>=0 and X<img.shape[1] and Y>=0 and Y<img.shape[0]
    
def joinFigures(imgBackground, imgPerson, head, beginX, beginY):
    labelHead = []
    new_background = imgBackground.copy()
    #join figures(brackground, person)
    for i in range(imgPerson.shape[0]):
        for j in range(imgPerson.shape[1]):
            new_x = beginY + i - imgPerson.shape[0]
            new_y = beginX + j
            if imgPerson[i][j] != 0:
                if isValid(new_x, new_y, imgBackground):
                    new_background[(int)(new_x)][(int)(new_y)]=imgPerson[i][j]
                        
    #find new position for heads
    for i in range(len(head)):
        new_y = head[i]['y'] + beginY - imgPerson.shape[0]
        new_x = head[i]['x'] + beginX
        if isValid(new_y , new_x , imgBackground):
            labelHead.append([{'x': new_x, 'y': new_y}])
                           
    return new_background, np.array(labelHead)

def createStringJSON(labels):
	if len(labels)==0:
		return '[]'
	line = '['
	for i in range(len(labels)):
		line += '{\"x\":'+str(labels[i][0]['x'])+',\"y\":'+str(labels[i][0]['y'])+'},'
	return line[0:len(line)-1]+']'

def countHeads(heads,labels,position, i, ii , j , jj):
    cont =0 
    new_labels = []
    for x in range(i,ii):
        for y in range(j,jj):
            if heads[x][y] == 1:
                cont += 1
                auxLabel = labels[int(position[x][y])].copy()
                auxLabel['y'] -= i
                auxLabel['x'] -= j
                new_labels.append(auxLabel)
    return cont,new_labels

def createDataset(fileNameOut = 'input/FAKE/', stepLen = 1.0, cnt =1):
	for (back, lines) in zip(background, labelBackground):
		labels = []
		print("Working on image ", str(cnt).zfill(4))
		for line in lines:
			direction = np.array([line['rx'] - line['lx'], line['ry'] - line['ly']])
			lineSize = np.linalg.norm(direction)
			direction /= lineSize
			sweepLen = 0
			noisyX = randint(-5, 5)
			noisyY = randint(-5, 5)
			while sweepLen < lineSize:
				personId = randint(0, len(people) - 1)
				back, label = joinFigures(back, people[personId], labelPeople[personId], line['lx'] + sweepLen*direction[0]+noisyX, line['ly'] + sweepLen*direction[1]+noisyY)
				sweepLen += stepLen
				for l in label:
					labels.append(l)
		if not os.path.exists('output/syntethicDataset'):
			os.makedirs('output/syntethicDataset')
		saveName = 'output/syntethicDataset/background' , str(cnt).zfill(4)    
		cv2.imwrite(saveName+ '.jpg', back)
		data = str(createStringJSON(labels))
		with open(saveName + '.json', 'w') as outfile:
			outfile.write(data)
		cnt += 1
	cnt = createPatchFromImages('output/syntethicDataset/',cnt)
	return cnt

def createSinteticDataset(fileName = 'input/FAKE/',scale=7, cnt=1):
	print("++++++++++ Creating Fake dataset +++++++++")	
	fileNameElements = fileName+'elements/'
	readImages(fileNameElements, scale = scale)
	readJSON(fileNameElements,scale = scale)
	for i in range(10,11,10):
		cnt = createDataset(fileName, stepLen = i, cnt=cnt)
	print("++++++++++ Fake dataset created +++++++++")	

def createSyntheticDataset(fileNameInput = 'input/FAKE/', fileNameOutput = 'output/synthetic/', scale=1, syntheticSamples = 23000, createParams = {}):
	print("Creating synthetic dataset")	
	fileNameElements = fileNameInput + 'elements/'
	readImages(fileNameElements, scale = scale)
	readJSON(fileNameElements,scale = scale)
	
	if not os.path.exists(fileNameOutput):
		os.makedirs(fileNameOutput)
		os.makedirs(fileNameOutput + 'original/')
		os.makedirs(fileNameOutput + 'label/')

	for i in range(syntheticSamples):
		print ('creating sample {} of {}'.format(i + 1, syntheticSamples))
		intersection = random()*(createParams['intersection'][1] - createParams['intersection'][0]) + createParams['intersection'][0]
		cameraPosition = randint(createParams['camera_position'][0], createParams['camera_position'][1])
		emptyLenght = createParams['empty_lenght']
		emptyNum = randint(createParams['empty_num'][0], createParams['empty_num'][1])
		upperBound = random()*(createParams['upper_bound'][1] - createParams['upper_bound'][0]) + createParams['upper_bound'][0]
		backgrnd = background[randint(0, len(background) - 1)].copy()
		print(upperBound)
		img, labels = syntheticData(backgrnd, intersection, cameraPosition, emptyLenght, emptyNum, upperBound)
		cv2.imwrite(fileNameOutput + 'original/' + str(i).zfill(5) + '.jpg', img)
		data = str(createStringJSON(labels))
		with open(fileNameOutput + 'label/' + str(i).zfill(5) + '.json', 'w') as outfile:
			outfile.write(data)

	print("Synthetic dataset created")

if __name__ == '__main__':
	#fileNameUCF = 'input/UCF_CC_50/'
	#UCF_50_dataAgmentation(fileNameUCF, fold = 3)

	syntheticFileNameInput = 'input/FAKE/'
	syntheticFileNameOutput = 'input/synthetic/'
	syntheticSamples = 23000
	syntheticParams = {'intersection': [0.0, 0.25], 'camera_position': [1, 3], 'empty_lenght': [0, 40], 
	'empty_num': [40, 1800], 'upper_bound': [0.45, 1]}
	createSyntheticDataset(fileNameInput = syntheticFileNameInput, fileNameOutput = syntheticFileNameOutput, 
		syntheticSamples = syntheticSamples, createParams = syntheticParams)
	