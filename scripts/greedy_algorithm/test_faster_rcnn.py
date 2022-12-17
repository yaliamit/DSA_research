from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import sys
sys.path.insert(0, '../scripts/greedy_algorithm')
from functions import find_bb, imshow, imshow_color
from utils import iou
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision.utils import make_grid
from torchvision.utils import draw_bounding_boxes
import time
import numpy as np
import math
import scipy.io
import random
import os
import pandas as pd
import shutil
from skimage import io
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torchvision.models.detection import FasterRCNN
# from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
#from albumentations.pytorch import ToTensor
from torchvision import utils
from albumentations import (HorizontalFlip, ShiftScaleRotate, VerticalFlip, Normalize,Flip,
                            Compose, GaussNoise)
start = time.time()
from collections import Counter

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = 11 #10+1

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
print(in_features)

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath, map_location=device)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    return model, optimizer, checkpoint['epoch']

model, optimizer, checkpoint = load_ckp('./fasterrcnn_original_500', model, optimizer)

test_data, test_gt, test_boxes_gt = np.load('../data_and_model/test_data.npy')/255,np.load(
    '../data_and_model/test_gt.npy'),np.load('../data_and_model/test_boxes_gt.npy')
print(test_data.shape, test_gt.shape, test_boxes_gt.shape)


def check_cases(labels,scores,ground_truth):
    box_acc = []
    label_acc = []
    assert len(labels)==len(scores)
    l = len(labels)
    for i in range(l):
      if labels[i]==10:
        labels[i]=0
    for j in range(100):
      thres = j/100
      selected = [ ]
      for i in range(l):
        if scores[i]>thres:
          selected.append(labels[i])
      box_acc.append(1 if len(selected)==ground_truth[0] else 0)
      label_acc.append(1 if Counter(selected)==Counter(ground_truth[1:(1+ground_truth[0])]) else 0)
    return box_acc,label_acc


validnum, testnum = 500, 500
boxes_acc,labels_acc=[0 for _ in range(100)],[0 for _ in range(100)]
model.to(device)
model.eval()
for i in range(validnum):
    gt = test_gt[i,:]
    x = [torch.tensor(test_data[i,:].reshape(200,200,3).copy().transpose(2,0,1), dtype=torch.float32).to(device)]
    predictions = model(x)
    #print(len(predictions[0]['labels']),list(predictions[0]['labels'].detach().cpu().numpy())[:5], list(predictions[0]['scores'].detach().cpu().numpy())[:5], gt[1:(1+gt[0])])
    box_acc,label_acc=check_cases(list(predictions[0]['labels'].detach().cpu().numpy()),predictions[0]['scores'],gt)
    for j in range(100):
      boxes_acc[j]+=box_acc[j]/validnum
      labels_acc[j]+=label_acc[j]/validnum


# find the best thresholds based on the validation set results
m1, m2 = max(boxes_acc), max(labels_acc)
print('best boxes_acc,  best labels_acc:', m1, m2)
best_thres1, best_thres2 = boxes_acc.index(m1)/100, labels_acc.index(m2)/100
print('best_thres1, best_thres2:', best_thres1, best_thres2)


#begin testing
test_boxes_acc, test_label_acc = 0, 0
for i in range(validnum,validnum+testnum):
    gt = test_gt[i,:]
    x = [torch.tensor(test_data[i,:].reshape(200,200,3).copy().transpose(2,0,1), dtype=torch.float32).to(device)]
    predictions = model(x)
    labels,scores,ground_truth = list(predictions[0]['labels'].detach().cpu().numpy()),predictions[0]['scores'],gt
    assert len(labels)==len(scores)
    l = len(labels)
    for i in range(l):
      if labels[i]==10:
        labels[i]=0
    selected1, selected2 = [], []
    for i in range(l):
        if scores[i]>best_thres1:
            selected1.append(labels[i])
        if scores[i]>best_thres2:
            selected2.append(labels[i])
    test_boxes_acc += (1/testnum if len(selected1)==ground_truth[0] else 0)
    test_label_acc += (1/testnum if Counter(selected2)==Counter(ground_truth[1:(1+ground_truth[0])]) else 0)
print("Original NMS use threshold", best_thres1, best_thres2,". test_boxes_acc, test_label_acc:", test_boxes_acc, test_label_acc)
print('time cost:', time.time()-start)
print("*********")

# Soft-NMS
def soft_nms_Gaussian_penalty(iou1, sigma=0.5):
    return np.exp(-(iou1**2)/sigma)

def soft_nms(labels, scores, bbs):
    l_pred = len(labels)
    visited = set()
    for i in range(l_pred-1):
      # find the one with max score
      max_ind, max_score = -1, -1
      for j in range(l_pred):
        if j not in visited and scores[j]>max_score:
          max_ind, max_score = j, scores[j]
      visited.add(max_ind)
      # update the scores
      for j in range(l_pred):
        if j not in visited:
          iou_j = iou(bb[max_ind], bb[j])
          Gaussian_penalty = soft_nms_Gaussian_penalty(iou_j)
          scores[j] = scores[j]*Gaussian_penalty
    return labels, scores


validnum, testnum = 500, 500
boxes_acc,labels_acc=[0 for _ in range(100)],[0 for _ in range(100)]
print('model.roi_heads.nms_thresh',model.roi_heads.nms_thresh)
model.roi_heads.nms_thresh = 1.0
print('model.roi_heads.nms_thresh',model.roi_heads.nms_thresh)
model.to(device)
model.eval()
for i in range(validnum):
    gt = test_gt[i,:]
    x = [torch.tensor(test_data[i,:].reshape(200,200,3).copy().transpose(2,0,1), dtype=torch.float32).to(device)]
    predictions = model(x)
    #print(len(predictions[0]['labels']),list(predictions[0]['labels'].detach().cpu().numpy())[:8], list(predictions[0]['scores'].detach().cpu().numpy())[:8], gt[1:(1+gt[0])])
    # Soft-NMS
    labels = list(predictions[0]['labels'].detach().cpu().numpy())
    scores = predictions[0]['scores']
    bb = predictions[0]['boxes'].cpu().detach().numpy().astype(int)
    labels, scores = soft_nms(labels, scores, bb)
    #calculate accuracy
    box_acc,label_acc=check_cases(labels,scores,gt)
    for j in range(100):
      boxes_acc[j]+=box_acc[j]/validnum
      labels_acc[j]+=label_acc[j]/validnum

# find the best thresholds based on the validation set results
m1, m2 = max(boxes_acc), max(labels_acc)
print('best boxes_acc,  best labels_acc:', m1, m2)
best_thres1, best_thres2 = boxes_acc.index(m1)/100, labels_acc.index(m2)/100
print('best_thres1, best_thres2:', best_thres1, best_thres2)

#begin testing
test_boxes_acc, test_label_acc = 0, 0
for i in range(validnum,validnum+testnum):
    gt = test_gt[i,:]
    x = [torch.tensor(test_data[i,:].reshape(200,200,3).copy().transpose(2,0,1), dtype=torch.float32).to(device)]
    predictions = model(x)

    # Soft-NMS
    labels,scores,ground_truth = list(predictions[0]['labels'].detach().cpu().numpy()),predictions[0]['scores'],gt
    bb = predictions[0]['boxes'].cpu().detach().numpy().astype(int)
    labels, scores = soft_nms(labels, scores, bb)

    #calculate accuracy
    assert len(labels)==len(scores)
    l = len(labels)
    for i in range(l):
      if labels[i]==10:
        labels[i]=0
    selected1, selected2 = [], []
    for i in range(l):
        if scores[i]>best_thres1:
            selected1.append(labels[i])
        if scores[i]>best_thres2:
            selected2.append(labels[i])
    test_boxes_acc += (1/testnum if len(selected1)==ground_truth[0] else 0)
    test_label_acc += (1/testnum if Counter(selected2)==Counter(ground_truth[1:(1+ground_truth[0])]) else 0)
print("Soft-NMS use threshold", best_thres1, best_thres2,". test_boxes_acc, test_label_acc:", test_boxes_acc, test_label_acc)
print('time cost:', time.time()-start)
print("*********")

# DIoU-NMS

def DIoU_metric(bb1, bb2):
    max1, max2, min1, min2 = max(bb1[2],bb2[2]), max(bb1[3],bb2[3]), min(bb1[0],bb2[0]), min(bb1[1],bb2[1])
    center1, center2 = (float(bb1[0]+bb1[2]-1.0)/2.0,float(bb1[1]+bb1[3]-1.0)/2.0), (float(bb2[0]+bb2[2]-1.0)/2.0,float(bb2[1]+bb2[3]-1.0)/2.0)
    return iou(bb1, bb2) - ((center1[0]-center2[0])**2+(center1[1]-center2[1])**2)/((max1-min1)**2+(max2-min2)**2)

def DIoU_NMS(labels, scores, bbs, epsilon=0.5):
    l_pred = len(labels)
    visited = set()
    for i in range(l_pred-1):
      # find the one with max score
      max_ind, max_score = -1, -1
      for j in range(l_pred):
        if j not in visited and scores[j]>max_score:
          max_ind, max_score = j, scores[j]
      visited.add(max_ind)
      # update the scores
      for j in range(l_pred):
        if j not in visited:
          DIoU_metric_j = DIoU_metric(bb[max_ind], bb[j])
          if DIoU_metric_j>=epsilon:
            scores[j] = 0.0
    return labels, scores

validnum, testnum = 500, 500
boxes_acc,labels_acc=[0 for _ in range(100)],[0 for _ in range(100)]
print('model.roi_heads.nms_thresh',model.roi_heads.nms_thresh)
model.roi_heads.nms_thresh = 1.0
print('model.roi_heads.nms_thresh',model.roi_heads.nms_thresh)
model.to(device)
model.eval()
for i in range(validnum):
    gt = test_gt[i,:]
    x = [torch.tensor(test_data[i,:].reshape(200,200,3).copy().transpose(2,0,1), dtype=torch.float32).to(device)]
    predictions = model(x)
    #print(len(predictions[0]['labels']),list(predictions[0]['labels'].detach().cpu().numpy())[:8], list(predictions[0]['scores'].detach().cpu().numpy())[:8], gt[1:(1+gt[0])])
    # Soft-NMS
    labels = list(predictions[0]['labels'].detach().cpu().numpy())
    scores = predictions[0]['scores']
    bb = predictions[0]['boxes'].cpu().detach().numpy().astype(int)
    labels, scores = DIoU_NMS(labels, scores, bb)
    #calculate accuracy
    box_acc,label_acc=check_cases(labels,scores,gt)
    for j in range(100):
      boxes_acc[j]+=box_acc[j]/validnum
      labels_acc[j]+=label_acc[j]/validnum

# find the best thresholds based on the validation set results
m1, m2 = max(boxes_acc), max(labels_acc)
print('best boxes_acc,  best labels_acc:', m1, m2)
best_thres1, best_thres2 = boxes_acc.index(m1)/100, labels_acc.index(m2)/100
print('best_thres1, best_thres2:', best_thres1, best_thres2)


#begin testing
test_boxes_acc, test_label_acc = 0, 0
for i in range(validnum,validnum+testnum):
    gt = test_gt[i,:]
    x = [torch.tensor(test_data[i,:].reshape(200,200,3).copy().transpose(2,0,1), dtype=torch.float32).to(device)]
    predictions = model(x)

    # DIoU_NMS
    labels,scores,ground_truth = list(predictions[0]['labels'].detach().cpu().numpy()),predictions[0]['scores'],gt
    bb = predictions[0]['boxes'].cpu().detach().numpy().astype(int)
    labels, scores = DIoU_NMS(labels, scores, bb)

    #calculate accuracy
    assert len(labels)==len(scores)
    l = len(labels)
    for i in range(l):
      if labels[i]==10:
        labels[i]=0
    selected1, selected2 = [], []
    for i in range(l):
        if scores[i]>best_thres1:
            selected1.append(labels[i])
        if scores[i]>best_thres2:
            selected2.append(labels[i])
    test_boxes_acc += (1/testnum if len(selected1)==ground_truth[0] else 0)
    test_label_acc += (1/testnum if Counter(selected2)==Counter(ground_truth[1:(1+ground_truth[0])]) else 0)
print("DIoU_NMS use threshold ", best_thres1, best_thres2,". test_boxes_acc, test_label_acc:", test_boxes_acc, test_label_acc)

print('time cost:', time.time()-start)
