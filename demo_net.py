# coding: utf-8
# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import cPickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import glob
import torchvision.transforms as transforms
import torchvision.datasets as dset
from PIL import Image
from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir

from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import prep_im_for_blob,im_list_to_blob
import pdb
from model.fpn.detnet_backbone import detnet

import warnings
warnings.filterwarnings("ignore")

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('exp_name', type=str, default=None, help='experiment name')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='adas', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/detnet59.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='detnet59',
                      default='detnet59', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="data/models",
                      nargs=argparse.REMAINDER)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--video_file', dest='video_file',
                      help='video file',default='', 
                      type=str)
  parser.add_argument('--image_dir', dest='image_dir',
                      help='directory to load images', default="data/images",
                      type=str)  
  parser.add_argument('--ngpu', dest='ngpu',
                      help='number of gpu',
                      default=1, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10000, type=int)

  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im: data of image
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_scales = []
  processed_ims = []
  scale_inds = np.random.randint(0, high=len(cfg.TRAIN.SCALES), size=1)

  target_size = cfg.TRAIN.SCALES[scale_inds[0]]
  im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, cfg.PIXEL_STDS, target_size, cfg.TRAIN.MAX_SIZE)

  im_scales.append(im_scale)
  processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  args.cfg_file = "cfgs/{}.yml".format(args.net)
  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  #print('Using config:')
  #pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)
  cfg.TRAIN.USE_FLIPPED = False

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.

  if args.exp_name is not None:
    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset + '/' + args.exp_name
  else:
    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'fpn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

  classes = cfg.TRAIN.CLASSES
  fpn = detnet(classes, 59, pretrained=False, class_agnostic=args.class_agnostic)
  fpn.create_architecture()

  print('load checkpoint %s' % (load_name))
  checkpoint = torch.load(load_name)

  fpn.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.ngpu > 0:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data, volatile=True)
  im_info = Variable(im_info, volatile=True)
  num_boxes = Variable(num_boxes, volatile=True)
  gt_boxes = Variable(gt_boxes, volatile=True)

  if args.ngpu > 0:
    cfg.CUDA = True

  if args.ngpu > 0:
    fpn.cuda()

  fpn.eval()

  max_per_image = 100
  thresh = 0.05
  vis_thresh = 0.8
  vis = True
  
  if not os.path.exists(args.video_file):
      raise Exception("video %s not exist".format(args.video_file))

  vc = cv2.VideoCapture(args.video_file)
  i = 0
  while True: 
      i += 1
      _, im = vc.read()
      if im is None:
          break
      
      blobs, im_scales = get_image_blob(im)
      assert len(im_scales) == 1, "Only single-image batch implemented"
      im_blob = blobs
      # (h,w,scale)
      im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
      im_data_pt = torch.from_numpy(im_blob)
      # exchange dimension->(b,c,h,w)
      im_data_pt = im_data_pt.permute(0, 3, 1, 2)

      im_info_pt = torch.from_numpy(im_info_np)
      #im_info_pt = im_info_pt.view(3)

      im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
      im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
      gt_boxes.data.resize_(1, 1, 5).zero_()
      num_boxes.data.resize_(1).zero_()
    
      det_tic = time.time()
      rois, cls_prob, bbox_pred, \
          _, _, _, _, _ = fpn(im_data, im_info, gt_boxes, num_boxes)
      #pdb.set_trace()

      scores = cls_prob.data
      boxes = rois.data[:, :, 1:5] 

      if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * len(classes))
            
            #pdb.set_trace()
            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))

      pred_boxes /= im_scales[0]

      scores = scores.squeeze()
      pred_boxes = pred_boxes.squeeze()
      det_toc = time.time()
      detect_time = det_toc - det_tic

      if vis:
          im2show = np.copy(im)

      sys.stdout.write('im_detect: {:d} {:.3f}s   \r'.format(i, detect_time))
      sys.stdout.flush()

      for j in xrange(1, len(classes)): # 0 for background
          inds = torch.nonzero(scores[:,j] > thresh).view(-1) 
          # if there is det
          if inds.numel() > 0: 
            cls_scores = scores[:,j][inds] # confidence of the specified class
            _, order = torch.sort(cls_scores, 0, True) # sorted scores and indexes
            if args.class_agnostic:
              cls_boxes = pred_boxes[inds, :] 
            else:
              cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
           
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_dets, cfg.TEST.NMS) # after nms
            cls_dets = cls_dets[keep.view(-1).long()] # keep shape is ?x1
            if vis: 
              # cls_dets.cpu().numpy() make tensor->numpy array
              im2show = vis_detections(im2show, classes[j], cls_dets.cpu().numpy(), vis_thresh)
              #drawpath = os.path.join('images', "{}.jpg".format(i))
              #cv2.imwrite(drawpath, im2show)
      cv2.imshow('demo', im2show)
      if (cv2.waitKey(25) & 0xFF) == ord('q'):
          break
  vc.release()
  cv2.destroyAllWindows()
