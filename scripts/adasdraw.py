#!/usr/bin/python
# coding: utf-8

import re
import os
import pdb
import glob
import time
from PIL import Image
import cv2
from adas_utils import recursive_get_images

#DATA_DIR = "/home/data4t/zs/valid_800_v4/"
DATA_DIR = [
"/home/data4t/zs/tired_mooncake/pictures",
"/home/data4t/zs/tired_mooncake/pictures1",
"/home/data4t/zs/tired_mooncake/pictures2",
"/home/data4t/zs/tired_mooncake/pictures3",
]

OUTPUT_DIR = "tired_mooncake"

class ADASDraw(object):
    """
    """
    def __init__(self, data_dirs=DATA_DIR, output_dir=OUTPUT_DIR):
        self.data_dirs = data_dirs
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.output_dir = output_dir

    def parse_label(self, image_path):
        label_path = re.sub(r'\.(jpg|jpeg|png|bmp)$', r'.txt', image_path, re.I)

        imsize = None
        try:
            imsize = Image.open(image_path).size
        except IOError as e:
            print("{0} open exception, {1}".format(image_path, e))
            return None

        boxes = []
        for l in open(label_path):
            l = l.strip()
            if not l:
                continue


            label = l.split(',')
            cls = label[0]
            (x, y, w, h) = [int(i) for i in label[2:6]]

            if x < 0:
                x = 0
            if x+w >= imsize[0]:
                w = imsize[0] - x
            if y < 0:
                y = 0
            if y+h >= imsize[1]:
                h = imsize[1] - y
 
            boxes.append([cls, x, y, w, h])

        return None if len(boxes) == 0 else boxes
      
    def run(self):
        for d in self.data_dirs:
            image_paths = recursive_get_images(d, pattern = r'\.jpg$')
            for image_path in image_paths:
                boxes = self.parse_label(image_path)
                if not boxes:
                    print("{} no valid boxes".format(image_path))
                    continue
                img = self.draw(image_path, boxes)
                image_name = os.path.basename(image_path)
                cv2.imwrite(os.path.join(self.output_dir, image_name), img) 

    def draw(self, image_path, annotations):
        img = cv2.imread(image_path)
        for (cls, x_lt, y_lt, w, h) in annotations:
            x_rb = x_lt + w
            y_rb = y_lt + h
            cv2.rectangle(img, (x_lt,y_lt), (x_rb,y_rb), (0,255,0), 1) 
            cv2.putText(img, cls, (x_lt,y_lt-6), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 255, 0))
        return img


if __name__ == "__main__":
    d = ADASDraw()
    d.run()
 
