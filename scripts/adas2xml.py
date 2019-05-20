#!/usr/bin/python
# coding: utf-8

import re
import os
import pdb
import glob
import time
import pprint
from PIL import Image
from xml.dom.minidom import parseString
from lxml.etree import Element, SubElement, tostring
from adas_utils import recursive_get_images
from datetime import datetime as dt

'''
疲劳检测
标签,类型
脸,     c
手机,   f
手机,   p
手机+手,w
烟,     q
烟+手,  s
食物/水,g
疑似,   o
Untag,  u?
---------
汽车数据集
p，f：普通车、轿车suv、公共汽车
w：异型车
q：行人
s：骑行
g：车挡
o：人挡
c：车头或其他
'''
G_LABEL_MAP = {
    'adas_car': {
        'ignore': ['q','s','g','o','c'],
        'multi_class': False,
        'default': 'car'
    },
    'adas': {
        'ignore': [],
        'multi_class': True,
        'c':'f',
        'g':'f',
        'p':'f',
        's':'q',
        'o':'q',
    },
    'adas_dms': {
        'ignore': ['u','q'],
        'multi_class': True,
        'p': 'w',
        'f': 'w',
        'l': 'c',
    }
}

class Adas2XML(object):
    """
    """
    def __init__(self, folder, dataset_type, store_path):
        self.folder = folder 
        self.store_path = store_path
        self._init_label_map(dataset_type)

    def _init_label_map(self, dataset_type):
        if G_LABEL_MAP.get(dataset_type):
            self.label_map = G_LABEL_MAP[dataset_type]
        else:
            raise Exception("invalid dataset_type:{}".format(dataset_type))

    def parse_boxes(self, image_path):
        txt = re.sub(r'\.(jpg|jpeg|png|bmp)$', r'.txt', image_path, re.I)

        imsize = None
        try:
            imsize = Image.open(image_path).size
        except IOError as e:
            LOG.error("{0} open exception, {1}".format(image_path, e))
            return None

        boxs = []
        for l in open(txt):
            l = l.strip()
            if not l:
                continue
           
            label = l.split(',')
            if label[0] in self.label_map['ignore']:
                continue
            elif self.label_map['multi_class']:
                if label[0] in self.label_map:
                    name = self.label_map[label[0]]
                else:
                    name = label[0]
            else:
                name = self.label_map['default']

            box = [int(i) for i in label[2:6]]
            (x, y, w, h) = box
            if x < 0:
                x = 0
            if x+w >= imsize[0]:
                w = imsize[0] - x
            if y < 0:
                y = 0
            if y+h >= imsize[1]:
                h = imsize[1] - y
            
            bndbox = {
                'name': name,
                'xmin': str(x + 1),
                'ymin': str(y + 1),
                'xmax': str(x + w + 1),
                'ymax': str(y + h + 1)    
            } 

            boxs.append(bndbox)

        return None if len(boxs) == 0 else boxs

    def convert(self, image_path):
        boxs = self.parse_boxes(image_path)
        if not boxs:
            LOG.info("{} no valid boxes".format(image_path))
            return None

        basename = os.path.basename(image_path)    
              
        root = Element('annotation')
        folder = SubElement(root, 'folder')
        folder.text = self.folder
     
        filename = SubElement(root, 'filename')
        filename.text = basename
       
        for box in boxs:
            self._add_object(root, box)
      
        return self._xml(root)
    
    def _add_size(self):
        pass
        
    def _add_owner(self):
        pass
    
    def _add_source(self):
        pass
                     
    def _add_object(self, root, box):
        object = SubElement(root, 'object')
        name = SubElement(object, 'name')
        name.text = box['name']

        bndbox = SubElement(object, 'bndbox')
        xmin = SubElement(bndbox, 'xmin')
        xmin.text = box['xmin']
        
        ymin = SubElement(bndbox, 'ymin')
        ymin.text = box['ymin']
        
        xmax = SubElement(bndbox, 'xmax')
        xmax.text = box['xmax']
        
        ymax = SubElement(bndbox, 'ymax')
        ymax.text = box['ymax']

    def _xml(self, root):
        """
        format string
        """
        return tostring(root, pretty_print=True)  

def test():
    c = Adas2XML("ADAS2017")
    image_path = '~/faster-rcnn.pytorch/data/ADASdevkit2017/ADAS2017/JPEGImages/img_final_train/day/backlight/day_backlight_1703_01_5000/2016_1128_135148_003.MOV_0003.jpg'
    xml = c.convert(image_path)
    print(xml)

import logging
LOG = logging.getLogger(__name__)
LOG.setLevel(level = logging.INFO)
handler = logging.FileHandler("run.{}.log".format(dt.now().strftime('%m%d%H%M')))
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
LOG.addHandler(handler)

def main(input_config):
   # get the input config data
    _config = {}
    exec(open(input_config).read(),None, _config)
    config = type('Config', (), _config)

    datadict = config.data
  
    c = Adas2XML("ADAS2017", "adas_dms", config.store_path)

    for key, data in datadict.items():
        if len(data['image_dirs']) == 0:
            continue

        imageset_path = os.path.join(
                config.store_path, 
                "ImageSets", 
                "Main", 
                key+".txt") 
        
        with open(imageset_path, 'w') as fset:
            for dir in data['image_dirs']:
                image_dir = os.path.join(config.data_path, dir)
                prefix = "{}_{}".format(key, dir)
                jpegimages_dir = os.path.join(
                        config.store_path, 
                        "JPEGImages", 
                        prefix)
                if not os.path.exists(jpegimages_dir):
                    os.symlink(image_dir, jpegimages_dir)

                image_paths = recursive_get_images(image_dir)
                for image_path in image_paths:
                    xml = c.convert(image_path)
                    if not xml:
                        #LOG.info("{} convert none".format(image_path))
                        continue

                    image_index = re.sub(r'\.jpg', r'', image_path.replace(config.data_path, ''), re.I)
                    image_index = image_index.replace(dir, prefix, 1)

                    xml_path = os.path.join(
                        config.store_path, 
                        'Annotations', 
                        image_index + '.xml')     
                    xml_dir = os.path.dirname(xml_path)
                    if not os.path.exists(xml_dir):
                        os.makedirs(xml_dir)
                    with open(xml_path, 'w') as fxml:
                        fxml.write(xml)
                        
                    fset.write(image_index + '\n')

 
if __name__ == "__main__":
    main("input_config.py")
