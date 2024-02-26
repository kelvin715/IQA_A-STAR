import cv2
import numpy as np


import sys
import os
def create_dataset_with_adjust_brightness(brightness=0, base="NEU-DET"):
    # mkdir
    name = f"{base}_GaussianBlur_{brightness}"
    path = os.path.join('/Data4/student_zhihan_data/data', name)
    if not os.path.exists(path):
        os.mkdir(path)
        # create train, val, test directory
        os.mkdir(os.path.join(path, 'train'))
        os.mkdir(os.path.join(path, 'valid'))
        os.mkdir(os.path.join(path, 'test'))
        # copy images
        os.system(f"cp -r /Data4/student_zhihan_data/data/{base}/train/* {path}/train")
        os.system(f"cp -r /Data4/student_zhihan_data/data/{base}/valid/* {path}/valid")
        os.system(f"cp -r /Data4/student_zhihan_data/data/{base}/test/* {path}/test")
        # copy configure file
        os.system(f"cp /Data4/student_zhihan_data/data/{base}/data.yaml {path}")

    # adjust brightness of images in train, val, test directory
    for dir in ['train', 'valid', 'test']:
        for img in os.listdir(os.path.join(path, dir, "images")):
            img_path = os.path.join(path, dir, "images", img)
            img = cv2.imread(img_path)
            img = cv2.GaussianBlur(img, ksize=(brightness, brightness), sigmaX=0, sigmaY=0) 
            cv2.imwrite(img_path, img)
            
for i in [13, 15, 17]:
    create_dataset_with_adjust_brightness(brightness=i , base="GC10-DET")