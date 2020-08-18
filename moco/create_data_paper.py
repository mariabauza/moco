import torch
from glob import glob
import numpy as np
import cv2
import os
import shutil
import sys
import importlib

sys.path.append('/home/mcube/tactile_localization/')
from tactile_localization.constants import constants
from tactile_localization.classes.grid import Grid2D, Grid3D
from tactile_localization.classes.object_manipulator import Object3D
from tactile_localization.classes.local_shape import LocalShape, Transformation


sensor_name = 'green_sensor'
object_name = 'pin_view2'
sensor = importlib.import_module('sensors.{}.sensor_params'.format(sensor_name))
list_images = glob('/home/mcube/tactile_localization/data_tactile_localization/data_paper/{}/depth_clean/*true_LS*npy'.format(object_name))
list_images2 = glob('/home/mcube/tactile_localization/data_tactile_localization/data_paper/{}/depth_clean/*predicted_LS*npy'.format(object_name))

for item in list_images:
        
    png_item = item.replace('.npy','.png')
    if not os.path.exists(png_item):
        ls = np.load(item)
        ls = (ls - sensor.DESIRED_DIST)/(sensor.MAX_VAL_DEPTH - sensor.DESIRED_DIST)*255
        print(png_item)
        cv2.imwrite(png_item, ls)


for item in list_images2:
    
    real_ls = LocalShape(np.load(item))
    png_item = item.replace('.npy','.png')
    if 1 or not os.path.exists(png_item):
        try:
            transformation_real = Transformation(np.load(item.replace('ed_LS', 'ed_trans')))
            if 'head' in object_name:
                orig_pos = np.dot(-transformation_real.trans[:3,3], transformation_real.trans[:3,:3])
                if orig_pos[0] > 0:
                    continue
        except:
            print("nop")
    
        real_ls.realToBin(0.0006, sensor, True)
        cv2.imwrite(png_item, (real_ls.ls < 1)*255)
