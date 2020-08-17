import torch
from glob import glob
import numpy as np
import cv2
import os
import shutil
import sys
import importlib

sys.path.append('/home/mcube/tactile_localization/')

sensor_name = 'green_sensor'
object_name = 'pin_view2'
sensor = importlib.import_module('sensors.{}.sensor_params'.format(sensor_name))
list_images = glob('/home/mcube/tactile_localization/data_tactile_localization/data_paper/{}/depth_clean/*true_LS*npy'.format(object_name))

for item in list_images:
        
    png_item = item.replace('.npy','.png')
    if not os.path.exists(png_item):
        ls = np.load(item)
        ls = (ls - sensor.DESIRED_DIST)/(sensor.MAX_VAL_DEPTH - sensor.DESIRED_DIST)*255
        print(png_item)
        cv2.imwrite(png_item, ls)
