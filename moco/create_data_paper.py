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
object_name = 'curved_view1'
sensor = importlib.import_module('sensors.{}.sensor_params'.format(sensor_name))
list_images = glob('/home/mcube/tactile_localization/data_tactile_localization/data_paper/{}/depth_clean/*true_LS*npy'.format(object_name))
list_images2 = glob('/home/mcube/tactile_localization/data_tactile_localization/data_paper/{}/depth_clean/*predicted_LS*npy'.format(object_name))

sensor = importlib.import_module('sensors.{}.sensor_params'.format(sensor_name))
object_3D = Object3D(object_name, sensor, False, False)

# Change detectron2
#aa = glob.glob('*npy')
#for a in aa:   
#cv2.imwrite(a.replace('.npy','.png'), (1-np.load(a))*255)
for item in list_images:
        
    png_item = item.replace('.npy','.png')
    if not os.path.exists(png_item):
        transformation = np.load(item.replace('ed_true_LS', "ed_trans"))
        
        transformation[2,3] *= -1 
        ls, transformation_real = object_3D.renderTransformation(Transformation(transformation))
        
        if ls is None:
            print('No LS for item:', item)
            continue
        else: 
            ls.toPNG(sensor)
            cv2.imwrite(png_item, ls.ls)
            print('Done:', item)


for item in list_images2:
    
    real_ls = LocalShape(np.load(item))
    png_item = item.replace('.npy','.png')
    if not os.path.exists(png_item):
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
