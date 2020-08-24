import torch
import glob
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
object_name = 'head_view1'
grid_name = 'face1'
sensor = importlib.import_module('sensors.{}.sensor_params'.format(sensor_name))
list_images = glob.glob('/home/mcube/tactile_localization/data_tactile_localization/data_paper/{}/depth_clean/*true_LS*npy'.format(object_name))
list_images2 = glob.glob('/home/mcube/tactile_localization/data_tactile_localization/data_paper/{}/depth_clean/*predicted_LS*npy'.format(object_name))

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


### Add somethinng that computtes closest in grid

print('Compputing closest')

sys.path.append('/home/mcube/tactile_localization/')
from tactile_localization.constants import constants
from tactile_localization.classes.grid import Grid2D, Grid3D
from tactile_localization.classes.object_manipulator import Object3D
from tactile_localization.classes.local_shape import LocalShape, Transformation



#sensor_name = np.load('moco/sensor_name.npy')

#sensor = importlib.import_module('sensors.{}.sensor_params'.format(sensor_name))

#object_name = np.load('moco/object_name.npy')
#grid_name = np.load('moco/grid_name.npy')  


print('Generate data from:', sensor_name, object_name, grid_name)

i = np.copy(grid_name); configs = []
grid_aux = Grid2D(object_name, sensor, i)
configs += [grid_aux.loadConfiguration()]
if configs[-1]['grid_3D']:
    object_3D = Object3D(object_name, sensor, False, True)
else:
    object_3D = Object3D(object_name, sensor, False, False)


path_data = 'data/{}_{}/train/'.format(object_name, grid_name)
list_trans = glob.glob(path_data + '*/transformation.npy')
all_trans = []
for path in list_trans:
    trans = np.load(path)
    #trans[:3,:3] = trans[:3,:3].T
    all_trans.append(trans)
all_trans = np.array(all_trans)



def compute_closest(trans):
    error = []
    
    trans[2,3] *=-1
    for it, all_tran in enumerate(all_trans):
        
        points1 = np.dot(object_3D.pcd_points_no_centering,trans[:3,:3].T) + trans[:3,3]
        
        points2 = np.dot(object_3D.pcd_points_no_centering,all_tran[:3,:3].T) + all_tran[:3,3]
        #points2 = np.dot(object_3D.pcd_points_no_centering,all_trans[:,:3,:3]) + all_trans[:,:3,3]
        
        err = np.mean(np.sqrt(np.sum((points1 - points2)**2, axis=-1)), axis=-1)
        error.append(err)
    
        
    it = np.argmin(error)
    return error[it], all_trans[it]



errors = []
for item in list_images2:
    
    path_trans = item.replace('ed_LS', 'ed_trans')
    path_closest = item.replace('ed_LS', 'ed_closest_trans')
    path_dist_closest = item.replace('ed_LS', 'ed_dist_closest_trans')
    if not os.path.exists(path_closest):
        try:
            transformation_real = Transformation(np.load(path_trans))
            if 'head' in object_name:
                orig_pos = np.dot(-transformation_real.trans[:3,3], transformation_real.trans[:3,:3])
                if orig_pos[0] > 0:
                    continue
        except:
            print("nop")
        
        
        error, trans = compute_closest(transformation_real.trans)
        errors.append(error)
        print('Error:', error, np.median(errors), np.mean(errors))
        np.save(path_closest, trans)
        np.save(path_dist_closest, error)
        #break
