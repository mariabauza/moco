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
object_name = 'grease_view1'
grid_name = 'final_grid'
sensor = importlib.import_module('sensors.{}.sensor_params'.format(sensor_name))
print(os.environ['HOME'] + '/tactile_localization/data_tactile_localization/{}/{}/grids/{}/transformation*4.npy'.format(sensor_name, object_name, grid_name))
list_images = glob.glob(os.environ['HOME'] + '/tactile_localization/data_tactile_localization/{}/{}/grids/{}/transformation*4.npy'.format(sensor_name, object_name, grid_name))
list_images.sort(key=os.path.getmtime)
list_images2 = glob.glob('/home/mcube/tactile_localization/data_tactile_localization/data_paper/{}/depth_clean/*predicted_LS*npy'.format(object_name))
list_images2.sort(key=os.path.getmtime)
print('How many images:', len(list_images), len(list_images2))
sensor = importlib.import_module('sensors.{}.sensor_params'.format(sensor_name))
object_3D = Object3D(object_name, sensor, False, False)

# Change detectron2
#aa = glob.glob('*npy')
#for a in aa:   
#cv2.imwrite(a.replace('.npy','.png'), (1-np.load(a))*255)

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


list_trans = glob.glob(os.environ['HOME'] + '/tactile_localization/data_tactile_localization/{}/{}/grids/{}/transformation*1.npy'.format(sensor_name, object_name, grid_name))
list_trans.sort(key=os.path.getmtime)
all_trans = []
for path in list_trans:
    trans = np.load(path)
    #trans[:3,:3] = trans[:3,:3].T
    all_trans.append(trans)
all_trans = np.array(all_trans)



def compute_closest(trans, from_grid = False):
    error = []
    
    if not from_grid:
        trans[2,3] *=-1
    for it, all_tran in enumerate(all_trans):
        
        trans[2,3]  *= 0
        all_tran[2,3]  *= 0
        
        points1 = np.dot(object_3D.pcd_points_no_centering,trans[:3,:3].T) + trans[:3,3]
        
        points2 = np.dot(object_3D.pcd_points_no_centering,all_tran[:3,:3].T) + all_tran[:3,3]
        #points2 = np.dot(object_3D.pcd_points_no_centering,all_trans[:,:3,:3]) + all_trans[:,:3,3]
        
        err = np.mean(np.sqrt(np.sum((points1 - points2)**2, axis=-1)), axis=-1)
        error.append(err)
    
        
    it = np.argmin(error)
    if from_grid:
        return error[it], all_trans[it], list_trans[it]
    return error[it], all_trans[it]



errors = []
if 0:
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
            
            
            error, trans = compute_closest(transformation_real.trans, from_grid = False)
            errors.append(error)
            
            print('Error:', error, np.median(errors), np.mean(errors))
            #np.save(path_closest, trans)
            #np.save(path_dist_closest, error)
            #break
            
    print('Grid errors:')


errors = []
counter  = 0
for it, item in enumerate(list_images):
    
    path_trans = item
    path_closest = item.replace('3.npy', '1.npy').replace('4.npy', '1.npy')
    
    
    
    try: transformation_real = Transformation(np.load(path_trans))
    except: 'Cant be loaded'; continue
    if 'head' in object_name:
        orig_pos = np.dot(-transformation_real.trans[:3,3], transformation_real.trans[:3,:3])
        if orig_pos[0] > 0:
            continue
        
        
    
    error, trans, actual_trans = compute_closest(transformation_real.trans, from_grid = True)
    errors.append(error)
    
    
    
    trans_true = np.load(path_closest)
    
    
    trans = np.copy(transformation_real.trans)
    all_tran = np.copy(trans_true)
    trans[2,3]  *= 0
    all_tran[2,3]  *= 0
    points1 = np.dot(object_3D.pcd_points_no_centering,trans[:3,:3].T) + trans[:3,3]        
    points2 = np.dot(object_3D.pcd_points_no_centering,all_tran[:3,:3].T) + all_tran[:3,3]
    #points2 = np.dot(object_3D.pcd_points_no_centering,all_trans[:,:3,:3]) + all_trans[:,:3,3]
    
    err = np.mean(np.sqrt(np.sum((points1 - points2)**2, axis=-1)), axis=-1)
    
    
    if path_closest!= actual_trans: 
        print('Fuckito', path_closest, actual_trans)
        counter += 1
    print('Error:', error, err, np.median(errors), np.mean(errors), counter, it)
    #np.save(path_closest, trans)
    #np.save(path_dist_closest, error)
    #break
