import torch
import glob
import numpy as np
import cv2
import os
import shutil
import sys
import importlib
import matplotlib.pyplot as plt
import time

sys.path.append('/home/mcube/tactile_localization/')
from tactile_localization.constants import constants
from tactile_localization.classes.grid import Grid2D, Grid3D
from tactile_localization.classes.object_manipulator import Object3D
from tactile_localization.classes.local_shape import LocalShape, Transformation


sensor_name = 'green_sensor'
object_name = 'grease_view1'
object_name = 'pin_view2'
grid_name = 'final'
sensor = importlib.import_module('sensors.{}.sensor_params'.format(sensor_name))
list_images = glob.glob('/home/mcube/tactile_localization/data_tactile_localization/data_paper/{}/depth_clean/*true_LS*npy'.format(object_name))
list_images.sort(key=os.path.getmtime)
list_images2 = glob.glob('/home/mcube/tactile_localization/data_tactile_localization/data_paper/{}/depth_clean/*predicted_LS_[0-9]*npy'.format(object_name))
list_images2.sort(key=os.path.getmtime)

sensor = importlib.import_module('sensors.{}.sensor_params'.format(sensor_name))
object_3D = Object3D(object_name, sensor, False, False)

best_err = 10000000000
best_err2 = 10000000000

threshold_reals = np.arange(5,20)*0.0001
threshold_reals = np.arange(5,9)*0.0001
#threshold_reals = np.arange(9,13)*0.0001
#threshold_reals = np.arange(13, 17)*0.0001
#threshold_reals = np.arange(17, 21)*0.0001
threshold_sims = np.linspace(sensor.MIN_DEPTH_THRESHOLD, sensor.DEPTH_THRESHOLD, 10)
BLURS1 = [1,3]
BLURS2 = np.arange(0,2)
ERODES = np.arange(2,10)
DILATES = np.arange(2,10)
print('Case with: ', threshold_reals)
for threshold_real in threshold_reals:
    for threshold_sim in threshold_sims:
        for blur1 in BLURS1:
            for erode in ERODES:
                for dilate in DILATES:
                
                    for blur2 in BLURS2:
                        time_in = time.time()
                        errors = []
                        errors2 = []
                        for item in list_images2:
                            
                            real_ls = LocalShape(np.load(item))
                            png_item = item.replace('.npy','.png')
                            if 1:
                                try:
                                    
                                    transformation_real = Transformation(np.load(item.replace('ed_LS', 'ed_trans')))
                                    if 'head' in object_name:
                                        orig_pos = np.dot(-transformation_real.trans[:3,3], transformation_real.trans[:3,:3])
                                        if orig_pos[0] > 0:
                                            continue
                                except:
                                    print("nop", item)
                            
                                transformation_real.trans[2,3] *= -1 
                                ls, transformation_real = object_3D.renderTransformation(transformation_real)
                                
                                ls.changeDepthThreshold(sensor, threshold = threshold_sim)
                                
                                
                                if ls is None:
                                    print('No LS for item:', item)
                                    continue
                                else: 
                                    ls.toBinary(sensor)
                            
                                BLUR = ((blur1,blur1),blur2)
                                ERODE = (erode,erode)
                                DILATE =(dilate, dilate)
                                real_ls.realToBin(threshold_real, sensor, True, BLUR, ERODE, DILATE)
                                #plt.imshow(np.concatenate([real_ls.ls, ls.ls,  np.abs(ls.ls-real_ls.ls)], axis=1)); plt.show()
                                #plt.imshow(np.concatenate([real_ls.ls, ls.ls,  np.maximum(ls.ls-real_ls.ls,0)], axis=1)); plt.show()
                                errors.append(np.sum(np.abs(ls.ls-real_ls.ls)))
                                errors2.append(np.sum(np.max(ls.ls-real_ls.ls,0)))
                        if np.median(errors) < best_err:
                            print('Case with: ', threshold_reals)
                            best_t_r = threshold_real
                            best_t_s = threshold_sim
                            best_b1 = blur1
                            best_b2 = blur2
                            best_erode = erode
                            best_dilate = dilate
                            best_err = np.median(errors)
                            print('So far', best_err, 'sim', best_t_s, 'real', best_t_r, best_b1, best_b2, best_erode, best_dilate, 'other err', np.median(errors2))
                        if np.median(errors2) < best_err2:
                            print('Case with: ', threshold_reals)
                            best_t_r2 = threshold_real
                            best_t_s2 = threshold_sim
                            best_err2 = np.median(errors2)
                            best_b1_2 = blur1
                            best_b2_2 = blur2
                            best_erode_2 = erode
                            best_dilate_2 = dilate
                            print('So far2', best_err2, 'sim', best_t_s2, 'real', best_t_r2, best_b1_2, best_b2_2, best_erode_2, best_dilate_2,'other err', np.median(errors))
            # print('Total time', -time_in + time.time()) # ~9s
print('Final', best_err, 'sim', best_t_s, 'real', best_t_r)
print('Final2', best_err2, 'sim', best_t_s2, 'real', best_t_r2)
print('Case with: ', threshold_reals)
assert(False)

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



def compute_closest(trans):
    error = []
    
    
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
    if 1 or not os.path.exists(path_closest):
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
np.save('errors_{}.npy'.format(grid_name), errors)
        #break
