import open3d
import torch
import glob
import numpy as np
import cv2
import os
import shutil
import sys
import importlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
#import pdb; pdb.set_trace()
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_epoch", type=int, default=-1)
parser.add_argument("-t", "--type_data", type=str, default=-1)
parser.add_argument("-q", "--is_queue", type=int, default=-1)
parser.add_argument("-o", "--object_name", type=str, default='grease_view1')
parser.add_argument("-d", "--date_name", type=str, default='29_oct')

args = parser.parse_args()
num_epoch = args.num_epoch
is_queue = args.is_queue
type_data = args.type_data
date_name = args.date_name

sensor_name = 'green_sensor'
object_name = args.object_name #'grease_view1'
#object_name = 'pin_view2'
grid_name = 'face1'
#grid_name = 'final'


path_data = 'data/{}_{}'.format(object_name, grid_name)
debug_data = path_data +'/{}_images_debug/'.format(date_name)
matches_data = path_data +'/{}_matches_{}_{}_queue={}/'.format(date_name, '{}','{}','{}')
os.makedirs(debug_data, exist_ok = True)

list_images = glob.glob(os.environ['HOME'] + '/tactile_localization/data_tactile_localization/{}/{}/grids/{}/transformation*1.npy'.format(sensor_name, object_name, grid_name))
list_images.sort(key=os.path.getmtime)
list_images2 = glob.glob('../tactile_localization/data_tactile_localization/data_paper/{}/depth_clean/*predicted_LS_[0-9]*npy'.format(object_name))
list_images2.sort(key=os.path.getmtime)
replacement = 'ed_LS'
if type_data == 'true':
    list_images2 = glob.glob('../tactile_localization/data_tactile_localization/data_paper/{}/depth_clean/*predicted_true_LS_[0-9]*npy'.format(object_name))
    list_images2.sort(key=os.path.getmtime)
    replacement = 'ed_true_LS'
print('Len list iamges:', len(list_images), len(list_images2))


#'''
sys.path.append('../tactile_localization/')
from tactile_localization.constants import constants
from tactile_localization.classes.grid import Grid2D, Grid3D
from tactile_localization.classes.object_manipulator import Object3D    
from tactile_localization.classes.local_shape import LocalShape, Transformation

sensor = importlib.import_module('sensors.{}.sensor_params'.format(sensor_name))

print('Generate data from:', sensor_name, object_name, grid_name)

i = np.copy(grid_name); configs = []
grid_aux = Grid2D(object_name, sensor, i)
configs += [grid_aux.loadConfiguration()]
if configs[-1]['grid_3D']:
    object_3D = Object3D(object_name, sensor, False, True)
else:
    object_3D = Object3D(object_name, sensor, False, False)
#'''

def trans_dist(trans, all_tran):
    
    #trans[2,3]  *= -1
    #all_tran[2,3]  *= 0
    
    points1 = np.dot(object_3D.pcd_points_no_centering,trans[:3,:3].T) + trans[:3,3]
    
    points2 = np.dot(object_3D.pcd_points_no_centering,all_tran[:3,:3].T) + all_tran[:3,3]
    
    
    err = np.mean(np.sqrt(np.sum((points1 - points2)**2, axis=-1)), axis=-1)
    return err

def compute_closest(trans, from_grid = False):
    error = []
    
    for it, all_tran in enumerate(all_trans):
        
        err = trans_dist(trans, all_tran)
        error.append(err)
    
        
    it = np.argmin(error)
    if from_grid:
        return error[it], all_trans[it], list_trans[it]
    return error[it], all_trans[it]


if num_epoch == -1:
    epochs = np.arange(79,200)
else:
    epochs = [num_epoch]

for it in epochs:
    errors = []
    errors10 = []
    errors50 = []
    rand_errors = []
    rand_errors10 = []
    rand_errors50 = []
    closest_pos = []
    closest_errors = []
    counter_path = matches_data.format(it, type_data, is_queue) + 'test_counter.npy'
    
    
    
    if type_data != 'test': path_trans_matches = matches_data.format(it, type_data, is_queue) + 'predicted' + list_images2[0].replace(replacement, 'ed_matches_moco={}'.format(it)).split('predicted')[-1]
    else: path_trans_matches = matches_data.format(it, type_data, is_queue) + 'predicted' + list_images[0].replace('transformation', 'matches_moco={}'.format(it)).split('/')[-1]
    if not os.path.exists(path_trans_matches): 
        print('NO: ', path_trans_matches)
        continue  
    
    if not os.path.exists(counter_path): 
        case_type = 0
    else:
        case_type = (np.load(counter_path)+1) % 4
    np.save(counter_path, case_type)
    
    print('Epoch:', it)
    count = 0
    #if case_type == 3: list_images2 = list_images3
    if type_data =='test': list_img = list_images
    else: list_img = list_images2
    max_val = 250
    for it2, item in enumerate(list_img):
        if 'head' in object_name and not os.path.exists(item.replace('npy','png')): continue
        ls1 = cv2.imread(item.replace('npy','png'))
        #max_val = np.amin(ls1) + 25
        ls1 = (ls1>max_val).astype(np.float32)*255.0
        path_trans = item.replace(replacement, 'ed_trans')
        path_closest = item.replace(replacement, 'ed_closest_trans')
        path_dist_closest = item.replace(replacement, 'ed_dist_closest_trans')
        path_trans_matches = matches_data.format(it, type_data, is_queue) + 'predicted' + item.replace(replacement, 'ed_matches_moco={}'.format(it)).split('predicted')[-1]
        path_LS_matches = matches_data.format(it, type_data, is_queue) + 'predicted' + item.replace(replacement, 'ed_LS_matches_moco={}'.format(it)).split('predicted')[-1]
        if type_data =='test':
            path_trans_matches = matches_data.format(it, type_data, is_queue) + 'predicted' + item.replace('transformation', 'matches_moco={}'.format(it)).split('/')[-1]
            path_LS_matches = matches_data.format(it, type_data, is_queue) + 'predicted' + item.replace('transformation', 'LS_matches_moco={}'.format(it)).split('/')[-1]
        if not os.path.exists(path_trans_matches): 
            count += 1; print('Path do not exist num:', count, path_trans_matches)
            continue  
        trans = np.load(path_trans)
        #if 'grease' not in object_name:
        #    trans[2,3]  *= -1
    
        matches = np.load(path_trans_matches)
        
        closest_error = np.load(path_dist_closest)
        closest_errors.append(closest_error)
        err_vec = []
        pos = -1
        for i, match in enumerate(matches):            
            
            all_tran = np.load(match.replace('local_shape', 'transformation').replace('png', 'npy'), allow_pickle=True)
            #if type_data =='test': all_tran = all_tran.trans
            
            err_i = trans_dist(trans, all_tran)
            err_vec.append(err_i)
            if type_data != 'test': 
                if abs(closest_error - err_i) < 0.00001:
                    pos = np.copy(i)
                
            if 0 and i == 0: 
                try:
                    aa = np.load(path_LS_matches)[i]
                
                    ls2 = cv2.imread(aa)            
                    #max_val = np.amin(ls2) + 25
                    ls2 = (ls2>max_val).astype(np.float32)*255.0
                    #print(trans, all_tran)
                    #print(err_i)
                    plt.imshow(np.concatenate([ls1, ls2], axis=1).astype(np.uint8)); 
                    if 0: 
                        plt.show()
                    fig_name = debug_data + 'best_match_epoch={}_type={}_case={}_err={}.png'.format(it, type_data, case_type,np.round(err_i*1000,1))
                    plt.savefig(fig_name)
                    fig_name = matches_data.format(it, type_data, is_queue) + '/best_match_epoch={}_type={}_case={}_err={}.png'.format(it, type_data, case_type,np.round(err_i*1000,1))
                    plt.savefig(fig_name)
                    np.save(fig_name.replace('png', 'npy'), [item.replace('npy','png'), aa])
                except: pass 
        
        errors.append(err_vec[1])
        errors10.append(np.amin(err_vec[:10]))
        errors50.append(np.amin(err_vec[:50]))
        closest_pos.append(pos)
        
        
        ### Compute random errors
        rand_err_vec = []
        path_trans = glob.glob(matches[0].split('transform')[0] + 'transformation_[0-9]*npy')
        #print(matches[0])
        perm = np.random.permutation(len(path_trans))[:len(matches)]
        rand_matches = np.array(path_trans)[perm]
        for i, match in enumerate(rand_matches):     
            all_tran = np.load(match)            
            err_i = trans_dist(trans, all_tran)
            rand_err_vec.append(err_i)       
            
        rand_errors.append(rand_err_vec[1])
        rand_errors10.append(np.amin(rand_err_vec[:10]))
        rand_errors50.append(np.amin(rand_err_vec[:50]))
        
    dict_error = {}
    dict_error['closest_errors'] = closest_errors
    dict_error['errors'] = errors
    dict_error['errors10'] = errors10
    dict_error['errors50'] = errors50
    dict_error['rand_errors'] = rand_errors
    dict_error['rand_errors10'] = rand_errors10
    dict_error['rand_errors50'] = rand_errors50
    np.save( matches_data.format(it, type_data, is_queue) + 'test_dict_error_case={}.npy'.format(case_type), dict_error)
    if 0:
        closest_errors = np.round(np.array(closest_errors)*1000,1)
        errors = np.round(np.array(errors)*1000,1)
        errors10 = np.round(np.array(errors10)*1000,1)
        errors50 = np.round(np.array(errors50)*1000,1)
        rand_errors = np.round(np.array(rand_errors)*1000,1)
        rand_errors10 = np.round(np.array(rand_errors10)*1000,1)
        rand_errors50 = np.round(np.array(rand_errors50)*1000,1)
    print(it, type_data, is_queue,'Error:', np.round(np.median(errors)*1000,1), np.round(np.mean(errors)*1000, 1))
    print('          ', np.round(np.median(rand_errors)*1000,1), 'random')
    print(it, type_data, is_queue,'Error10:', np.round(np.median(errors10)*1000,1), np.round(np.mean(errors10)*1000, 1))
    print('            ', np.round(np.median(rand_errors10)*1000,1), 'random')
    print(it, type_data, is_queue, 'Error50:', np.round(np.median(errors50)*1000,1), np.round(np.mean(errors50)*1000, 1))
    print('            ', np.round(np.median(rand_errors50)*1000,1), 'random')
    print(it, type_data, is_queue, 'Closest Error:',  np.round(np.median(closest_errors)*1000,1), np.round(np.mean(closest_errors)*1000,1))
    print('            ', np.round(np.median(rand_errors)*1000,1), 'random')
    arr_closest_pos = np.array(closest_pos)
    print(it, type_data, is_queue, 'Closest Pos:', np.median(arr_closest_pos[arr_closest_pos > 0]), np.mean(arr_closest_pos[arr_closest_pos > 0]), np.sum(arr_closest_pos > 0), len(arr_closest_pos))
        
