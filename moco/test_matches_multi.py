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
import time
import copy
#import pdb; pdb.set_trace()
start = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_epoch", type=int, default=-1)
parser.add_argument("-t", "--type_data", type=str, default=-1)
parser.add_argument("-q", "--is_queue", type=int, default=-1)
parser.add_argument("-o", "--object_name", type=str, default='grease_view1')
parser.add_argument("-d", "--date_name", type=str, default='29_oct')
parser.add_argument("-s", "--is_save", type=bool, default=True)

args = parser.parse_args()
num_epoch = args.num_epoch
is_queue = args.is_queue
type_data = args.type_data
date_name = args.date_name
is_save = args.is_save
sensor_name = 'green_sensor'
object_name = args.object_name #'grease_view1'
#object_name = 'pin_view2'
grid_name = 'face1'
#grid_name = 'final'
print('Saving:', is_save)

path_data = 'data/{}_{}'.format(object_name, grid_name)
debug_data = path_data +'/{}_images_debug/'.format(date_name)
matches_data = path_data +'/{}_matches_{}_{}_queue={}/'.format(date_name, '{}','{}','{}')
os.makedirs(debug_data, exist_ok = True)

list_grid_trans = np.load(path_data + '/{}_matches_list_trans_{}_queue={}.npy'.format(date_name, type_data, is_queue))
listTrans = []

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

if 'head' in object_name:
    new_list = []
    for list_img in list_images2:
        if os.path.exists(list_img.replace('npy','png')):
            new_list.append(list_img)
    list_images2 = copy.copy(new_list)

sys.path.append('../tactile_localization/')
from tactile_localization.constants import constants
from tactile_localization.utils import helper
from tactile_localization.classes.grid import Grid2D, Grid3D
from tactile_localization.classes.object_manipulator import Object3D    
from tactile_localization.classes.local_shape import LocalShape, Transformation

sensor = importlib.import_module('sensors.{}.sensor_params'.format(sensor_name))

print('Generate data from:', sensor_name, object_name, grid_name)

avoid_z = False

i = np.copy(grid_name); configs = []
grid_aux = Grid2D(object_name, sensor, i)
configs += [grid_aux.loadConfiguration()]
if configs[-1]['grid_3D']:
    object_3D = Object3D(object_name, sensor, False, True)
else:
    object_3D = Object3D(object_name, sensor, False, False)


def log_softmax(x):
    c = x.max()
    logsumexp = np.log(np.exp(x - c).sum())
    return x - c - logsumexp

def trans_dist(trans, all_tran):
    
    #trans[2,3]  *= -1
    #all_tran[2,3]  *= 0
    
    points1 = np.dot(object_3D.pcd_points_no_centering,trans[:3,:3].T) + trans[:3,3]
    
    points2 = np.dot(object_3D.pcd_points_no_centering,all_tran[:3,:3].T) + all_tran[:3,3]
    
    
    err = np.mean(np.sqrt(np.sum((points1 - points2)**2, axis=-1)), axis=-1)
    return err

def change_LS(ls):
    ls = np.where(ls >= sensor.DEPTH_THRESHOLD-0.0001, sensor.MAX_VAL_DEPTH-sensor.DESIRED_DIST, ls)
    ls += sensor.DESIRED_DIST

    return ls
            

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
    errors_ICP = []
    errors_ICP10 = []
    errors_ICP50 = []
    rand_errors = []
    rand_errors10 = []
    rand_errors50 = []
    closest_pos = []
    closest_errors = []
    counter_path = matches_data.format(it, type_data, is_queue) + 'counter.npy'
    
    
    
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

    num_contacts = 7
    import copy
    num_examples = 100
    examples_so_far = 0
    for num_sensors in range(num_contacts,num_contacts+1):
        if 1 and os.path.exists(matches_data.format(it, type_data, is_queue)  + 'avg6_errors_log_multicontact_{}_without_LS={}.npy'.format(num_sensors, True)): continue
        np.random.seed(10)
        masks_path = np.copy(list_images2)
        listLocalShapes = np.copy(list_images)
        #TODOS:
        # masks_path --> path to LS
        multi_errors = np.zeros((num_sensors, num_examples)) #[[]]*num_sensors
        multi_kin_errors = np.zeros((num_sensors, num_examples))
        while examples_so_far < num_examples:
            
            # Pre compute transformations sensors
            perm = np.random.permutation(len(masks_path))[:num_sensors]
            transformation_sensors = []  # will come from random pulling
            transformation_sensors_rel = []  # will come from random pulling
            closest_errors = []
            for j in range (num_sensors):
                #print(masks_path[perm[j]].split('.')[-2][-4:])
                transformation_real = Transformation(np.load(masks_path[perm[j]].replace('true_LS', 'trans').replace('LS', 'trans')))
                if 'head' in object_name and not os.path.exists(masks_path[perm[j]].replace('npy','png')): continue
                '''
                if 'head' in object_name:
                    orig_pos = np.dot(-transformation_real.trans[:3,3], transformation_real.trans[:3,:3])
                    if orig_pos[0] > 0:
                        print('PLEASE FIX HEAD DATA')
                        continue
                '''
                transformation_sensors.append(transformation_real)
                transformation_sensors_rel.append(np.matmul(transformation_real.trans, np.linalg.inv(transformation_sensors[0].trans)))
                path_dist_closest = masks_path[perm[j]].replace(replacement, 'ed_dist_closest_trans')
                closest_errors.append(np.load(path_dist_closest))
            ls = []
            probabilities_all = []
            
            
            not_computed = False
            for it_sensor in range (num_sensors):
                real_ls = LocalShape(np.load(masks_path[perm[it_sensor]]))
                real_ls.realToBin(0.0006, sensor, True)
                
                ls.append(real_ls)
                try:
                    trans_matches1 =  np.load(matches_data.format(it, type_data, is_queue) + 'predicted' + masks_path[perm[it_sensor]].replace('predicted_LS', '_index_matches_moco={}'.format(it)).replace('predicted_true_LS', '_index_matches_moco={}'.format(it)).split('/')[-1])
                except: print('Path not found for:', masks_path[perm[it_sensor]], 'fuck'); not_computed = True; continue
                val_matches1 =  np.load(matches_data.format(it, type_data, is_queue) + 'predicted' + masks_path[perm[it_sensor]].replace('predicted_LS', '_vals_matches_moco={}'.format(it)).replace('predicted_true_LS', '_vals_matches_moco={}'.format(it)).split('/')[-1])
                val_matches1 = log_softmax(val_matches1)
                if it_sensor == 0:
                    try:
                        trans_matches =  np.load(matches_data.format(it, type_data, is_queue) + 'predicted' + masks_path[perm[it_sensor]].replace('predicted_LS', '_index_matches_moco={}'.format(it)).replace('predicted_true_LS', '_index_matches_moco={}'.format(it)).split('/')[-1])
                    except: print('Path not found for the important one:', masks_path[perm[it_sensor]], 'fuck'); not_computed = True; continue
                    val_matches = np.load(matches_data.format(it, type_data, is_queue) + 'predicted' + masks_path[perm[it_sensor]].replace('predicted_LS', '_vals_matches_moco={}'.format(it)).replace('predicted_true_LS', '_vals_matches_moco={}'.format(it)).split('/')[-1])
                    val_matches = log_softmax(val_matches)
                
                probabilities = {}
                for probs, coords in zip(val_matches1, trans_matches1):
                    probabilities[list_grid_trans[coords]] = probs #, 0) #hack M
                probabilities_all += [probabilities]

            if not_computed:
                print('Failed', i, 'reruninig')
                continue

            best = [-1]*num_sensors
            best_score = [-10000]*num_sensors
            all_scores = [] #*num_sensors
            all_coords = [] #*num_sensors
            all_inds = [] #*num_sensors
            best_ind = [-1]*num_sensors
            counter = -1
            for prob, coords in zip(val_matches, trans_matches):
                counter += 1
                #print(prob, 'first')
                coords_path = list_grid_trans[coords]
                transformation = Transformation(np.load(coords_path.replace('true_LS', 'trans').replace('LS', 'trans')))
                bad = False
                probs = np.zeros(num_sensors)-1000000
                probs[0] = prob
                for it_sensor in range(1, num_sensors):
                    if bad:
                        continue
                    aux = -1
                    transformation2 = Transformation(np.matmul(transformation_sensors_rel[it_sensor], transformation.trans))
                    #if len(listTrans) == 0: closest_dist, closest_trans, closest_it, listTrans = object_3D.closestElement(transformation2, listLocalShapes)
                    #else: closest_dist, closest_trans, closest_it, _ = object_3D.closestElement(transformation2, listTrans)
                    try:
                        ang, x, y, face = grid_aux.closestElement(transformation2)
                        closest_path = grid_aux.data_dir + 'transformation_{}_{}_{}_{}_{}.npy'.format(ang, x, y, face, 1)
                    except: bad = True
                    if not bad and closest_path in probabilities_all[it_sensor]:
                        aux = probabilities_all[it_sensor][closest_path]
                        #aux = max(aux, score)
                        #print(aux)
                        probs[it_sensor] =  probs[it_sensor-1]+aux #using logs!!!!!!!!!!!!!!!!
                        closest_trans = grid_aux.loadTransformation(ang, x, y, face, 1)   
                        closest_dist = object_3D.poseDistance(closest_trans, transformation2)
                        #if closest_dist > 0.005: print(counter, 'High distance for closest:', closest_dist); bad = True

                        ### add an else?
                    #else:
                    #    continue#aux= -10000000 #pass # We will assume it is kinematics or something...
                #if bad:
                #    continue
                #print('probabilities as sensors increase', np.maximum(probs, -100))
                for it_sensor in range(0, num_sensors):
                    if probs[it_sensor] > best_score[it_sensor]:
                        best_score[it_sensor] =probs[it_sensor]
                        best[it_sensor] = copy.copy(coords_path)
                        best_ind[it_sensor] = copy.copy(counter)
                    if len(all_scores) == it_sensor:
                        all_scores.append([])
                        all_coords.append([])
                        all_inds.append([])
                    all_scores[it_sensor].append(probs[it_sensor])
                    all_coords[it_sensor].append(coords_path)
                    all_inds[it_sensor].append(counter)
            for it_sensor in range(0, num_sensors):
                #print(it_sensor, 'this should decrease', np.sum(np.array(all_coords[it_sensor]) > -1000))
                print(it_sensor, 'this should decrease', np.sum(np.array(all_scores[it_sensor]) > -1000), np.amin(all_scores[it_sensor])); 
                if best[it_sensor] == -1: print('This fails:', masks_path[perm[0]]); continue
                #print(best[it_sensor])
                transformation = Transformation(np.load(best[it_sensor]))
                if len(all_scores[it_sensor]) == 0: print('This fails:', masks_path[perm[0]]); continue
                kin_it = np.random.randint(len(all_scores[it_sensor]))
                kin_perm = np.random.permutation(len(all_scores[it_sensor]))
                kin_err = []
                found = 0
                for kin_it in kin_perm:
                    if found > 100: break
                    if all_scores[it_sensor][kin_it] < -1000 and it_sensor > 0: continue 
                    transformation_kin = Transformation(np.load(all_coords[it_sensor][kin_it]))
                    kin_err.append(object_3D.poseDistance(transformation_kin, transformation_sensors[0]))
                    found += 1
                multi_errors[it_sensor][examples_so_far] = object_3D.poseDistance(transformation, transformation_sensors[0])
                multi_kin_errors[it_sensor][examples_so_far] = np.mean(kin_err)

                #print(it_sensor, 'Multi error:', len(multi_errors[it_sensor]), len(multi_kin_errors[it_sensor]))
                print('Individual, sensor:' ,it_sensor, best_ind[it_sensor], 'Multi error:', np.round(multi_errors[it_sensor][examples_so_far]*1000,1), np.round(multi_kin_errors[it_sensor][examples_so_far]*1000,2))
                #print('For kin is ind:', all_inds[it_sensor][kin_it])
                #print('CLosest:', closest_errors)
                #multi_errors = np.array(multi_errors)
                #multi_kin_errors = np.array(multi_kin_errors)
                print(examples_so_far, it_sensor+1, 'Multi errors:', np.round(np.median(multi_errors[it_sensor][:examples_so_far+1])*1000,1), np.round(np.mean(multi_errors[it_sensor][:examples_so_far+1])*1000,1))
                print(examples_so_far, it_sensor+1, 'Multi kin errors:', np.round(np.median(multi_kin_errors[it_sensor][:examples_so_far+1])*1000,1), np.round(np.mean(multi_kin_errors[it_sensor][:examples_so_far+1])*1000,1))
                if is_save:
                    np.save(matches_data.format(it, type_data, is_queue)  + 'avg6_errors_log_multicontact_{}_without_LS={}.npy'.format(it_sensor+1, False), multi_errors[it_sensor])
                    np.save(matches_data.format(it, type_data, is_queue)  + 'avg6_errors_log_multicontact_{}_without_LS={}.npy'.format(it_sensor+1, True), multi_kin_errors[it_sensor])
            examples_so_far +=1

            print('Time used: ', np.round(time.time() - start))
print('Time used: ', np.round(time.time() - start))

print('Plotting time:')
import numpy as np
import sys
import matplotlib.pyplot as plt
import cv2

font_size = 30

plt.rc('font', size=font_size)          # controls default text sizes                                                                                                                                    
plt.rc('axes', titlesize=font_size)     # fontsize of the axes title                                                                                                                                      
plt.rc('axes', labelsize=font_size)    # fontsize of the x and y labels                                                                                                                                  
plt.rc('xtick', labelsize=font_size)    # fontsize of the tick labels                                                                                                                                     
plt.rc('ytick', labelsize=font_size)    # fontsize of the tick labels                                                                                                                                      
#plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize                                                                                                                                              
plt.rc('figure', titlesize=35)  # fontsize of the figure title         

### TODO: get random_mean!
save_path = matches_data.format(it, type_data, is_queue)

errorpath = glob.glob(save_path + '/rand_dict_error_case=*.npy')[0]
print( np.load(errorpath, allow_pickle=True).item().keys())
errors = np.load(errorpath, allow_pickle=True).item(); 
random_error = errors['rand_errors']
print('errorpath')
random_mean = np.mean(random_error)

#import pdb; pdb.set_trace()

fig, ax = plt.subplots(constrained_layout=True)
fig.set_size_inches(10, 8)

#plt.figure(figsize=(9,8))

num_contacts = 7
mean_match_errors = []
std_match_errors = []
median_match_errors = []
mean_match_no_LS_errors = []
std_match_no_LS_errors = []
median_match_no_LS_errors = []
mean_best_dist_errors = []
std_best_dist_errors = []
median_best_dist_errors = []
mean_best_no_LS_dist_errors = []
std_best_no_LS_dist_errors = []
median_best_no_LS_dist_errors = []
contacts = np.arange(1,num_contacts+1)

for i in contacts:
    print(i)
    match_errors = np.load(save_path + '/avg6_errors_log_multicontact_{}_without_LS={}.npy'.format(i, False))/random_mean
    match_no_LS_errors=np.load(save_path + '/avg6_errors_log_multicontact_{}_without_LS={}.npy'.format(i, True))/random_mean

    print('Multi errors:', np.round(np.median(match_errors)*1000,1), np.round(np.mean(match_errors)*1000,1))

    if i < 2:
        print(match_no_LS_errors)
    mean_match_errors.append(np.mean(match_errors))
    std_match_errors.append(np.std(match_errors))
    median_match_errors.append(np.median(match_errors))
    mean_match_no_LS_errors.append(np.mean(match_no_LS_errors))
    std_match_no_LS_errors.append(np.std(match_no_LS_errors))
    median_match_no_LS_errors.append(np.median(match_no_LS_errors))

print(match_errors)
print(median_match_errors)
print(mean_match_errors)
print(median_match_no_LS_errors)
print(mean_match_no_LS_errors)
plt.plot(contacts, mean_match_errors, 'bo-', markersize=13)
plt.fill_between(contacts, np.array(mean_match_errors)+np.array(std_match_errors), np.array(mean_match_errors)-np.array(std_match_errors), facecolor='blue', alpha=0.1)
plt.plot(contacts, median_match_errors, 'bx--', markersize=18)
plt.plot(contacts, mean_match_no_LS_errors, 'mo-', markersize=13)
plt.plot(contacts, median_match_no_LS_errors, 'mx--', markersize=18)

plt.xlabel("Number of Contacts")
plt.xticks(contacts)
bottom, top = plt.ylim()
plt.yticks(np.arange(0,top,0.20))

#plt.show()


ax2 = plt.twinx()
axes = plt.gca()
mn, mx = axes.get_ylim()
ax.set_ylim(0, top)
ax2.set_ylim(0*random_mean*1000, top*random_mean*1000)

ax.set_ylabel('Normalized Pose Error', labelpad=2)
ax2.set_ylabel('Pose Error (mm)')


plt.ylabel('Pose error (mm)', labelpad=10)
plt.xlabel('Number of contacts')
plt.title('{}'.format(object_name))
plt.ylim(bottom=0)
#plt.xticks(ind, ('Best1', 'ICP1', 'Best10', 'ICP10'))
#plt.yticks(np.arange(0, 81, 10))
print('Saving path:', save_path + '/avg6_plot_errors_log_multicontact_{}.png'.format(i))

plt.savefig(save_path + '/avg6_plot_errors_log_multicontact_{}.png'.format(i))

