import glob
import numpy as np
import cv2
import os
import shutil
import sys
import importlib
sys.path.append('../tactile_localization/')
from tactile_localization.constants import constants
from tactile_localization.classes.grid import Grid2D, Grid3D
from tactile_localization.classes.object_manipulator import Object3D
from tactile_localization.classes.local_shape import LocalShape, Transformation
import matplotlib.pyplot as plt
import time

sensor_name = np.load('moco/sensor_name.npy')

sensor = importlib.import_module('sensors.{}.sensor_params'.format(sensor_name))

object_name = np.load('moco/object_name.npy')
grid_name = np.load('moco/grid_name.npy')  
is_vision = np.load('moco/is_vision.npy') 
change_gripper = np.load('moco/change_gripper.npy') 
program_pid = np.load('moco/program_pid.npy') 

print('Generate data from:', sensor_name, object_name, grid_name)

grids = []; i = np.copy(grid_name)        
configs = []
grid_aux = Grid2D(object_name, sensor, i)
configs += [grid_aux.loadConfiguration()]
if configs[-1]['grid_3D']:
    grids += [Grid3D(object_name, sensor, i)]
    object_3D = Object3D(object_name, sensor, False, True, vision = is_vision)
else:
    grids += [Grid2D(object_name, sensor, i)]
    object_3D = Object3D(object_name, sensor, False, False, vision = is_vision)

print('we are done')


def is_closest(trans_path,transformation, transformation_noisy):
    
    orig_pose = object_3D.poseDistance(transformation,transformation_noisy)
    values = trans_path.split('_')
    for it, value in enumerate(values[-5:-1]):
        new_values = trans_path.split('_')
        for i in range(2):
            itt = -5 + it
            #print(new_values[itt], str(int(value) + 2*i-1))
            new_values[itt] = str(int(value) + 2*i-1)
            
            trans_path2 = '_'.join(new_values)
            #print(new_values[-5:-1][it])
            try:transformation2 = Transformation(np.load(trans_path2))
            except: continue
            pose = object_3D.poseDistance(transformation2,transformation_noisy)
            #print(orig_pose, pose)
            if pose < orig_pose: return False 
    return True

def generate_noisy_sample(path):
    #transformation = Transformation(np.load(path.replace('0.png','transformation.npy')))
    trans_path = path.replace('local_shape', 'transformation').replace('.png','.npy')
    transformation = Transformation(np.load(trans_path))
    count = 0
    while True:
        transformation2 = transformation.noisyTransformation(max_dx = constants.MAX_DX/2.0, max_dy = constants.MAX_DY/2.0, max_dangle =constants.MAX_DANGLE/2.0)
        if not is_closest(trans_path, transformation, transformation2): continue
        ls, trans1 = object_3D.renderTransformation(transformation2, go_fast = True)
        if ls is not None:
            if is_vision:
                ls2, trans2 = object_3D.renderTransformation(transformation2, 2, go_fast = True)
                if ls2 is not None:
                    opening = (sensor.SECOND_CAMERA - 2*sensor.DESIRED_DIST) + trans2.trans[2][3] - trans1.trans[2][3]
                    break
            else: break
        else:
            count += 1
            if count > 100: # Hack
                print('No LS for this one')
                os.system('cp {}  {}'.format(path, path.replace('1.png', '4.png')))
                return
    #plt.imshow(ls.ls); plt.savefig('debug.png')
    ls.toPNG(sensor)
    #cv2.imwrite(path.replace('0.png', '1.png'), ls.ls)
    np.save(trans_path.replace('1.npy', '4.npy'), trans1.trans)
    cv2.imwrite(path.replace('1.png', '4.png'), ls.ls)
    
    if is_vision:
        np.save(trans_path.replace('1.npy', '5.npy'), trans2.trans)
        np.save(trans_path.replace('.npy', '_4.npy').replace('transformation','opening'), opening)
    if is_vision:
        if change_gripper:
            aux_rot = np.copy(object_3D.rot)
            object_3D.rot[:3,3] += (np.random.rand(3)-0.5)*0.1 #
            depth, depth_obj = object_3D.renderDepth(transformation2, opening)
        object_3D.rot = np.copy(aux_rot)
        max_depth = 0.81520295
        min_depth = 0.38
        png_depth = (depth-min_depth)/(max_depth-min_depth)*255
        png_obj_depth = np.where(depth_obj == 0, depth_obj, png_depth)*255
        cv2.imwrite(path.replace('0.png', 'depth_1.png'), png_depth)
        cv2.imwrite(path.replace('0.png', 'obj_depth_1.png'), png_obj_depth)
        plt.imshow(cv2.imread(path.replace('0.png', 'depth_0.png'))); plt.show()
        plt.imshow(png_depth); plt.show()
        
    return

def check_pid(pid):        
    """ Check For the existence of a unix pid. """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True

if __name__ == "__main__":
    last_path = None
    time.sleep(5)
    while check_pid(program_pid):
        try:
            list_files = glob.glob('tmp_data/*{}*.npy'.format(object_name))
            list_files.sort(key=os.path.getmtime)
        except: continue
        if len(list_files) > 0:
            num = np.random.randint( min(1000, len(list_files) ) )
            count = 0
            while os.path.exists(list_files[num]) and count < 100:
                try:
                    path = np.load(list_files[num])[0]
                    os.system('rm {}'.format(list_files[num]))

                    if last_path == path: continue
                    generate_noisy_sample(path)
                    last_path = np.copy(path)

                    break
                except:
                    #print('Fails load or compute file:', list_files[num])
                    count += 1
        else: continue
    else: print('Exiting generate data program')
