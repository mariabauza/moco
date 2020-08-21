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



sensor_name = np.load('moco/sensor_name.npy')

sensor = importlib.import_module('sensors.{}.sensor_params'.format(sensor_name))

object_name = np.load('moco/object_name.npy')
grid_name = np.load('moco/grid_name.npy')  


print('Generate data from:', sensor_name, object_name, grid_name)

grids = []; i = np.copy(grid_name)        
configs = []
grid_aux = Grid2D(object_name, sensor, i)
configs += [grid_aux.loadConfiguration()]
if configs[-1]['grid_3D']:
    grids += [Grid3D(object_name, sensor, i)]
    object_3D = Object3D(object_name, sensor, False, True)
else:
    grids += [Grid2D(object_name, sensor, i)]
    object_3D = Object3D(object_name, sensor, False, False)


def generate_noisy_sample(path):
    transformation = Transformation(np.load(path.replace('0.png','transformation.npy')))
    count = 0
    while True:
        transformation2 = transformation.noisyTransformation(max_dx = constants.TRANSLATION_INCREMENT/2.0, max_dy = constants.TRANSLATION_INCREMENT/2.0, max_dangle =constants.ANGLE_INCREMENT/2.0)
        ls, transformation_real = object_3D.renderTransformation(transformation2)
        if ls is not None:
            break
        else: 
            count += 1
            if count > 100: # Hack
                print('No LS for this one')
                os.system('cp {}  {}'.format(path, path.replace('0.png', '1.png')))
                return
    
    ls.toPNG(sensor)
    cv2.imwrite(path.replace('0.png', '1.png'), ls.ls)
    return

if __name__ == "__main__":
    while True:
        list_files = glob.glob('/home/mcube/moco/tmp_data/*{}*.npy'.format(object_name))
        list_files.sort(key=os.path.getmtime)
        if len(list_files) > 0:
            while os.path.exists(list_files[0]):
                try:
                    path = np.load(list_files[0])[0]
                    break
                except:
                    print(list_files[0])
        else: continue
        
        generate_noisy_sample(path)
        os.system('rm {}'.format(list_files[0]))
