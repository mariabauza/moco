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



sensor_name = np.load('moco/sensor_name.npy')

sensor = importlib.import_module('sensors.{}.sensor_params'.format(sensor_name))

object_name = np.load('moco/object_name.npy')
grid_name = np.load('moco/grid_name.npy')  


print(sensor_name, object_name, grid_name)

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


last_path = None
path = None


def generate_noisy_sample(path):
    transformation = Transformation(np.load(path[0].replace('0.png','transformation.npy')))
    count = 0
    while True:
        transformation2 = transformation.noisyTransformation(max_dx = 0.0005, max_dy = 0.0005, max_dangle = 0.25)
        ls, transformation_real = object_3D.renderTransformation(transformation2)
        if ls is not None:
            break
        else: 
            count += 1
            if count > 100: # Hack
                os.system('cp {}  {}'.format(path[0], path[0].replace('0.png', '1.png')))
                return
    
    ls.toPNG(sensor)
    cv2.imwrite(path[0].replace('0.png', '1.png'), ls.ls)
    return

if __name__ == "__main__":
    while True:
        try:
            path = np.load('moco/last_item.npy')
        except:
            #print('no last item')
            pass
        if path is not None and path[0][-3:] != 'png': print('incomplete'); continue
        if path == last_path: continue
        
        last_path = np.copy(path)
        generate_noisy_sample(path)
