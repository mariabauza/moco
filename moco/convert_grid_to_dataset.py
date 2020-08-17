import sys
import os
import time, pdb
missing_libs = []
try:import rospy
except: missing_libs.append('rospy')
import numpy as np
import matplotlib.pyplot as plt
import glob
import copy
import cv2
try: import yaml
except: missing_libs.append('yaml')
try: from pyquaternion import Quaternion
except: missing_libs.append('pyquaternion')






object_name = 'pin_view2'
sensor_name = 'green_sensor'
grid_name = 'face1'

np.save('moco/object_name.npy',object_name)
np.save('moco/grid_name.npy',grid_name)
np.save('moco/sensor_name.npy',sensor_name)
import generate_data

grid_path = '/home/mcube/tactile_localization/data_tactile_localization/{}/{}/grids/{}/'.format(sensor_name, object_name, grid_name)
paths = glob.glob(grid_path + 'local_s*png')
paths.sort(key=os.path.getmtime)
data_path = 'data/{}_{}/'.format(object_name, grid_name)
os.makedirs(data_path, exist_ok = True)
data_path += 'train/'
os.makedirs(data_path, exist_ok = True)


for it, path in enumerate(paths):
    element_name = path.replace(grid_path + 'local_shape_', '').replace('.png', '')
    os.makedirs(data_path + element_name, exist_ok = True)
    os.system('cp {} {}/0.png'.format(path, data_path + element_name))
    os.system('cp {} {}/transformation.npy'.format(path.replace('local_shape', 'transformation').replace('.png','.npy'), data_path + element_name))
    path2 = ['{}/0.png'.format( data_path + element_name)]
    generate_data.generate_noisy_sample(path2)


