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


only_eval = 0
is_test = 1
is_real = 1
is_detectrion2 = 1
vis = 0

object_name = 'pin_view2'
sensor_name = 'green_sensor'
grid_name = 'face1'

desired_checkpoint = 29
for checkpoint in np.arange(9, 300, 10):
    if desired_checkpoint is not None:
        if checkpoint != desired_checkpoint: continue
    base_command = "python3  train_tactile.py -a resnet50 --lr 0.03 --batch-size 16 --multiprocessing-distributed --world-size 1 --rank 0 "
    num_epoch = 300
    base_command += "--epoch {} ".format(num_epoch)
    localhost = 10001
    base_command += "--dist-url 'tcp://localhost:{}' ".format(localhost)
    
    if checkpoint:
        base_command += "--resume checkpoint_{}.pth.tar ".format(str(checkpoint).zfill(4))
    if only_eval:
        base_command += "--only_eval "
        if is_test:
            base_command += "--is_test "
            if is_real:
                base_command += "--is_real "
                if is_detectrion2:
                    base_command += "--is_detectron2 "
    base_command += "--object_name {} ".format(object_name)
    base_command += "--grid_name {} ".format(grid_name)
    base_command += "--sensor_name {} ".format(sensor_name)
    if vis:
        base_command += "--vis "
    os.system(base_command)
    path_data = 'data/{}_{}/'.format(object_name, grid_name)
    #os.system('cp {} {}'.format(path_data + 'errors.npy', path_data + 'errors_change_binary_checkpoint={}_is_test={}_is_real={}.npy'.format(checkpoint,is_test,is_real)))



