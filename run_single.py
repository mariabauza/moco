import numpy as np
import sys
import os
import time, pdb
missing_libs = []
try:import rospy
except: missing_libs.append('rospy')
import matplotlib.pyplot as plt
import glob
import copy
import cv2
try: import yaml
except: missing_libs.append('yaml')
try: from pyquaternion import Quaternion
except: missing_libs.append('pyquaternion')




def execute(config):
    for checkpoint in np.arange(config['start_epoch'], config['num_epoch'], 10):
        if config['desired_checkpoint'] is not None:
            if checkpoint != config['desired_checkpoint']: continue
        base_command = "python3  test_tactile.py -a resnet50 --lr 0.03 --batch-size 16 --multiprocessing-distributed --world-size 1 --rank 0 "
        
        base_command += "--epoch {} ".format(config['num_epoch'])
        
        base_command += "--dist-url 'tcp://localhost:{}' ".format(config['localhost'])

        base_command += "--date_name {} ".format(config['date_name'])
        
        if checkpoint:
            base_command += "--resume {}_checkpoint_{}.pth.tar ".format(config['date_name'],str(checkpoint).zfill(4))
        #base_command += "--resume 20_oct_checkpoint_{}.pth.tar ".format(str(119).zfill(4))
        #base_command += "--resume grease_view1_final_paper.pt ".format(str(checkpoint).zfill(4))
        #base_command += "--resume curved.pt ".format(str(checkpoint).zfill(4))
        if config['only_eval']:
            base_command += "--only_eval "
            if config['is_test']:
                base_command += "--is_test "
                if config['is_real']:
                    base_command += "--is_real "
                    if config['is_detectrion2']:
                        base_command += "--is_detectron2 "
        base_command += "--object_name {} ".format(config['object_name'])
        base_command += "--grid_name {} ".format(config['grid_name'])
        base_command += "--sensor_name {} ".format(config['sensor_name'])
        base_command += "--model_dir {} ".format(config['model_dir'])
        if 'binary' in config['model_dir']:
            base_command += "--is_binary "
        if 'pose' in config['model_dir']:
            base_command += "--input_pose "
        if config['is_vision']:
            base_command += "--is_vision {}".format(config['is_vision'])
        if config['vis']:
            base_command += "--vis "
        print(base_command)
        match_path = 'test_{}_matches_{}_{}_queue={}/'.format(config['date_name'], checkpoint+1, 'real', 1)
        print('trying?:', 'data/{}_face1/'.format(config['object_name']) + match_path)
        if os.path.exists('data/{}_face1/'.format(config['object_name']) + match_path): continue
        print('trying:', 'data/{}_face1/'.format(config['object_name']) + match_path)
        os.system(base_command)
        

config = {}
config['only_eval'] = 0
config['is_test'] = 0
config['is_real'] = 0
config['is_detectrion2'] = 0
config['vis'] = 0
config['is_vision'] = 0  #0, 1, 2 (tactile_vision)


config['date_name'] = '4_nov_k=20'
# Paper

try: id =os.environ['SLURM_ARRAY_TASK_ID']
except: print('No id'); id = 3
list_obj = ['pin_view2', 'grease_view1', 'head_view1', 'curved_view1']
config['object_name'] = list_obj[int(id)]
config['sensor_name'] = 'green_sensor'
config['grid_name'] = 'face1'

#config['grid_name'] = 'grid_depth_1'
#config['grid_name'] = 'final'

# Kitting
#config['object_name'] = 'big_head'
#config['sensor_name'] = 'kitting_sensor'
#config['grid_name'] = 'test_vision'


config['desired_checkpoint'] = None
config['num_epoch'] = 190
config['localhost'] = 10002
config['start_epoch'] = 0

#print('sleeeeping')
#time.sleep(18000)
#model_dirs = ['basic', 'binary'] #, 'poses', 'binary_poses']
#model_dirs = ['poses', 'binary_poses', 'binary', 'basic']
#model_dirs = ['binary_poses']
model_dirs = ['poses_5']
print('Change loss same, depth max smaller')
for model_dir in model_dirs:
    #for object_name in object_names:
    config['model_dir'] = model_dir  #+ '_23_aug'
    # Train
    if 1:
        print('Starting epoch 119')
        config['start_epoch'] = 9
        
        
        config['only_eval'] = 0
        config['desired_checkpoint'] = None
        print('Train')
        execute(config)
    if 0:
        config['desired_checkpoint'] = None
        config['only_eval'] = 1
        config['start_epoch'] = 119
        if 0:
            config['is_test'] = 1
            config['is_real'] = 1
            print('Eval real')
            execute(config)
        if 1:
            config['is_test'] = 1
            config['is_real'] = 0
            print('Eval simulated')
            execute(config)
        if 0:
            config['is_test'] = 0
            config['is_real'] = 0
            print('Eval grid')
            execute(config)
    if 0:
         from plots.plot_over_epoch import plot_violin
         plot_violin(config['object_name'], config['grid_name'], config['model_dir'])
