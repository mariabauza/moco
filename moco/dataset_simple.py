import torch
from glob import glob
import numpy as np
import cv2
import os
import shutil
import sys
import importlib


class Dataset(torch.utils.data.Dataset):
    def __init__(self, object_name, sensor_name, grid_name, is_test =True):
        super().__init__()
        
        
        self.is_test=is_test
        if self.is_test:
            self.list_images = glob('/home/mcube/tactile_localization/data_tactile_localization/data_paper/{}/depth_clean/*true_LS*png'.format(object_name))
        else:
            self.list_images = glob('data/{}_{}/train/*/0.png'.format(object_name,grid_name))
        self.list_images.sort(key=os.path.getmtime)
        np.save('moco/object_name.npy',object_name)
        np.save('moco/grid_name.npy',grid_name)
        np.save('moco/sensor_name.npy',sensor_name)
        try:
            os.system('rm moco/last_item.npy')
        except: pass
        print('Done dataset init')
        
    def __getitem__(self, it):

        item = self.list_images[it]
        
        ls1 = cv2.resize(cv2.imread(item), (200,200) ).astype(np.float32)
        
        try:
            if self.is_test:
                ls2 = np.copy(ls1)
            else:
                ls2 = cv2.resize(cv2.imread(item.replace('0.png','1.png')), (200,200) ).astype(np.float32)
        except:
            print('No ls2', item)
            pass
        np.save('moco/last_item.npy',[item])
        ls1 = ls1.swapaxes(0,2).swapaxes(1,2)
        ls2 = ls2.swapaxes(0,2).swapaxes(1,2)
        
        #print(ls1.shape, ls2.shape)
        '''
        xv, yv = np.meshgrid(np.arange(ls1.shape[0])-(ls1.shape[0]-1)/2.0, np.arange(ls1.shape[1])-(ls1.shape[1]-1)/2.0, sparse=False, indexing='ij')
        
        ls1 = np.repeat(ls1[np.newaxis, :, :], 3, axis=0)
        ls2 = np.repeat(ls2[np.newaxis, :, :], 3, axis=0)
        
        ls1[1,:,:] = xv/80.0
        ls1[2,:,:] = yv/60.0
        
        ls2[1,:,:] = xv/80.0
        ls2[2,:,:] = yv/60.0
        
        
        print('hi')
        print(ls1.shape, ls2.shape)
        '''
        return (torch.from_numpy(ls2), torch.from_numpy(ls1), it)  #Flipped, noisy image should be query

    def __len__(self):
        return len(self.list_images)
