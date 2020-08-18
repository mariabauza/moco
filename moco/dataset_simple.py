import torch
import glob
import numpy as np
import cv2
import os
import shutil
import sys
import importlib
import time

class Dataset(torch.utils.data.Dataset):
    def __init__(self, object_name, sensor_name, grid_name, is_test =False):
        super().__init__()
        
        self.object_name = object_name
        data_path = 'data/{}_{}/train/'.format(object_name,grid_name)
        self.is_test=is_test
        if self.is_test:
            print('Loading paper data')
            self.list_images = glob.glob('/home/mcube/tactile_localization/data_tactile_localization/data_paper/{}/depth_clean/*true_LS*png'.format(object_name))
        else:
            self.list_images = glob.glob(data_path + '*/0.png')
        self.list_images.sort(key=os.path.getmtime)
        np.save('moco/object_name.npy',object_name)
        np.save('moco/grid_name.npy',grid_name)
        np.save('moco/sensor_name.npy',sensor_name)
        self.tmp_data_path = '/home/mcube/moco/tmp_data/'
        tmp_data = self.tmp_data_path + '*{}*.npy'.format(object_name)
        if len(glob.glob(tmp_data)):
            os.system('rm ' + tmp_data)
        print('Done dataset init')
        self.len = len(self.list_images)
    
        
        mean_path = data_path + 'mean.npy'
        if not os.path.exists(mean_path):
            self.compute_normalization(data_path)
        self.mean = np.load(data_path + 'mean.npy').astype(np.float32)
        self.std = np.load(data_path + 'std.npy').astype(np.float32)
    
    def compute_normalization(self,data_path):
        list_images = glob.glob(data_path + '*/0.png')
        images = []
        for path in list_images:
            images.append(cv2.imread(path))
        np.save(data_path + 'mean.npy', np.mean(np.mean(np.mean(images, axis=0), axis=0), axis=0))
        np.save(data_path + 'std.npy', np.std(np.std(np.std(images, axis=0), axis=0), axis=0))

    
    def __getitem__(self, it):

        item = self.list_images[it]
        ls1 = cv2.resize(cv2.imread(item), (200,200) ).astype(np.float32)
        
        try:
            if self.is_test:  #NO given LS for testing with real data
                ls2 = np.copy(ls1)
            else:
                ls2 = cv2.resize(cv2.imread(item.replace('0.png','1.png')), (200,200) ).astype(np.float32)
        except:
            print('No ls2', item)
            pass
        if len(glob.glob(self.tmp_data_path + '*{}*.npy'.format(self.object_name) )) < self.len:
            np.save(self.tmp_data_path + '{}_{}_{}.npy'.format(self.object_name, it, time.time()),[item])
        ls1 = ls1.swapaxes(0,2).swapaxes(1,2)
        ls2 = ls2.swapaxes(0,2).swapaxes(1,2)
        
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
        #Normalize
        for i in range(len(self.std)):
            ls2[i] = (ls2[i]-self.mean[i])/self.std[i]
            ls1[i] = (ls1[i]-self.mean[i])/self.std[i]
        
        return (torch.from_numpy(ls2), torch.from_numpy(ls1), it)  #Flipped, noisy image should be query

    def __len__(self):
        return len(self.list_images)
