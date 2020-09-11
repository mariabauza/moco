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
    def __init__(self, args,is_train = False):
        super().__init__()

        self.object_name = args.object_name
        self.data_path = 'data/{}_{}/train/'.format(self.object_name,args.grid_name)
        self.is_test=args.is_test
        if is_train: self.is_test = False
        self.is_real=args.is_real
        self.is_detectron2=args.is_detectron2
        self.input_pose = args.input_pose
        self.is_binary = args.is_binary
        self.only_eval = args.only_eval
        if self.is_test:
            print('Loading paper data')
            if self.is_real:
                self.list_images = glob.glob('../tactile_localization/data_tactile_localization/data_paper/{}/depth_clean/*ed_LS*png'.format(self.object_name))
            else:
                self.list_images = glob.glob('../tactile_localization/data_tactile_localization/data_paper/{}/depth_clean/*ed_true_LS*png'.format(self.object_name))
        else:
            self.list_images = glob.glob(self.data_path + '*/0.png')
        self.list_images.sort(key=os.path.getmtime)
        np.save('moco/object_name.npy',self.object_name)
        np.save('moco/grid_name.npy',args.grid_name)
        np.save('moco/sensor_name.npy',args.sensor_name)
        self.tmp_data_path = 'tmp_data/'
        tmp_data = self.tmp_data_path + '*{}*.npy'.format(self.object_name)
        if len(glob.glob(tmp_data)):
            os.system('rm ' + tmp_data)
        print('Done dataset init')
        self.len = len(self.list_images)

        self.size1 = 200
        self.size2 = 200

        if self.input_pose:
            ls1 = cv2.imread(self.list_images[0])
            xv, yv = np.meshgrid(np.arange(self.size1)-(self.size1-1)/2.0, np.arange(self.size2)-(self.size2-1)/2.0, sparse=False, indexing='ij')
            self.xv = (xv - np.mean(xv))/np.std(xv)
            self.yv = (yv - np.mean(yv))/np.std(yv)

        self.mean_path = self.data_path + '../models/' + args.model_dir + '/mean.npy'
        if not os.path.exists(self.mean_path):
            self.compute_normalization()
        self.mean = np.load(self.mean_path).astype(np.float32)
        self.std = np.load(self.mean_path.replace('mean','std')).astype(np.float32)

    def compute_normalization(self):
        images = []

        self.std = np.ones(3)
        self.mean = np.zeros(3)
        print('Computing normalization')
        for it in range(self.__len__()):
            if it%1000 == 0: print(it)
            _, ls, _  = self.__getitem__(it)
            images.append(ls.cpu().numpy())
        np.save(self.mean_path, np.mean(np.mean(np.mean(images, axis=0), axis=-1), axis=-1))
        np.save(self.mean_path.replace('mean','std'), np.std(np.std(np.std(images, axis=-1), axis=-1), axis=0))
        
        
        '''
        list_images = glob.glob(self.data_path + '*/0.png')
        images = []
        for path in list_images:
            images.append(cv2.imread(path))
        print(images[0].shape, np.mean(np.mean(np.mean(images, axis=0), axis=0), axis=0).shape)
        np.save(self.mean_path, np.mean(np.mean(np.mean(images, axis=0), axis=0), axis=0))
        np.save(self.mean_path.replace('mean','std'), np.std(np.std(np.std(images, axis=0), axis=0), axis=0))
        '''
    
    def __getitem__(self, it):
        item = self.list_images[it]
        if self.is_test and self.is_detectron2:
            mask_num = item.replace('.png', '').split('_')[-1]
            item = '../claudia/position/{}/{}_pointrend/predicted_mask_{}.png'.format(self.object_name, self.object_name, mask_num)
        ls1 = cv2.resize(cv2.imread(item), (self.size2,self.size1) ).astype(np.float32) 
        ls1 = (ls1>250).astype(np.float32)*255.0
        ls2 = None
        while ls2 is None:
            try:
                if self.is_test:  #NO given LS for testing with real data
                    ls2 = np.copy(ls1)
                else:
                    ls2 = cv2.resize(cv2.imread(item.replace('0.png','1.png')), (self.size2,self.size1) ).astype(np.float32)
                    max_val = np.amin(ls2) + np.random.randint(int(51/5),51)
                    ls2 = (ls2>max_val).astype(np.float32)*255.0
            except:
                print('No ls2', item)
                pass
            if not self.only_eval and len(glob.glob(self.tmp_data_path + '*{}*.npy'.format(self.object_name) )) < self.len:
                np.save(self.tmp_data_path + '{}_{}_{}.npy'.format(self.object_name, it, time.time()),[item])
        if self.is_binary:
            ls1 = np.amax(ls1) - ls1
            ls2 = np.amax(ls2) - ls2
        ls1 = ls1.swapaxes(0,2).swapaxes(1,2)
        ls2 = ls2.swapaxes(0,2).swapaxes(1,2)
        
        
        if self.input_pose:
        
            ls1[1,:,:] = self.xv
            ls1[2,:,:] = self.yv
            
            ls2[1,:,:] = self.xv
            ls2[2,:,:] = self.yv

            for i in range(1):
                ls2[i] = (ls2[i]-self.mean[i])/self.std[i]
                ls1[i] = (ls1[i]-self.mean[i])/self.std[i]
        else:
            #Normalize
            for i in range(len(self.std)):
                ls2[i] = (ls2[i]-self.mean[i])/self.std[i]
                ls1[i] = (ls1[i]-self.mean[i])/self.std[i]
        return (torch.from_numpy(ls2), torch.from_numpy(ls1), it)  #Flipped, noisy image should be query

    def __len__(self):
        return len(self.list_images)
