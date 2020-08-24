#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import moco.loader
import moco.builder
import moco.dataset_simple

import matplotlib.pyplot as plt
import sys

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

# Custom
parser.add_argument('--is_detectron2', action='store_true',
                    help='if real of true paper data')
parser.add_argument('--is_real', action='store_true',
                    help='if real of true paper data')
parser.add_argument('--is_test', action='store_true',
                    help='if paper or grdi data')
parser.add_argument('--only_eval', action='store_true',
                    help=' will build queue')
parser.add_argument('--object_name', default='', type=str, 
                    help='object name')
parser.add_argument('--grid_name', default='', type=str, 
                    help=' grid name')
parser.add_argument('--sensor_name', default='', type=str, 
                    help='sensor name')                                        
parser.add_argument('--vis', default='', type=str, 
                    help='sensor name')                                        
parser.add_argument('--input_pose', action='store_true',
                    help='if paper or grdi data')                                      
parser.add_argument('--is_binary', action='store_true',
                    help='if paper or grdi data')                                        
parser.add_argument('--model_dir', default='', type=str, 
                    help='folder save checkpoints')                                        


sys.path.append('/home/mcube/tactile_localization/')
with_tactile=parser.parse_args().only_eval
if with_tactile: 
    from tactile_localization.constants import constants
    from tactile_localization.classes.grid import Grid2D, Grid3D
    from tactile_localization.classes.object_manipulator import Object3D
    from tactile_localization.classes.local_shape import LocalShape, Transformation
import importlib

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
    args.model_dir += '/'

dists = []
def main_worker(gpu, ngpus_per_node, args):
    
    #####################################################################
    #####################################################################
    #####################################################################
    
    
    object_name = args.object_name
    sensor_name = args.sensor_name
    grid_name = args.grid_name
    print(object_name, sensor_name, grid_name)  
    if with_tactile:
        args.sensor = importlib.import_module('sensors.{}.sensor_params'.format(args.sensor_name))
        args.object_3D = Object3D(args.object_name, args.sensor, False, False)
        args.grid = Grid2D(args.object_name, args.sensor, args.grid_name)
    path_data = 'data/{}_{}'.format(object_name, grid_name)
    
    args.path_data = path_data
    
    os.makedirs(path_data + '/models/', exist_ok = True)
    os.makedirs(path_data + '/models/' + args.model_dir, exist_ok = True)
    
    
    train_dataset = moco.dataset_simple.Dataset(args, is_train = True)
    val_dataset = moco.dataset_simple.Dataset(args)
    args.moco_k = train_dataset.len
    print('Got dataset, val is_test', args.is_test, 'is_real', args.is_real)    
    import subprocess
    if not args.only_eval:
        subprocess.Popen(["python3", "moco/generate_data.py"])
    #####################################################################
    #####################################################################
    #####################################################################
    
    
    
    
    args.gpu = gpu
    
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = moco.builder.MoCo(
        models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    #print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        args.resume = path_data + '/models/' + args.model_dir +'/' + args.resume
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
        
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        cal_sampler = None
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers,
            pin_memory=True, sampler=train_sampler, drop_last=True)
        
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
            num_workers=args.workers,
            pin_memory=True, sampler=val_sampler, drop_last=True)

    if not args.only_eval:
        best_dist = 1
        for epoch in range(args.start_epoch, args.epochs):
            init = time.time()
            
            if args.distributed:
                train_sampler.set_epoch(epoch)
                print('Epoch', epoch, 'set')
            adjust_learning_rate(optimizer, epoch, args)
            
            # train for one epoch
            dists = train(train_loader, model, criterion, optimizer, epoch, args)
            print('Epoch: ', epoch, np.median(dists), np.mean(dists))
            
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                
                if (epoch +1) % 10 == 0:
                    save_best = False
                    if best_dist< np.median(dists):
                        best_dist = np.copy(np.median(dists))
                        save_best = True
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                    }, is_best=save_best, filename= path_data + '/models/{}/checkpoint_{:04d}.pth.tar'.format(args.model_dir, epoch))
                        # train for one epoch
            print('Time: ', time.time()-init)
    
    update_queue(train_loader, model, args)
    # train for one epoch
    dists, closest_dists = evaluate(val_loader, model, criterion, path_data, args)
    case_name = 'checkpoint={}_is_test={}_is_real={}_is_detectron2={}'.format(args.resume[:-8].split('_')[-1], args.is_test, args.is_real, args.is_detectron2)
    np.save(path_data + '/models/' + args.model_dir + '/errors_{}.npy'.format(case_name), dists)
    print('Median and mean:', np.median(dists), np.mean(dists), 'Closests Median and mean:', np.median(closest_dists), np.mean(closest_dists))
    np.save(path_data + '/models/' + args.model_dir + '/closest_errors_{}.npy'.format(case_name), closest_dists)    

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    dists_all = []
    closest_dists_all = []
    
    for i, images in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        #####################################################################
        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            indexes = images[2]
        #####################################################################    
        
        # compute output
        output, target = model(im_q=images[0], im_k=images[1], indexes = indexes)
        loss = criterion(output, target)
        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        res, dists, closest_dists = accuracy(output, target, args, topk=(1, 5))
        acc1, acc5 = res
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    progress.display(len(train_loader)-1)
    dists_all.append(dists)
    closest_dists_all.append(dists)
    return dists_all, closest_dists_all

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def evaluate(train_loader, model, criterion, path_data, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    epoch = 0
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    
    model.eval()
    end = time.time()
    dists_all = []
    closest_dists_all = []
    
    for i, images in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            indexes = images[2]
        # compute output
        queries = model(images[0])
        queries = nn.functional.normalize(queries, dim=1)
        
        output = torch.einsum('nc,ck->nk', [queries, model.queue.clone().detach()])
        target = indexes.cuda()
        loss = criterion(output, target)
        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        res, dists, closest_dists = accuracy(output, target, args, topk=(1, 5))
        print(np.median(dists), np.mean(dists), 'closests:', np.median(closest_dists), np.mean(closest_dists))
        acc1, acc5 = res
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        dists_all.append(dists)
        closest_dists_all.append(closest_dists)
        print('Median and mean so far:', np.median(dists_all), np.mean(dists_all))
    return dists_all, closest_dists_all

def update_queue(train_loader, model, args):
    
    model.eval()
    #print(model.named_modules)
    model.register_buffer("queue", torch.randn(args.moco_dim, args.moco_k))
    model.queue = nn.functional.normalize(model.queue, dim=0).cuda()
    for i, images in enumerate(train_loader):
        
        if args.gpu is not None:
            #images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            indexes = images[2]
        keys = model(images[1])
        keys = nn.functional.normalize(keys, dim=1)
        
        keys = moco.builder.concat_all_gather(keys)
        model.queue[:, indexes] = keys.T


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy2(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
        

import glob
import cv2
import numpy as np
def accuracy(output, target, args, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        if  with_tactile:
            dists =[]
            closest_dists =[]
            
            list_images = glob.glob(args.path_data + '/train/*/0.png')
            list_images.sort(key=os.path.getmtime)
            if args.is_test:
                if args.is_real:
                    list_images2 = glob.glob('/home/mcube/tactile_localization/data_tactile_localization/data_paper/{}/depth_clean/*ed_LS*png'.format(args.object_name))
                else:
                    list_images2 = glob.glob('/home/mcube/tactile_localization/data_tactile_localization/data_paper/{}/depth_clean/*true_LS*png'.format(args.object_name))
                list_images2.sort(key=os.path.getmtime)
            else:
                list_images2 = list_images

            for i in range(batch_size):
                path_pred = list_images[pred[0,i]]
                transformation = np.load(path_pred.replace('0.png', 'transformation.npy'))
                
                path_query = list_images2[target[i]]
                if not args.is_test:
                    path_query = path_query.replace('0.png', '1.png')
                    trans_path = path_query.replace('1.png', 'transformation.npy')
                    
                    transformation_real = np.load(trans_path)
                    closest_dist = args.grid.closestElement(Transformation(transformation_real))
                else:
                    transformation_real = np.load(path_query.replace('true_LS', 'trans').replace('LS', 'trans').replace('png', 'npy'))
                    transformation_real[2,3] *=-1
                    closest_dist = np.load(path_query.replace('_true', '').replace('ed_LS', 'ed_dist_closest_trans').replace('png', 'npy'))
                
                dist = args.object_3D.poseDistance(Transformation(transformation), Transformation(transformation_real))
                #transformation = Transformation(helper.single_filterreg_run(local_shape.ls, ICP_ls, 20, transformation.trans, sensor))
                #ICP_dist = args.object_3D.poseDistance(Transformation(transformation), Transformation(transformation_real))
                #print(dist)
                dists.append(dist)
                closest_dists.append(closest_dist)
                if args.vis:
                    if i == 0:
                        saving_path = args.path_data + '/debug_images/'
                        os.makedirs(saving_path, exist_ok = True)
                    if args.is_test and args.is_detectron2:
                        mask_num = path_query.replace('.png', '').split('_')[-1]
                        path_query = '/home/mcube/claudia/position/{}/{}_pointrend/predicted_mask_{}.png'.format(args.object_name, args.object_name, mask_num)
                    query = cv2.resize(cv2.imread(path_query), (235,235))
                    query = (query>250).astype('uint8')*255.0
                    prediction = cv2.imread(path_pred)
                    prediction = (prediction>250).astype('uint8')*255.0
                    result_img = np.concatenate([query, prediction, np.abs(query-prediction)], axis=1)
                    plt.imshow(np.array(result_img,np.int32)); 
                    plt.savefig(saving_path +'{}_correct={}.png'.format(target[i],correct[0,i]) ); plt.close()
                    
                    plt.imshow(np.array(result_img,np.int32)); 
                    plt.savefig(saving_path +'dist={}_{}.png'.format(dist, target[i]) ); 
                    #if dist > 0.01: plt.show()
                    plt.close()
            return res, dists, closest_dists 
    return res, [1000]*batch_size, [1000]*batch_size


if __name__ == '__main__':
    main()
