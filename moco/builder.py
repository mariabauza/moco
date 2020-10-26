# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import numpy as np


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(pretrained=True) #, num_classes=dim)
        self.encoder_k = base_encoder(pretrained=True) #, num_classes=dim)
        self.use_old_model = True
        self.use_cosine = False
        def change_batch(encoder):
            encoder = nn.Sequential(*list(encoder.children()))
            encoder[1] = torch.nn.Identity()
            for i in range(4,8):
                for j in range(6):
                    try:
                        encoder[i][j].bn1 = torch.nn.Identity()
                        encoder[i][j].bn2 = torch.nn.Identity()
                        encoder[i][j].bn3 = torch.nn.Identity()
                        encoder[i][j].downsample[1] = torch.nn.Identity()
                    except:
                        print('failed', i)
            return encoder
        #change_batch(self.encoder_q)
        #change_batch(self.encoder_k)
        #self.encoder_q.bn1 = torch.nn.Identity()
        #self.encoder_k.bn1 = torch.nn.Identity()
        if self.use_old_model:
            self.encoder_q = nn.Sequential(*list(self.encoder_q.children())[:-1])
            self.encoder_k = nn.Sequential(*list(self.encoder_k.children())[:-1])
        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        
        self.queue = nn.functional.normalize(self.queue, dim=0)
        
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, indexes = None):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        # replace the keys at ptr (dequeue and enqueue)
        if indexes is None:
            assert self.K % batch_size == 0  # for simplicity
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            self.queue[:, indexes] = keys.T

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def cosine_distance(self, x1, x2=None, eps=1e-8):
            x2 = x1 if x2 is None else x2
            w1 = x1.norm(p=2, dim=1, keepdim=True)
            w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
            return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

    def forward(self, im_q, im_k = None, indexes= None, only_eval = False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        
        q = self.encoder_q(im_q)  # queries: NxC
        if self.use_old_model:
            q = q.mean(2).mean(2)
        q = nn.functional.normalize(q, dim=1)
        
        if only_eval: 
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
            return l_neg
        if im_k is None: return q
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k = self.encoder_k(im_k)  # keys: NxC
            if self.use_old_model:
                k = k.mean(2).mean(2)
            k = nn.functional.normalize(k, dim=1)
            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        # compute logits
        
        if self.use_cosine:
            l_pos = torch.nn.CosineSimilarity()(q,k).unsqueeze(-1)
            l_neg = self.cosine_distance(q,self.queue.T.clone().detach())
        else:
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # TODOOOOOOOOO
        l_neg[:,indexes] = -10000 # Hack not add to loss
        #how_many = l_neg.shape[1]
        #perm = np.random.permutation(how_many)[:-500]  #only consider last 500 for comparisons
        #l_neg[:,perm] = -100 # Hack not add to loss

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, indexes)
        return logits, labels
        
    def encode(self, im):
        """
        Input:
            im: a batch of images
        Output:
            encodings
        """

        # compute query features
        q = self.encoder_q(im) 
        if self.use_old_model:
            q = q.mean(2).mean(2)
        q = nn.functional.normalize(q, dim=1)
        return q


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
