import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

import sys
sys.path.append("./lib/vdb/build/")
from plenvdb import DensityVDB, ColorVDB

from torch.utils.cpp_extension import load
parent_dir = os.path.dirname(os.path.abspath(__file__))
render_utils_cuda = load(
        name='render_utils_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
        verbose=True)

total_variation_cuda = load(
        name='total_variation_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/total_variation.cpp', 'cuda/total_variation_kernel.cu']],
        verbose=True)


def create_grid(type, **kwargs):
    if type == 'DenseGrid':
        return DenseGrid(**kwargs)
    elif type == 'VDBGrid':
        return VDBGrid(**kwargs)
    else:
        raise NotImplementedError

class QueryVerticalInVDB(autograd.Function):
    @staticmethod
    def forward(ctx, pts, grid): # (3,N)
        N = pts.shape[-1]
        pts = pts.detach().cpu()
        ctx.save_for_backward(pts)
        ctx.grid = grid
        if N == 0:
            return torch.zeros([N, grid.getndim()]).float().cuda()
        x, y, z = pts.numpy()
        res = grid.forward(x, y, z)
        return torch.tensor(res).reshape(N, grid.getndim())
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.needs_input_grad[0]:
            pts = ctx.saved_tensors[0]
            if pts.shape[-1] > 0:
                x, y, z = pts.numpy()
                grad = grad_output.reshape(-1).detach().cpu().numpy()
                ctx.grid.backward(x, y, z, grad)
        return None, None


''' VDB 3D grid
'''
class VDBGrid(nn.Module):
    def __init__(self, channels, world_size, xyz_min, xyz_max, **kwargs):
        super(VDBGrid, self).__init__()
        self.channels = channels
        self.world_size = world_size
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        if self.channels == 1:
            self.grid = DensityVDB(world_size.tolist(), 1)
        else:
            self.grid = ColorVDB(world_size.tolist(), self.channels)

    def wld2idx(self, pts):
        return (pts-self.xyz_min) / (self.xyz_max - self.xyz_min) * (self.world_size-1)

    def forward(self, xyz):
        '''
        xyz: global coordinates to query
        '''
        shape = xyz.shape[:-1] # (N,3)
        pts = self.wld2idx(xyz).reshape(-1,3).t().contiguous().float().requires_grad_() # (3,N)
        out = QueryVerticalInVDB.apply(pts, self.grid).reshape([*shape, self.channels])
        if self.channels == 1:
            out = out.squeeze(-1)
        return out

    def scale_volume_grid(self, new_world_size):
        data = self.get_dense_grid()
        new_data = F.interpolate(data, size=tuple(new_world_size), mode='trilinear', align_corners=True).permute(0,2,3,4,1).squeeze().contiguous()
        # print(f"========> data transfer uses: {self.grid.getTimer()} secs")
        if self.channels == 1:
            self.grid = DensityVDB(new_world_size.tolist(), 1)
        else:
            self.grid = ColorVDB(new_world_size.tolist(), self.channels)
        self.grid.resetTimer()
        self.grid.copyFromDense(new_data.reshape(-1).detach().cpu().numpy())
        self.world_size = new_world_size
        # self.grid.scale_volume_grid(new_world_size.tolist())

    def total_variation_add_grad(self, wx, wy, wz, dense_mode):
        '''Add gradients by total variation loss in-place'''
        return
        self.grid.total_variation_add_grad(wx, wy, wz, dense_mode)

    def setValuesOn_bymask(self, mask, val):
        assert self.channels == 1
        self.grid.setValuesOn_bymask(mask.reshape(-1).detach().cpu().numpy(), val)

    def get_dense_grid(self):
        dsgrid = torch.tensor(self.grid.get_dense_grid()).reshape(self.world_size.tolist()+[-1]).permute(3,0,1,2).unsqueeze(0).contiguous()
        return dsgrid

    def load_from(self, loadpath):
        if self.channels == 1:
            self.grid.load_from(loadpath + 'density.vdb')
        else:
            self.grid.load_from(loadpath + 'color.vdb')

    def save_to(self, savepath):
        if self.channels == 1:
            self.grid.save_to(savepath + 'density.vdb')
        else:
            self.grid.save_to(savepath + 'color.vdb')
    
    def resetTimer(self):
        self.grid.resetTimer()
    
    def getTimer(self):
        pass
        # print(f"========> data transfer uses: {self.grid.getTimer()} secs")

    def extra_repr(self):
        return f'channels={self.channels}, world_size={self.world_size.tolist()}'


''' Dense 3D grid
'''
class DenseGrid(nn.Module):
    def __init__(self, channels, world_size, xyz_min, xyz_max, **kwargs):
        super(DenseGrid, self).__init__()
        self.channels = channels
        self.world_size = world_size
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.grid = nn.Parameter(torch.zeros([1, channels, *world_size]))

    def forward(self, xyz):
        '''
        xyz: global coordinates to query
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        out = F.grid_sample(self.grid, ind_norm, mode='bilinear', align_corners=True)
        out = out.reshape(self.channels,-1).T.reshape(*shape,self.channels)
        if self.channels == 1:
            out = out.squeeze(-1)
        return out

    def scale_volume_grid(self, new_world_size):
        if self.channels == 0:
            self.grid = nn.Parameter(torch.zeros([1, self.channels, *new_world_size]))
        else:
            self.grid = nn.Parameter(
                F.interpolate(self.grid.data, size=tuple(new_world_size), mode='trilinear', align_corners=True))

    def total_variation_add_grad(self, wx, wy, wz, dense_mode):
        '''Add gradients by total variation loss in-place'''
        total_variation_cuda.total_variation_add_grad(
            self.grid, self.grid.grad, wx, wy, wz, dense_mode)

    def get_dense_grid(self):
        return self.grid

    @torch.no_grad()
    def __isub__(self, val):
        self.grid.data -= val
        return self

    def extra_repr(self):
        return f'channels={self.channels}, world_size={self.world_size.tolist()}'

    @torch.no_grad()
    def setValuesOn_bymask(self, mask, val):
        self.grid[mask] = val
    
    def load_from(self, loaddir):
        raise TypeError

    def save_to(self, savedir):
        raise TypeError
    
    def resetTimer(self):
        pass
    
    def getTimer(self):
        pass


''' Mask grid
It supports query for the known free space and unknown space.
'''
class MaskGrid(nn.Module):
    def __init__(self, path=None, mask_cache_thres=None, mask=None, xyz_min=None, xyz_max=None):
        super(MaskGrid, self).__init__()
        if path is not None:
            st = torch.load(path)
            self.mask_cache_thres = mask_cache_thres
            if 'density.grid' in st['model_state_dict'].keys():
                density = F.max_pool3d(st['model_state_dict']['density.grid'], kernel_size=3, padding=1, stride=1)
            else:
                tmpmodel = DensityVDB([2,2,2],1)
                tmpmodel.load_from(path[:-9] + 'density.vdb')
                densitygrid = torch.tensor(tmpmodel.get_dense_grid()).reshape(st['model_kwargs']['mask_cache_world_size']+[-1]).permute(3,0,1,2).unsqueeze(0).contiguous()
                density = F.max_pool3d(densitygrid, kernel_size=3, padding=1, stride=1)
            alpha = 1 - torch.exp(-F.softplus(density + st['model_state_dict']['act_shift']) * st['model_kwargs']['voxel_size_ratio'])
            mask = (alpha >= self.mask_cache_thres).squeeze(0).squeeze(0)
            xyz_min = torch.Tensor(st['model_kwargs']['xyz_min'])
            xyz_max = torch.Tensor(st['model_kwargs']['xyz_max'])
        else:
            mask = mask.bool()
            xyz_min = torch.Tensor(xyz_min)
            xyz_max = torch.Tensor(xyz_max)

        self.register_buffer('mask', mask)
        xyz_len = xyz_max - xyz_min
        self.register_buffer('xyz2ijk_scale', (torch.Tensor(list(mask.shape)) - 1) / xyz_len)
        self.register_buffer('xyz2ijk_shift', -xyz_min * self.xyz2ijk_scale)

    @torch.no_grad()
    def forward(self, xyz):
        '''Skip know freespace
        @xyz:   [..., 3] the xyz in global coordinate.
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(-1, 3)
        mask = render_utils_cuda.maskcache_lookup(self.mask, xyz, self.xyz2ijk_scale, self.xyz2ijk_shift)
        mask = mask.reshape(shape)
        return mask

    def extra_repr(self):
        return f'mask.shape=list(self.mask.shape)'

