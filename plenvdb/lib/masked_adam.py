import os
import torch
# from torch._C import per_channel_symmetric
from torch.utils.cpp_extension import load

import sys
sys.path.append("./lib/vdb/build/")
from plenvdb import DensityOpt, ColorOpt

parent_dir = os.path.dirname(os.path.abspath(__file__))
sources=['cuda/adam_upd.cpp', 'cuda/adam_upd_kernel.cu']
adam_upd_cuda = load(
        name='adam_upd_cuda',
        sources=[os.path.join(parent_dir, path) for path in sources],
        verbose=True)


class VDBAdam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        assert len(params) == 2
        self.has_per_lr = False
        self.skip_density, self.skip_color = False, False
        for param_group in params:
            param = param_group['params']
            lr = param_group['lr']
            if param.channels == 1:
                self.densityparam = param
                self.densityOpt = DensityOpt(param.grid, lr, eps, betas[0], betas[1])
                self.skip_density = param_group['skip_zero_grad']
            else:
                self.colorparam = param
                self.colorOpt = ColorOpt(param.grid, lr, eps, betas[0], betas[1])
                self.skip_color = param_group['skip_zero_grad']
    
    def set_pervoxel_lr(self, count):
        self.has_per_lr = True
        per_lr = (count.float() / count.max()).reshape(-1).detach().cpu().numpy()
        self.densityOpt.set_pervoxel_lr(per_lr)
        # self.colorOpt.set_pervoxel_lr(per_lr)
    
    def update_lr(self, factor):
        self.densityOpt.update_lr(factor)
        self.colorOpt.update_lr(factor)
    
    def zero_grad(self):
        self.densityOpt.zero_grad()
        self.colorOpt.zero_grad()

    def step(self):
        # stepmode: 0 for default, 1 for skipzerograd, 2 for perlr
        if self.has_per_lr:
            self.densityOpt.step(2)
        elif self.skip_density:
            self.densityOpt.step(1)
        else:
            self.densityOpt.step(0)

        if self.skip_color:
            self.colorOpt.step(1)
        else:
            self.colorOpt.step(0)
    
    def load_from(self, loadpath):
        ckpt = torch.load(loadpath + "vdbopt.tar")
        self.densityOpt.setStep(ckpt['dstep'])
        self.densityOpt.setLr(ckpt['dlr'])
        self.densityOpt.setBeta0(ckpt['dbeta0'])
        self.densityOpt.setBeta1(ckpt['dbeta1'])
        self.densityOpt.setEps(ckpt['deps'])
        self.colorOpt.setStep(ckpt['cstep'])
        self.colorOpt.setLr(ckpt['clr'])
        self.colorOpt.setBeta0(ckpt['cbeta0'])
        self.colorOpt.setBeta1(ckpt['cbeta1'])
        self.colorOpt.setEps(ckpt['ceps'])
        self.densityOpt.load_from(loadpath + 'density_')
        self.colorOpt.load_from(loadpath + 'color_')

    def save_to(self, savepath):
        torch.save(
            {'dlr': self.densityOpt.getLr(), 'clr': self.colorOpt.getLr(),
             'deps': self.densityOpt.getEps(), 'ceps': self.colorOpt.getEps(),
             'dbeta0': self.densityOpt.getBeta0(), 'cbeta0': self.colorOpt.getBeta0(),
             'dbeta1': self.densityOpt.getBeta1(), 'cbeta1': self.colorOpt.getBeta1(),
             'dstep': self.densityOpt.getStep(), 'cstep': self.colorOpt.getStep(),
             }, 
            savepath + "vdbopt.tar")
        self.densityOpt.save_to(savepath + 'density_')
        self.colorOpt.save_to(savepath + 'color_')
        
        

        

''' Extend Adam optimizer
1. support per-voxel learning rate
2. masked update (ignore zero grad) which speeduping training
'''
class MaskedAdam(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps)
        self.per_lr = None
        super(MaskedAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MaskedAdam, self).__setstate__(state)

    def set_pervoxel_lr(self, count):
        assert self.param_groups[0]['params'][0].shape == count.shape
        self.per_lr = count.float() / count.max()

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            skip_zero_grad = group['skip_zero_grad']

            for param in group['params']:
                if param.grad is not None:
                    state = self.state[param]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)

                    state['step'] += 1

                    if self.per_lr is not None and param.shape == self.per_lr.shape:
                        adam_upd_cuda.adam_upd_with_perlr(
                                param, param.grad, state['exp_avg'], state['exp_avg_sq'], self.per_lr,
                                state['step'], beta1, beta2, lr, eps)
                    elif skip_zero_grad:
                        adam_upd_cuda.masked_adam_upd(
                                param, param.grad, state['exp_avg'], state['exp_avg_sq'],
                                state['step'], beta1, beta2, lr, eps)
                    else:
                        adam_upd_cuda.adam_upd(
                                param, param.grad, state['exp_avg'], state['exp_avg_sq'],
                                state['step'], beta1, beta2, lr, eps)