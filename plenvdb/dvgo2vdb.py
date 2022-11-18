import torch 
import pyopenvdb as vdb

pth = 'logs/nerf_synthetic/dvgo_mic_ori/fine_last.tar'

ckpt = torch.load(pth)
grid = ckpt['model_state_dict']['density.grid'].permute(0,2,3,4,1).squeeze().contiguous()
tree = vdb.FloatGrid()
tree.copyFromArray(grid.detach().cpu().numpy())
vdb.write('logs/nerf_synthetic/dvgo_mic_ori/fine_density.vdb', grids=[tree])

