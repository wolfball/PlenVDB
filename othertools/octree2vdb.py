import svox
import torch
import pyopenvdb as vdb 
import time
import math

sigma_thres = 0.01
density_thres = math.log(math.e ** 0.01 - 1)
ckptdir = "/root/code/VDBNeRF/VDBvsOctree/plenoctree/ckpts/nerf_synthetic/mic/tree_opt.npz"

def get_den_from_svox(t):
    D = 2 ** (t.depth_limit+1)
    arr = (torch.arange(D) + 0.5) / D
    X, Y, Z = torch.meshgrid(arr, arr, arr)
    XYZ = torch.stack([X, Y, Z], -1).cuda().reshape(-1,3).cuda().reshape(-1,3)
    del X,Y,Z,arr
    t1 = time.time()
    res = t(XYZ, world=False)
    print(time.time()-t1)
    return D, res.reshape(D,D,D,-1)



octree = svox.N3Tree.load(ckptdir, map_location="cuda")
D, res = get_den_from_svox(octree)
savedir = f"/root/code/VDBNeRF/VDBvsOctree/plenoctree/mic_512_thres_"

den = res[...,res.shape[-1]-1].contiguous()
res[den<sigma_thres] = 0
del den
# print(den.max())

dentree = vdb.FloatGrid()
dentree.copyFromArray(res[...,res.shape[-1]-1].contiguous().detach().cpu().numpy())
vdb.write(savedir+"density.vdb", grids=[dentree])


coltrees = [vdb.Vec3SGrid() for _ in range((res.shape[-1]-1)//3)]
for i in range((res.shape[-1]-1)//3):
    coltrees[i].copyFromArray(res[..., i*3:(i+1)*3].contiguous().detach().cpu().numpy())
vdb.write(savedir+"color.vdb", grids=coltrees)


