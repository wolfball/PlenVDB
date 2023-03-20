import numpy as np
import torch
import os
from tqdm import tqdm
from torch import nn
import sys
sys.path.append("../openvdb/build/openvdb/openvdb/python/")
import pyopenvdb as vdb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--no_cpr', action='store_true', help='whether to use compression')
parser.add_argument('--basedir', type=str, default="logs/nerf_synthetic/")
parser.add_argument('--scenes', nargs='+')
args = parser.parse_args()

cpr = not args.no_cpr
basedir = args.basedir
scenes = args.scenes

# scenes = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]

# basedir = "logs/nsvf_synthetic/"
# scenes = ["Bike", "Lifestyle", "Palace", "Robot", "Spaceship", "Steamtrain", "Toad", "Wineholder"]

# basedir = "logs/blended_mvs/"
# scenes = ["Character", "Fountain", "Jade", "Statues"]

# basedir = "logs/deepvoxels/"
# scenes = ["armchair", "cube", "greek", "vase"]

# op = nn.MaxPool3d(3, stride=1, padding=1, dilation=1, return_indices=False, ceil_mode=False)

for scene in scenes[:]:
    ckptdir = os.path.join(basedir, "dvgo_"+scene)
    denTree = vdb.readAll(os.path.join(ckptdir, "finedensity.vdb"))[0][0]
    colTrees = vdb.readAll(os.path.join(ckptdir, "finecolor.vdb"))[0]
    ckpt = torch.load(os.path.join(ckptdir, "fine_last.tar"))
    mask = ckpt['model_state_dict']['mask_cache.mask'].detach().cpu()
    thres = ckpt['model_kwargs']['fast_color_thres']
    interval = ckpt['model_kwargs']['voxel_size_ratio'].item()
    act_shift = ckpt['model_state_dict']['act_shift'].item()
    reso = mask.shape
    rSum = reso[0] * reso[1] * reso[2]

    # # important!!!
    # mask = op(mask.reshape(1,1,reso[0],reso[1],reso[2]).float()).reshape(reso).bool()
                        
    N = torch.sum(mask).item()
    print(N, N/rSum)
    idxs = np.arange(1, N+1)
    idxsArr = np.zeros(reso)
    idxsArr[mask] = idxs

    denArr = np.zeros(reso)
    denTree.copyToArray(denArr)
    dendata = denArr[mask]
    dendata = np.concatenate([[0], dendata])
    # print(idxsArr.shape, dendata.shape)

    # tmp = 1 - np.power(1 + np.exp(denArr + act_shift), -interval)
    # alphamask = tmp > thres
    # print(np.sum(alphamask&mask.numpy()), np.sum(alphamask&mask.numpy())/rSum)

    # assert 1 - np.power(1 + np.exp(act_shift), -interval) > thres
    # print(f"density vdb reset {np.sum(denArr[~mask]<0)} voxels")
    # denArr[~mask] = 0
    # denTree = vdb.FloatGrid()
    # denTree.copyFromArray(denArr)
    coldata = []
    for i in range(len(colTrees)):
        colArr = np.zeros([*reso, 3])
        colTrees[i].copyToArray(colArr)
        coldata.append(colArr[mask])
        # print(f"color vdb [{i}] reset {np.sum(colArr[~mask] < 0)//3} voxels")
        # colArr[~mask][:] = 0
        # colTrees[i] = vdb.Vec3SGrid()
        # colTrees[i].copyFromArray(colArr)
    coldata = np.concatenate(coldata, -1)
    coldata = np.concatenate([[[0]*len(colTrees)*3], coldata])
    print(coldata.shape)

    idxsTree = vdb.FloatGrid()
    idxsTree.copyFromArray(idxsArr)
    vdb.write(os.path.join(ckptdir, "mergedidxs.vdb"), grids=[idxsTree])

    if not cpr:
        dendata = torch.tensor(dendata).float().numpy()
        coldata = torch.tensor(coldata).float().numpy()
        vdbdata = {"den": dendata, "col": coldata}
        print(dendata.dtype, coldata.dtype)
        np.savez(os.path.join(ckptdir, "mergeddata"), **vdbdata)
    else:
        dendata = torch.tensor(dendata).half().numpy()
        coldata = torch.tensor(coldata).half().numpy()
        vdbdata = {"den": dendata, "col": coldata}
        print(dendata.dtype, coldata.dtype)
        np.savez_compressed(os.path.join(ckptdir, "mergeddata"), **vdbdata)

    # maskTree = vdb.FloatGrid()
    # vdbmask = mask.float().numpy()
    # # vdbmask = (mask & alphamask).float().numpy() 
    # print(np.sum(vdbmask), np.sum(vdbmask)/rSum)

    # print(vdbmask.dtype)
    # print(vdbmask.shape)
    # maskTree.copyFromArray(vdbmask)
    # vdb.write(os.path.join(ckptdir, "finemask.vdb"), grids=[maskTree])
