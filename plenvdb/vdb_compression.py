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
parser.add_argument('--basedir', type=str, default="logs/nerf_synthetic/")
parser.add_argument('--scenes', nargs='+')
args = parser.parse_args()

basedir = args.basedir
scenes = args.scenes

for scene in scenes[:]:
    ckptdir = os.path.join(basedir, "vdb_"+scene)
    denTree = vdb.readAll(os.path.join(ckptdir, "finedensity.vdb"))[0][0]
    colTrees = vdb.readAll(os.path.join(ckptdir, "finecolor.vdb"))[0]
    ckpt = torch.load(os.path.join(ckptdir, "fine_last.tar"))
    mask = ckpt['model_state_dict']['mask_cache.mask'].detach().cpu()
    reso = mask.shape
    rSum = reso[0] * reso[1] * reso[2]
    
    # count active voxels
    N = torch.sum(mask).item()
    print(f"[{scene}] Total active voxels: {N} , sparsity: {N/rSum}")
    idxs = np.arange(1, N+1)
    idxsArr = np.zeros(reso)
    idxsArr[mask] = idxs

    # get density data
    denArr = np.zeros(reso)
    denTree.copyToArray(denArr)
    dendata = denArr[mask]
    dendata = np.concatenate([[0], dendata])

    # get color data
    coldata = []
    for i in range(len(colTrees)):
        colArr = np.zeros([*reso, 3])
        colTrees[i].copyToArray(colArr)
        coldata.append(colArr[mask])
    coldata = np.concatenate(coldata, -1)
    coldata = np.concatenate([[[0]*len(colTrees)*3], coldata])

    # create MergedIndexVDB
    idxsTree = vdb.FloatGrid()
    idxsTree.copyFromArray(idxsArr)
    vdb.write(os.path.join(ckptdir, "mergedidxs.vdb"), grids=[idxsTree])

    # save core data to float16
    dendata = torch.tensor(dendata).half().numpy()
    coldata = torch.tensor(coldata).half().numpy()
    vdbdata = {"den": dendata, "col": coldata}
    np.savez_compressed(os.path.join(ckptdir, "mergeddata"), **vdbdata)
