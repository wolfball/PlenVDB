import torch
import pyopenvdb as vdb
import argparse
import time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loadname", type=str, default='')
    parser.add_argument("--savename", type=str, default='')
    return parser.parse_args()

def main():
    args = get_args()
    ckpt = torch.load(args.loadname)
    density = ckpt['model_state_dict']['density.grid'].detach().cpu().numpy()[0][0]
    print(density.shape)
    tree = vdb.FloatGrid()
    t = time.time()
    tree.copyFromArray(density)
    print("Time used:", time.time()-t)
    vdb.write(args.savename, grids=[tree])
    return



if __name__ == '__main__':
    main()