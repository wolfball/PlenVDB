import torch
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loaddir", type=str, default='')
    parser.add_argument("--singlefile", action='store_true')
    parser.add_argument("--tarname", type=str, default='fine_last.tar')
    parser.add_argument("--thres", type=float, default=1e-3, help="Threshold of density")
    parser.add_argument("--savedir", type=str, default='')
    parser.add_argument("--loadtxt", type=str, default='')

    return parser.parse_args()

class StatAnaly:
    def __init__(self, fnames, args):
        self.args = args
        self.modeldirs = []
        self.fnames = []
        self.savedir = os.path.join(args.savedir, 'stat_res')
        self.datas = []
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir, exist_ok=True)
        for fname in fnames:
            self.modeldirs.append(os.path.join(args.loaddir, fname, args.tarname))
            if os.path.exists(self.modeldirs[-1]):
                model = torch.load(self.modeldirs[-1])
                if 'density.grid' in model['model_state_dict'].keys():
                    self.datas.append(model['model_state_dict']['density.grid'])
                    self.fnames.append(fname)
                    continue
            self.modeldirs.pop(-1)

        assert len(self.modeldirs) == len(self.fnames) and len(self.fnames) == len(self.datas)
        print(f"* {len(self.datas)} in {len(fnames)} density grids are loaded...")
    
    def compute_sparsity(self, thres):
        for data in self.datas:
            eptnum = torch.sum(data < thres)
            gridsz = torch.tensor(data.shape).prod()
            print(f"The sparsity is [{eptnum / gridsz}] within grid size [{data.squeeze().shape}]")

    def print_descrip(self):
        for data in self.datas:
            data = pd.Series(data.detach().cpu().numpy().reshape(-1))
            print(data.describe())
    
    def plot_trend(self, reso=50):
        idx = [i/reso*0.5+0.5 for i in range(reso)]
        for data, fname in zip(self.datas, self.fnames):
            data = torch.sort(data.detach().cpu().reshape(-1))[0]
            N = len(data)
            arr = [data[int(N*(i/reso*0.5+0.5))] for i in range(reso)]
            plt.plot(idx, arr, label=fname)
        plt.grid()
        # plt.legend(bbox_to_anchor=(1.05, 1.0))
        plt.tight_layout()
        plt.savefig(os.path.join(self.savedir, "trend.png"))




def main():
    args = get_args()
    if len(args.loadtxt) != 0:
        with open(args.loadtxt, "r") as f:
            filenames = [fname.strip("\n") for fname in f.readlines()]
        stat = StatAnaly(filenames, args)
        stat.plot_trend()
        return


    if len(args.loaddir) == 0:
        print("Please indicate the file name")
        return
    print(f"* Density Stat Analysis for {args.loaddir}")
    if args.singlefile:
        stat = StatAnaly([''], args)
        stat.compute_sparsity(args.thres)
        # stat.print_descrip()
        # stat.plot_violin()
        stat.plot_trend()
    else:
        args.savedir = args.loaddir
        stat = StatAnaly(os.listdir(args.loaddir), args)
        stat.plot_trend()



if __name__ == '__main__':
    main()