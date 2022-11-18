import torch
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default='')
    parser.add_argument("--singlefile", action='store_true')
    parser.add_argument("--tarname", type=str, default='fine_last.tar')
    parser.add_argument("--thres", type=float, default=1e-3, help="Threshold of density")
    return parser.parse_args()

def compute_sparsity(fname, args):
    if os.path.exists(os.path.join(args.filename, fname, args.tarname)):
        model = torch.load(os.path.join(args.filename, fname, args.tarname))
        if 'density.grid' in model['model_state_dict'].keys():
            dsgrid = model['model_state_dict']['density.grid']
            eptnum = torch.sum(dsgrid < args.thres)
            gridsz = torch.tensor(dsgrid.shape).prod()
            print(f"The sparsity of <{fname}> is [{eptnum / gridsz}] within grid size [{dsgrid.squeeze().shape}]")
        return
    print(f"<{fname}> dont have grid!!!")


def main():
    args = get_args()
    if len(args.filename) == 0:
        print("Please indicate the file name")
        return
    print(f">> File path = {args.filename}")
    print(f">> Threshold = {args.thres}")
    if args.singlefile:
        compute_sparsity(args.filename, args)
    else:
        for fname in os.listdir(args.filename):
            compute_sparsity(fname, args)


if __name__ == '__main__':
    main()