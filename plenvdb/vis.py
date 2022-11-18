import matplotlib.pyplot as plt
import torch 
import numpy as np
import os

basedir = "logs/nerf_synthetic/"
scenes = ["chair", "drums", "ficus", "lego"] # "hotdog", "materials", "mic", "ship"
den_timesteps = []
den_psnrs = []
vdb_timesteps = []
vdb_psnrs = []

for scn in scenes:
    den_timestep = []
    den_psnr = []
    vdb_timestep = []
    vdb_psnr = []
    denckpt = torch.load(os.path.join(basedir, "dvgo_"+scn+"_den", "fine_last.tar"))['time_recoder']
    vdbckpt = torch.load(os.path.join(basedir, "dvgo_"+scn, "fine_last.tar"))['time_recoder']
    assert(len(denckpt) == 40)
    assert(len(vdbckpt) == 40)
    for i in range(40):
        den_timestep.append(denckpt[i][1])
        den_psnr.append(denckpt[i][2])
        vdb_timestep.append(vdbckpt[i][1])
        vdb_psnr.append(vdbckpt[i][2])
    den_timesteps.append(den_timestep)
    den_psnrs.append(den_psnr)
    vdb_timesteps.append(vdb_timestep)
    vdb_psnrs.append(vdb_psnr)

den_timesteps = torch.cat([torch.tensor([0]), torch.tensor(den_timesteps).mean(0)])
den_psnrs = torch.cat([torch.tensor([12.0225]), torch.tensor(den_psnrs).mean(0)]) 
vdb_timesteps = torch.cat([torch.tensor([0]), torch.tensor(vdb_timesteps).mean(0)])
vdb_psnrs = torch.cat([torch.tensor([12.0225]), torch.tensor(vdb_psnrs).mean(0)]) 

print(den_timesteps.shape, den_psnrs.shape, vdb_timesteps.shape, vdb_psnrs.shape)

nerf_tpsnr = torch.load("nerf_tpsnr.tar")
nerf_tpsnr = np.concatenate([np.array([[0.16349786520004272, 8.588096141815186]]), nerf_tpsnr])
plt.plot(nerf_tpsnr[:10, 0]/60/60, nerf_tpsnr[:10, 1])

# effnerf_tpsnr = torch.load("effnerf_tpsnr.tar")
# plt.plot(effnerf_tpsnr[:20, 0]/60/60, effnerf_tpsnr[:20, 1])

plenoxels_tpsnr = torch.load("plenoxels_tpsnr.tar")
plt.plot(plenoxels_tpsnr[:10, 0]/60/60, plenoxels_tpsnr[:10, 1])

plt.plot(den_timesteps/60/60, den_psnrs)
plt.plot(vdb_timesteps/60/60, vdb_psnrs)
# plt.plot([0, max(den_timesteps[-1], vdb_timesteps[-1])], [max(den_psnrs[-1], vdb_psnrs[-1])]*2)
plt.title("Training Speed")
plt.xlabel("Time(h)")
plt.ylabel("Training PSNR")
plt.legend(["NeRF(Pytorch)", "PlenOxels", "DVGO", "ours"])
plt.tight_layout()
plt.grid()
plt.savefig("nerf_t_psnr.png")
print(den_timesteps[-1], den_psnrs[-1])
print(vdb_timesteps[-1], vdb_psnrs[-1])