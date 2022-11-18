viewdirs = viewdirs[ray_id]
N_k0 = k0.shape[0]
rgb = []
N_batch = 204800
for i in range(0, N_k0, N_batch):
    rgb_raw = k0[i:i+N_batch]
    rgb.append(eval_sh(2, rgb_raw.reshape(
        *rgb_raw.shape[:-1],
        -1,
        (2 + 1) ** 2), viewdirs[i:i+N_batch]))
# print(len(rgb))
rgb = torch.cat(rgb, dim=0)