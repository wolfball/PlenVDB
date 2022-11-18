grid_wldcoord = model.grid_wldcoord.reshape(-1,1,3)   # [reso**3, 1, 3]
grid_rays_o = rays_o_tr.repeat(grid_wldcoord.shape[0],1,1) # [reso**3, 100, 3]
grid_rays_d = grid_wldcoord - grid_rays_o # [reso**3, 100, 3]
grid_viewdirs = grid_rays_d / torch.norm(grid_rays_d, dim=-1, keepdim=True) # [reso**3, 100, 3]
grid_dirs = torch.sum(torch.linalg.inv(poses[:,:3,:3]).unsqueeze(0) * grid_rays_d.unsqueeze(2), -1) #[reso**3, 100, 3]
grid_ijs = (-grid_dirs[..., :2] / grid_dirs[..., 2:]) * torch.tensor([K[0][0], -K[1][1]]) + torch.tensor([K[0][2], K[1][2]]) #[reso**3, 100, 2]
# mask_ijs = (grid_ijs[..., 0]>=0) & (grid_ijs[..., 0]<=W) & (grid_ijs[..., 1]>=0) & (grid_ijs[..., 1]<=H) #[reso**3,100]
print(images.shape, grid_ijs.shape)
print(grid_ijs.max(), grid_ijs.min())
grid_rgb_tr = F.grid_sample(images.permute(0,3,1,2).contiguous(), \
                            (grid_ijs.permute(1,0,2).unsqueeze(1).contiguous()-torch.tensor([H/2,W/2]))/torch.tensor([H,W]), \
                            align_corners=True).permute(3,0,1,2).squeeze().contiguous() # [reso**3,100,3]
print(grid_rgb_tr.shape)
print(grid_rgb_tr.is_contiguous())