class DirectVoxGO:
    def voxel_count_loss(self, rays_o_tr, rays_d_tr, near, far, stepsize, loss_i):
        # print('* Voxel_count_loss start')
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        # eps_time = time.time()
        N_samples = int(np.array((3*self.reso**2)**(0.5)+1) / stepsize) + 1
        rng = torch.arange(N_samples)[None].float()
        count = torch.zeros([1]+[self.reso]*3)
        device = rng.device

        ones = torch.zeros([1, *count.shape])
        ones.requires_grad_()
        rays_o = rays_o_tr.to(device)  # [N,3]
        rays_d = rays_d_tr.to(device) # [N,3]
        vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.xyz_max - rays_o) / vec
        rate_b = (self.xyz_min - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
        # t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)
        step = stepsize * self.voxellen * rng
        interpx = (t_min[...,None] + step/rays_d.norm(dim=-1,keepdim=True))
        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]  # [N, Nsample, 3]
        out = F.grid_sample(ones, self.wld2idx01(rays_pts).flip((-1,)).reshape([1,1,*rays_pts.shape]), mode='bilinear', align_corners=True)
        
        (out[0][0][0] * loss_i.detach().unsqueeze(1)).sum().backward()
        with torch.no_grad():
            count += ones.grad[0]

        # eps_time = time.time() - eps_time
        # print('dvgo: voxel_count_loss finish (eps time:', eps_time, 'sec)')
        self.density.set_pervoxel_lr(count.unsqueeze(-1), 3.0)
        return count

    def voxel_count_loss_pts(self, ray_pts, ray_loss):
        # print('* Voxel_count_loss start')
        count = torch.zeros([1]+[self.reso]*3)
        
        for pts, loss in zip(ray_pts.split(81920), ray_loss.split(81920)):
            ones = torch.zeros([1, *count.shape])
            ones.requires_grad_()
            out = F.grid_sample(ones, self.wld2idx01(pts).flip((-1,)).reshape([1,1,1,-1,3]), mode='bilinear', align_corners=True)
            (out[0][0][0] * loss.detach().unsqueeze(1)).sum().backward()
            with torch.no_grad():
                count += ones.grad[0]

        # eps_time = time.time() - eps_time
        # print('dvgo: voxel_count_loss finish (eps time:', eps_time, 'sec)')
        self.density.set_pervoxel_lr(count.unsqueeze(-1), 2)
        return count
    
    def voxel_count_loss_plus(self, pts, loss):
        count = torch.zeros_like(self.density.get_dense_grid())
        ones = grid.DenseGrid(1, self.world_size, self.xyz_min, self.xyz_max)
        (ones(pts) * loss.detach().unsqueeze(1)).sum().backward()
        with torch.no_grad():
            count += ones.grid.grad[0]
        return count
