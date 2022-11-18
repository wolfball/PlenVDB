import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange
import pyopenvdb as vdb
import mmcv
import imageio
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import utils, dvgo, dcvgo, dmpigo
from lib.load_data import load_data
from lib.dvgo import Alphas2Weights
from lib.dmpigo import create_full_step_id

from torch_efficient_distloss import flatten_eff_distloss
import pyopenvdb as vdb
import matplotlib.pyplot as plt

def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_coarse_only", type=str, default='')
    parser.add_argument("--expname")

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_stat", action='store_true')
    parser.add_argument("--render_validpts", action='store_true')
    parser.add_argument("--render_video_flipy", action='store_true')
    parser.add_argument("--render_video_rot90", default=0, type=int)
    parser.add_argument("--render_video_factor", type=float, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--dump_images", action='store_true')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')
    parser.add_argument("--render_sigma_thres", type=float, default=None) ####

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')
    return parser


@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, dump_images=False,
                      render_factor=0, render_video_flipy=False, render_video_rot90=0,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW = (HW/render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor

    rgbs = []
    depths = []
    bgmaps = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []

    for i, c2w in enumerate(tqdm(render_poses)):

        H, W = HW[i]
        K = Ks[i]
        c2w = torch.Tensor(c2w)
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'depth', 'alphainv_last']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()
        bgmap = render_result['alphainv_last'].cpu().numpy()

        rgbs.append(rgb)
        depths.append(depth)
        bgmaps.append(bgmap)
        
        if i==0:
            print('Testing', rgb.shape)

        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))

    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
        if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')

    if render_video_flipy:
        for i in range(len(rgbs)):
            rgbs[i] = np.flip(rgbs[i], axis=0)
            depths[i] = np.flip(depths[i], axis=0)
            bgmaps[i] = np.flip(bgmaps[i], axis=0)

    if render_video_rot90 != 0:
        for i in range(len(rgbs)):
            rgbs[i] = np.rot90(rgbs[i], k=render_video_rot90, axes=(0,1))
            depths[i] = np.rot90(depths[i], k=render_video_rot90, axes=(0,1))
            bgmaps[i] = np.rot90(bgmaps[i], k=render_video_rot90, axes=(0,1))

    if savedir is not None and dump_images:
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    depths = np.array(depths)
    bgmaps = np.array(bgmaps)

    return rgbs, depths, bgmaps


def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_data(cfg.data)

    # remove useless field
    kept_keys = {
            'hwf', 'HW', 'Ks', 'near', 'far', 'near_clip',
            'i_train', 'i_val', 'i_test', 'irregular_shape',
            'poses', 'render_poses', 'images'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict


def _compute_bbox_by_cam_frustrm_bounded(cfg, HW, Ks, poses, i_train, near, far):
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        if cfg.data.ndc:
            pts_nf = torch.stack([rays_o+rays_d*near, rays_o+rays_d*far])
        else:
            pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    return xyz_min, xyz_max

def _compute_bbox_by_cam_frustrm_unbounded(cfg, HW, Ks, poses, i_train, near_clip):
    # Find a tightest cube that cover all camera centers
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        pts = rays_o + rays_d * near_clip
        xyz_min = torch.minimum(xyz_min, pts.amin((0,1)))
        xyz_max = torch.maximum(xyz_max, pts.amax((0,1)))
    center = (xyz_min + xyz_max) * 0.5
    radius = (center - xyz_min).max() * cfg.data.unbounded_inner_r
    xyz_min = center - radius
    xyz_max = center + radius
    return xyz_min, xyz_max

def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, i_train, near, far, **kwargs):
    print('compute_bbox_by_cam_frustrm: start')
    if cfg.data.unbounded_inward:
        xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_unbounded(
                cfg, HW, Ks, poses, i_train, kwargs.get('near_clip', None))

    else:
        xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_bounded(
                cfg, HW, Ks, poses, i_train, near, far)
    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max

@torch.no_grad()
def compute_bbox_by_coarse_geo(model_class, model_path, thres):
    print('compute_bbox_by_coarse_geo: start')
    eps_time = time.time()
    model = utils.load_model(model_class, model_path)
    interp = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, model.world_size[0]),
        torch.linspace(0, 1, model.world_size[1]),
        torch.linspace(0, 1, model.world_size[2]),
    ), -1)
    dense_xyz = model.xyz_min * (1-interp) + model.xyz_max * interp
    density = model.density(dense_xyz)
    alpha = model.activate_density(density)
    mask = (alpha > thres)
    active_xyz = dense_xyz[mask]
    xyz_min = active_xyz.amin(0)
    xyz_max = active_xyz.amax(0)
    print('compute_bbox_by_coarse_geo: xyz_min', xyz_min)
    print('compute_bbox_by_coarse_geo: xyz_max', xyz_max)
    eps_time = time.time() - eps_time
    print('compute_bbox_by_coarse_geo: finish (eps time:', eps_time, 'secs)')
    return xyz_min, xyz_max

def create_new_model(cfg, cfg_model, cfg_train, xyz_min, xyz_max, stage, coarse_ckpt_path):
    model_kwargs = copy.deepcopy(cfg_model)
    num_voxels = model_kwargs.pop('num_voxels')
    if len(cfg_train.pg_scale):
        num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))

    if cfg.data.ndc:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse multiplane images\033[0m')
        model = dmpigo.DirectMPIGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            **model_kwargs)
    elif cfg.data.unbounded_inward:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse contraced voxel grid (covering unbounded)\033[0m')
        model = dcvgo.DirectContractedVoxGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            **model_kwargs)
    else:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse dense voxel grid\033[0m')
        model = dvgo.DirectVoxGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            mask_cache_path=coarse_ckpt_path,
            **model_kwargs)
    model = model.to(device)
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
    return model, optimizer

def load_existed_model(args, cfg, cfg_train, reload_ckpt_path):
    if cfg.data.ndc:
        model_class = dmpigo.DirectMPIGO
    elif cfg.data.unbounded_inward:
        model_class = dcvgo.DirectContractedVoxGO
    else:
        model_class = dvgo.DirectVoxGO
    model = utils.load_model(model_class, reload_ckpt_path).to(device)
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)

    ### The following load the checkpoints
    ### change: torch.load -> vdb.readAll

    model, optimizer, start = utils.load_checkpoint(
            model, optimizer, reload_ckpt_path, args.no_reload_optimizer)
    return model, optimizer, start


def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, data_dict, stage, coarse_ckpt_path=None):
    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
    HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images'
        ]
    ]

    # find whether there is existing checkpoint path
    last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last.tar')
    if args.no_reload:
        reload_ckpt_path = None
    elif args.ft_path:
        reload_ckpt_path = args.ft_path
    elif os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None

    # init model and optimizer
    if reload_ckpt_path is None:
        print(f'scene_rep_reconstruction ({stage}): train from scratch')
        model, optimizer = create_new_model(cfg, cfg_model, cfg_train, xyz_min, xyz_max, stage, coarse_ckpt_path)
        start = 0
        if cfg_model.maskout_near_cam_vox:
            model.maskout_near_cam_vox(poses[i_train,:3,3], near)
    else:
        print(f'scene_rep_reconstruction ({stage}): reload from {reload_ckpt_path}')
        model, optimizer, start = load_existed_model(args, cfg, cfg_train, reload_ckpt_path)

    # init rendering setup
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'rand_bkgd': cfg.data.rand_bkgd,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
    }

    # init batch rays sampler
    def gather_training_rays():
        if data_dict['irregular_shape']:
            rgb_tr_ori = [images[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_train]
        else:
            rgb_tr_ori = images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)

        if cfg_train.ray_sampler == 'in_maskcache':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_in_maskcache_sampling(
                    rgb_tr_ori=rgb_tr_ori,
                    train_poses=poses[i_train],
                    HW=HW[i_train], Ks=Ks[i_train],
                    ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                    model=model, render_kwargs=render_kwargs)
        elif cfg_train.ray_sampler == 'flatten':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_flatten(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        else:
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays(
                rgb_tr=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        index_generator = dvgo.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        batch_index_sampler = lambda: next(index_generator)
        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler

    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler = gather_training_rays()

    # view-count-based learning rate
    if cfg_train.pervoxel_lr:
        def per_voxel_init():
            cnt = model.voxel_count_views(
                    rays_o_tr=rays_o_tr, rays_d_tr=rays_d_tr, imsz=imsz, near=near, far=far,
                    stepsize=cfg_model.stepsize, downrate=cfg_train.pervoxel_lr_downrate,
                    irregular_shape=data_dict['irregular_shape'])
            optimizer.set_pervoxel_lr(cnt)
            model.mask_cache.mask[cnt.squeeze() <= 2] = False
        per_voxel_init()

    if cfg_train.maskout_lt_nviews > 0:
        model.update_occupancy_cache_lt_nviews(
                rays_o_tr, rays_d_tr, imsz, render_kwargs, cfg_train.maskout_lt_nviews)

    # GOGO
    torch.cuda.empty_cache()
    psnr_lst = []
    time0 = time.time()
    global_step = -1
    loss_cnt = 0

    for global_step in trange(1+start, 1+cfg_train.N_iters):

        # renew occupancy grid
        if model.mask_cache is not None and (global_step + 500) % 1000 == 0:
            model.update_occupancy_cache()

        # progress scaling checkpoint
        if global_step in cfg_train.pg_scale:
            n_rest_scales = len(cfg_train.pg_scale)-cfg_train.pg_scale.index(global_step)-1
            cur_voxels = int(cfg_model.num_voxels / (2**n_rest_scales))
            if isinstance(model, (dvgo.DirectVoxGO, dcvgo.DirectContractedVoxGO)):
                model.scale_volume_grid(cur_voxels)
            elif isinstance(model, dmpigo.DirectMPIGO):
                model.scale_volume_grid(cur_voxels, model.mpi_depth)
            else:
                raise NotImplementedError
            optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
            model.act_shift -= cfg_train.decay_after_scale
            torch.cuda.empty_cache()

        # random sample rays
        if cfg_train.ray_sampler in ['flatten', 'in_maskcache']:
            sel_i = batch_index_sampler()
            target = rgb_tr[sel_i]
            rays_o = rays_o_tr[sel_i]
            rays_d = rays_d_tr[sel_i]
            viewdirs = viewdirs_tr[sel_i]
        elif cfg_train.ray_sampler == 'random':
            sel_b = torch.randint(rgb_tr.shape[0], [cfg_train.N_rand])
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
        else:
            raise NotImplementedError

        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)

        # volume rendering
        render_result = model(
            rays_o, rays_d, viewdirs,
            global_step=global_step, is_train=True,
            **render_kwargs)

        loss_i = torch.mean((render_result['rgb_marched']-target)**2, -1)
        # thres_loss = torch.mean(loss_i) * 0.1
        pts, rayids = render_result['ray_pts'], render_result['ray_id']
        print(pts.shape)
        pts_i = pts[rayids]
        loss_i = loss_i[rayids]
        # mask = (loss_i > thres_loss)
        # pts_i = pts_i[mask]
        # loss_i = loss_i[mask]


        if stage == 'fine' and global_step>=4000 and False:
            cnt = model.voxel_count_loss(
                rays_o_tr=rays_o, rays_d_tr=rays_d, near=near, far=far, stepsize=cfg_model.stepsize, loss_i=loss_i)
            if loss_cnt == 0:
                lossarr = cnt
            else:
                lossarr += cnt
            loss_cnt += 1
            if loss_cnt == 7000:
                optimizer.set_pervoxel_lr(lossarr, scale=2.0)
                loss_cnt = 0

            # cnt = model.voxel_count_loss_plus(pts_i, loss_i)
            # optimizer.update_pervoxel_lr(cnt, scale=1.5)
            # if global_step % 5000 == 0 and optimizer.per_lr is not None:
                print(torch.mean(loss_i))
                # print(loss_i[mask].shape)
                tree = vdb.FloatGrid()
                cntarr = optimizer.per_lr[0][0].detach().cpu().numpy()
                tree.copyFromArray(cntarr)
                vdb.write(f"cnt_loss_{global_step}.vdb", grids=[tree])

        # gradient descent step
        optimizer.zero_grad(set_to_none=True)
        loss = cfg_train.weight_main * F.mse_loss(render_result['rgb_marched'], target)
        psnr = utils.mse2psnr(loss.detach())
        if cfg_train.weight_entropy_last > 0:
            pout = render_result['alphainv_last'].clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss
        if cfg_train.weight_nearclip > 0:
            near_thres = data_dict['near_clip'] / model.scene_radius[0].item()
            near_mask = (render_result['t'] < near_thres)
            density = render_result['raw_density'][near_mask]
            if len(density):
                nearclip_loss = (density - density.detach()).sum()
                loss += cfg_train.weight_nearclip * nearclip_loss
        if cfg_train.weight_distortion > 0:
            n_max = render_result['n_max']
            s = render_result['s']
            w = render_result['weights']
            ray_id = render_result['ray_id']
            loss_distortion = flatten_eff_distloss(w, s, 1/n_max, ray_id)
            loss += cfg_train.weight_distortion * loss_distortion
        if cfg_train.weight_rgbper > 0:
            rgbper = (render_result['raw_rgb'] - target[render_result['ray_id']]).pow(2).sum(-1)
            rgbper_loss = (rgbper * render_result['weights'].detach()).sum() / len(rays_o)
            loss += cfg_train.weight_rgbper * rgbper_loss
        loss.backward()

        if global_step<cfg_train.tv_before and global_step>cfg_train.tv_after and global_step%cfg_train.tv_every==0:
            if cfg_train.weight_tv_density>0:
                model.density_total_variation_add_grad(
                    cfg_train.weight_tv_density/len(rays_o), global_step<cfg_train.tv_dense_before)
            if cfg_train.weight_tv_k0>0:
                model.k0_total_variation_add_grad(
                    cfg_train.weight_tv_k0/len(rays_o), global_step<cfg_train.tv_dense_before)

        optimizer.step()
        psnr_lst.append(psnr.item())

        # update lr
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (1/decay_steps)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * decay_factor

        # check log & save
        if global_step%args.i_print==0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            tqdm.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                       f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / '
                       f'Eps: {eps_time_str}')
            psnr_lst = []

        if global_step%args.i_weights==0:
            path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_{global_step:06d}.tar')
            torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', path)

    ### The following and the above are saving ckpts
    ### Change: torch.save -> vdb.write

    if global_step != -1:
        # model_state = model.state_dict()
        # density_grid, k0_grid = model_state['density.grid'], model_state['k0.grid']
        # density_grid = density_grid[0][0].detach().cpu().numpy()
        # k0_grid = k0_grid[0].permute(1,2,3,0).contiguous().detach().cpu().numpy()
        
        # assert density_grid.shape[:3] == k0_grid.shape[:3]
        # grid_shape = density_grid.shape[:3]
        # ijk = [-grid_shape[0]//2, -grid_shape[1]//2, -grid_shape[2]//2]
        # tree_ds = vdb.FloatGrid()
        # tree_ds.copyFromArray(density_grid, ijk=ijk)
        # tree_ds.name = 'density'
        
        # num_vdb = k0_grid.shape[-1] // 3
        # tree_ds['num_k0'] = num_vdb
        # tree_ks = [vdb.Vec3SGrid() for _ in range(num_vdb)]
        # for i in range(num_vdb):
        #     tree_ks[i].name = 'k0_' + str(i)
        #     tree_ks[i].copyFromArray(np.ascontiguousarray(k0_grid[..., i*3:(i+1)*3]), ijk=ijk)
        # vdb.write(last_ckpt_path[:-3]+"vdb", grids=[tree_ds, *tree_ks])
        # del model_state['density.grid'], model_state['k0.grid']
        # torch.save({
        #     'global_step': global_step,
        #     'model_kwargs': model.get_kwargs(),
        #     'model_state_dict': model_state,
        #     'optimizer_state_dict': optimizer.state_dict(),
        # }, last_ckpt_path)
        torch.save({
            'global_step': global_step,
            'model_kwargs': model.get_kwargs(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, last_ckpt_path)
        print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', last_ckpt_path)


def train(args, cfg, data_dict):

    # init
    print('train: start')
    eps_time = time.time()
    os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))

    # coarse geometry searching (only works for inward bounded scenes)
    eps_coarse = time.time()
    xyz_min_coarse, xyz_max_coarse = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)

    if cfg.coarse_train.N_iters > 0:
        scene_rep_reconstruction(
                args=args, cfg=cfg,
                cfg_model=cfg.coarse_model_and_render, cfg_train=cfg.coarse_train,
                xyz_min=xyz_min_coarse, xyz_max=xyz_max_coarse,
                data_dict=data_dict, stage='coarse')
        eps_coarse = time.time() - eps_coarse
        eps_time_str = f'{eps_coarse//3600:02.0f}:{eps_coarse//60%60:02.0f}:{eps_coarse%60:02.0f}'
        print('train: coarse geometry searching in', eps_time_str)
        coarse_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'coarse_last.tar')
    else:
        print('train: skip coarse geometry searching')
        coarse_ckpt_path = None

    # fine detail reconstruction
    eps_fine = time.time()
    if cfg.coarse_train.N_iters == 0:
        xyz_min_fine, xyz_max_fine = xyz_min_coarse.clone(), xyz_max_coarse.clone()
    else:
        xyz_min_fine, xyz_max_fine = compute_bbox_by_coarse_geo(
                model_class=dvgo.DirectVoxGO, model_path=coarse_ckpt_path,
                thres=cfg.fine_model_and_render.bbox_thres)
    scene_rep_reconstruction(
            args=args, cfg=cfg,
            cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
            xyz_min=xyz_min_fine, xyz_max=xyz_max_fine,
            data_dict=data_dict, stage='fine',
            coarse_ckpt_path=coarse_ckpt_path)
    eps_fine = time.time() - eps_fine
    eps_time_str = f'{eps_fine//3600:02.0f}:{eps_fine//60%60:02.0f}:{eps_fine%60:02.0f}'
    print('train: fine detail reconstruction in', eps_time_str)

    eps_time = time.time() - eps_time
    eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
    print('train: finish (eps time', eps_time_str, ')')


if __name__=='__main__':
    total_t = time.time()
    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)

    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()

    # load images / poses / camera settings / data split
    data_dict = load_everything(args=args, cfg=cfg)

    # export scene bbox and camera poses in 3d for debugging and visualization
    if args.export_bbox_and_cams_only:
        print('Export bbox and cameras...')
        xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
        poses, HW, Ks, i_train = data_dict['poses'], data_dict['HW'], data_dict['Ks'], data_dict['i_train']
        near, far = data_dict['near'], data_dict['far']
        if data_dict['near_clip'] is not None:
            near = data_dict['near_clip']
        cam_lst = []
        for c2w, (H, W), K in zip(poses[i_train], HW[i_train], Ks[i_train]):
            rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                    H, W, K, c2w, cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,)
            cam_o = rays_o[0,0].cpu().numpy()
            cam_d = rays_d[[0,0,-1,-1],[0,-1,0,-1]].cpu().numpy()
            cam_lst.append(np.array([cam_o, *(cam_o+cam_d*max(near, far*0.05))]))
        np.savez_compressed(args.export_bbox_and_cams_only,
            xyz_min=xyz_min.cpu().numpy(), xyz_max=xyz_max.cpu().numpy(),
            cam_lst=np.array(cam_lst))
        print('done')
        sys.exit()

    if args.export_coarse_only:
        print('Export coarse visualization...')
        with torch.no_grad():
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'coarse_last.tar')
            model = utils.load_model(dvgo.DirectVoxGO, ckpt_path).to(device)
            alpha = model.activate_density(model.density.get_dense_grid()).squeeze().cpu().numpy()
            rgb = torch.sigmoid(model.k0.get_dense_grid()).squeeze().permute(1,2,3,0).cpu().numpy()
        np.savez_compressed(args.export_coarse_only, alpha=alpha, rgb=rgb)
        print('done')
        sys.exit()

    # train
    if not args.render_only:
        train(args, cfg, data_dict)

    # load model for rendring
    if args.render_test or args.render_train or args.render_video or args.render_stat or args.render_validpts:
        if args.ft_path:
            ckpt_path = args.ft_path
        else:
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
        ckpt_name = ckpt_path.split('/')[-1][:-4]
        if cfg.data.ndc:
            model_class = dmpigo.DirectMPIGO
        elif cfg.data.unbounded_inward:
            model_class = dcvgo.DirectContractedVoxGO
        else:
            model_class = dvgo.DirectVoxGO
        
        ### The following loads the model state
        ### Change: Load model -> Load VDB model
        model = utils.load_model(model_class, ckpt_path).to(device)
        stepsize = cfg.fine_model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
                'render_depth': True,
                'render_sigma_thres': args.render_sigma_thres,
            },
        }

    # render trainset and eval
    if args.render_train:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_train_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        rgbs, depths, bgmaps = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_train']],
                HW=data_dict['HW'][data_dict['i_train']],
                Ks=data_dict['Ks'][data_dict['i_train']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
                savedir=testsavedir, dump_images=args.dump_images,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(1 - depths / np.max(depths)), fps=30, quality=8)

    # render testset and eval
    if args.render_test:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        print('All results are based on sigma threshold: ', args.render_sigma_thres)
        rgbs, depths, bgmaps = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_test']],
                HW=data_dict['HW'][data_dict['i_test']],
                Ks=data_dict['Ks'][data_dict['i_test']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
                savedir=testsavedir, dump_images=args.dump_images,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(1 - depths / np.max(depths)), fps=30, quality=8)

    if args.render_stat:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        print('All results are based on sigma threshold: ', args.render_sigma_thres)

        HW=data_dict['HW'][data_dict['i_test']]
        Ks=data_dict['Ks'][data_dict['i_test']]
        render_poses=data_dict['poses'][data_dict['i_test']]
        gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']]
        render_kwargs = render_viewpoints_kwargs['render_kwargs']
        rgbs = []
        depths = []
        alphainv_lasts = []
        det_imgs = []
        loss_imgs = []
        psnrs = []
        ray_dists = []

        with torch.no_grad():
            for i, c2w in enumerate(tqdm(render_poses)):
                H, W = HW[i]
                K = Ks[i]
                c2w = torch.Tensor(c2w)
                rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                        H, W, K, c2w, cfg.data.ndc, inverse_y=render_kwargs['inverse_y'],
                        flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)

                keys = ['rgb_marched', 'depth', 'alphainv_last']
                rays_o = rays_o.flatten(0,-2)
                rays_d = rays_d.flatten(0,-2)
                viewdirs = viewdirs.flatten(0,-2)
                
                render_result_chunks = [
                    {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
                    for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
                ]
                render_result = {
                    k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
                    for k in render_result_chunks[0].keys()
                }
                rgb = render_result['rgb_marched'].detach().cpu().numpy()
                rgbs.append(rgb)
                depth = render_result['depth'].detach().cpu().numpy()
                depths.append(depth)
                alphainv_last = render_result['alphainv_last'].detach().cpu().numpy()
                alphainv_lasts.append(alphainv_last)
                
                if i==0:
                    print('Testing', rgb.shape)

                loss_img = None
                if gt_imgs is not None:
                    p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
                    psnrs.append(p)
                    loss_img = np.mean((rgb-gt_imgs[i])**2, -1)
                    
                
                rdcs_ray_ids = np.random.choice(H*W, size=[4], replace=False, p=loss_img.reshape(-1)/loss_img.sum())
                ray_dist = {}
                for rdid in rdcs_ray_ids:
                    rayo, rayd, viewdir = rays_o[rdid:rdid+1], rays_d[rdid:rdid+1], viewdirs[rdid:rdid+1]
                    if cfg.data.ndc:
                        pts, ray_id, _, _ = model.sample_ray(rays_o=rayo, rays_d=rayd, **render_kwargs)
                        interval = render_kwargs['stepsize'] * model.voxel_size_ratio
                        density = model.density(pts) + model.act_shift(pts)
                    elif cfg.data.unbounded_inward:
                        pts, _, _ = model.sample_ray(
                            ori_rays_o=rayo, ori_rays_d=rayd, is_train=False, **render_kwargs)
                        interval = render_kwargs['stepsize'] * model.voxel_size_ratio
                        ray_id, step_id = create_full_step_id(pts.shape[:2])
                        pts = pts[0]
                        density = model.density(pts)
                    else:
                        pts, ray_id, _ = model.sample_ray(rays_o=rayo, rays_d=rayd, **render_kwargs)
                        interval = render_kwargs['stepsize'] * model.voxel_size_ratio
                        density = model.density(pts)
                    alpha = model.activate_density(density, interval)
                    # if i == 0:
                    #     print("alpha's shape:", alpha.shape)
                    weight, _ = Alphas2Weights.apply(alpha, ray_id.contiguous(), 1)
                    
                    if len(alpha) != 0:
                        zdepth = torch.sum((pts - rays_o[rdid]).pow(2), -1).pow(0.5)
                        ray_dist[f'({rdid%W},{rdid//W})'] = {'alpha': alpha.cpu(), 'weight': weight.cpu(), 'zdepth': zdepth.cpu(),
                                                             'sigma': density.cpu(), 'x': rdid%W, 'y': rdid//W}
                
                ray_dists.append(ray_dist)
                loss_imgs.append(loss_img)

        if testsavedir is not None and args.dump_images:
            for i in trange(len(rgbs)):
                rgb8 = utils.to8b(rgbs[i])
                filename = os.path.join(testsavedir, '{:03d}.png'.format(i))
                imageio.imwrite(filename, rgb8)

        loss_imgs = np.array(loss_imgs)
        depths = np.array(depths)
        rgbs = np.array(rgbs)
        alphainv_lasts = np.array(alphainv_lasts)
        # os.makedirs(os.path.join(testsavedir, 'lossimgs'), exist_ok=True)
        statimgs = []
        # voxels = model.density.grid.data[0][0][::40,::40,::40].cpu()>0 # (X, Y, Z)
        # xyz_min, xyz_max = model.density.xyz_min.unsqueeze(0), model.density.xyz_max.unsqueeze(0)
        # cam = ((render_poses[:, :3, 3] - xyz_min) / (xyz_max- xyz_min))
        # cam = cam.detach().cpu().numpy()
        # st = torch.arange(5).detach().cpu()/5
        # sx, sy, sz = torch.meshgrid(st, st, st)
        # sh = torch.tensor(voxels.shape)
        for i, lossimg in enumerate(tqdm(loss_imgs)):
            fig = plt.figure(figsize=(12, 12))

            # ax = fig.add_subplot(331, projection="3d")
            # ax.scatter(cam[i, 0], cam[i, 1], cam[i, 2], marker='o', color='red')
            # # ax.scatter(cam[:i, 0], cam[:, 1], cam[:, 2], marker='o')
            # ax.scatter(sx, sy, sz, marker='s')
            # ax.set_title(f'look from {cam[i]}')

            # ax = fig.add_subplot(333)
            # ax.imshow(utils.to8b(alphainv_lasts[i]))
            # ax.set_title("alphainv_last")

            ax1 = fig.add_subplot(335)
            im = ax1.imshow(lossimg, cmap=plt.cm.gist_heat_r)
            plt.colorbar(im)
            ax1.set_title('loss map')

            ax = fig.add_subplot(336)
            ax.imshow(utils.to8b(rgbs[i]))
            ax.set_title('rendered image')

            ax2 = fig.add_subplot(334)
            ax2.imshow(gt_imgs[i])
            for k,v in ray_dists[i].items():
                rect = plt.Rectangle((v['x']-5, v['y']-5), 11, 11, edgecolor='red', fill=False)
                ax2.add_patch(rect)
            ax2.set_title('gt image')

            ax3 = fig.add_subplot(332)
            ax3.imshow(depths[i]/(depths[i].max()+1e-6))
            ax3.set_title('depth map')

            ax4 = fig.add_subplot(337)
            for k,v in ray_dists[i].items():
                ax4.plot(v['zdepth'], v['sigma'], label=k)
            ax4.set_title('sigma')
            
            ax5 = fig.add_subplot(338)
            for k,v in ray_dists[i].items():
                ax5.plot(v['zdepth'], v['alpha'], label=k)
            ax5.set_title('alpha')
            
            ax6 = fig.add_subplot(339)
            for k,v in ray_dists[i].items():
                ax6.plot(v['zdepth'], v['weight'], label=k)
            ax6.set_title('weight')
            plt.tight_layout()
            plt.legend()
            plt.savefig(os.path.join(testsavedir, "last_lossimg.png"))
            with open(os.path.join(testsavedir, "last_lossimg.png"), "rb") as imgin:
                image = np.array(Image.open(imgin), dtype=np.float32) / 255.0
            statimgs.append(image)
            plt.close(fig)
        statimgs = np.array(statimgs)
        imageio.mimwrite(os.path.join(testsavedir, 'stat.mp4'), utils.to8b(statimgs), fps=30, quality=8)
        

    if args.render_validpts:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        print('All results are based on sigma threshold: ', args.render_sigma_thres)

        HW=data_dict['HW'][data_dict['i_test']]
        Ks=data_dict['Ks'][data_dict['i_test']]
        render_poses=data_dict['poses'][data_dict['i_test']]
        gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']]
        render_kwargs = render_viewpoints_kwargs['render_kwargs']
        psnrs = []
        ray_dists = []
        
        with torch.no_grad():
            # rd_idx = np.random.choice(HW[0]*HW[1], size=[1], replace=False).item()
            statimgs = []
            for i, c2w in enumerate(tqdm(render_poses)):
                H, W = HW[i]
                K = Ks[i]
                c2w = torch.Tensor(c2w)
                rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                        H, W, K, c2w, cfg.data.ndc, inverse_y=render_kwargs['inverse_y'],
                        flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)

                render_result_chunks = []

                for row in torch.arange(0, H, H//10):
                    for col in torch.arange(0, W, W//10):
                        ro = rays_o[row:row+H//10, col:col+W//10, :].reshape(-1, 3)
                        rd = rays_d[row:row+H//10, col:col+W//10, :].reshape(-1, 3)
                        vd = viewdirs[row:row+H//10, col:col+W//10, :].reshape(-1, 3)
                        render_result_chunks.append(model(ro, rd, vd, **render_kwargs)['validpts'])

                render_result = np.array(render_result_chunks).reshape(10+int(H%10!=0), 10+int(W%10!=0), -1)
                percent_valid = render_result[..., 1] / (render_result[..., 0] + 1e-6) * 100
                fig = plt.figure(figsize=(8,8))
                ax = fig.add_subplot(121)
                im = ax.matshow(percent_valid, cmap=plt.cm.Greens) 
                plt.colorbar(im)
                for i in range(percent_valid.shape[0]): 
                    for j in range(percent_valid.shape[1]):
                        ax.annotate(f"{int(percent_valid[i,j])}", xy=(j, i), horizontalalignment='center', verticalalignment='center')
                        # plt.annotate(f"{render_result[i,j,1]}/{render_result[i,j,0]}", xy=(i, j), horizontalalignment='center', verticalalignment='center')
                ax.set_ylabel(f'H*{H//10}')
                ax.set_xlabel(f'W*{W//10}')
                ax.set_title(f'Valid Percent, total sparsity: {(np.sum(render_result[...,1])/np.sum(render_result[..., 0])*100):.2f}%')

                ax = fig.add_subplot(122)
                keys = ['rgb_marched']
                rays_o = rays_o.flatten(0,-2)
                rays_d = rays_d.flatten(0,-2)
                viewdirs = viewdirs.flatten(0,-2)
                render_result_chunks = [
                    {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
                    for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
                ]
                render_result = {
                    k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
                    for k in render_result_chunks[0].keys()
                }
                rgb = render_result['rgb_marched'].detach().cpu().numpy()
                ax.imshow(utils.to8b(rgb))
                ax.set_title("rendered rgb")

                plt.savefig(os.path.join(testsavedir, "last_validpct.png"))
                with open(os.path.join(testsavedir, "last_validpct.png"), "rb") as imgin:
                    image = np.array(Image.open(imgin), dtype=np.float32) / 255.0
                statimgs.append(image)
                plt.close()
        statimgs = np.array(statimgs)
        imageio.mimwrite(os.path.join(testsavedir, 'validpcts.mp4'), utils.to8b(statimgs), fps=30, quality=8)
        


    # render video
    if args.render_video:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_video_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        rgbs, depths, bgmaps = render_viewpoints(
                render_poses=data_dict['render_poses'],
                HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                render_factor=args.render_video_factor,
                render_video_flipy=args.render_video_flipy,
                render_video_rot90=args.render_video_rot90,
                savedir=testsavedir, dump_images=args.dump_images,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        import matplotlib.pyplot as plt
        depths_vis = depths * (1-bgmaps) + bgmaps
        dmin, dmax = np.percentile(depths_vis[bgmaps < 0.1], q=[5, 95])
        depth_vis = plt.get_cmap('rainbow')(1 - np.clip((depths_vis - dmin) / (dmax - dmin), 0, 1)).squeeze()[..., :3]
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(depth_vis), fps=30, quality=8)

    print('Done')
    used_t = time.time()-total_t
    print('Total Time: ', used_t)
    torch.save({'RunningTime': used_t}, os.path.join(cfg.basedir, cfg.expname, 'runtime.tar'))

