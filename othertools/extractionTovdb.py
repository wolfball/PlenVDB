from numpy.lib.npyio import save
import pyopenvdb as vdb
import os
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from octree.nerf import models

from octree.nerf.models import construct_nerf
from absl import flags
from absl import app
from octree.nerf import models
from octree.nerf import utils
from octree.nerf import datasets
from octree.nerf import sh_proj

FLAGS = flags.FLAGS

utils.define_flags()

flags.DEFINE_string(
    "output",
    "./tree.npz",
    "Output file",
)
flags.DEFINE_string(
    "center",
    "0 0 0",
    "Center of volume in x y z OR single number",
)
flags.DEFINE_string(
    "radius",
    "1.5",
    "1/2 side length of volume",
)
flags.DEFINE_float(
    "alpha_thresh",
    0.01,
    "Alpha threshold to keep a voxel in initial sigma thresholding",
)
flags.DEFINE_float(
    "max_refine_prop",
    0.5,
    "Max proportion of cells to refine",
)
flags.DEFINE_float(
    "z_min",
    None,
    "Discard z axis points below this value, for NDC use",
)
flags.DEFINE_float(
    "z_max",
    None,
    "Discard z axis points above this value, for NDC use",
)
flags.DEFINE_integer(
    "tree_branch_n",
    2,
    "Tree branch factor (2=octree)",
)
flags.DEFINE_integer(
    "init_grid_depth",
    8,
    "Initial evaluation grid (2^{x+1} voxel grid)",
)
flags.DEFINE_integer(
    "samples_per_cell",
    8,
    "Samples per cell in step 2 (3D antialiasing)",
    short_name='S',
)
flags.DEFINE_bool(
    "is_jaxnerf_ckpt",
    False,
    "Whether the ckpt is from jaxnerf or not.",
)
flags.DEFINE_enum(
    "masking_mode",
    "sigma",  # weight mode for vdb is not implemented
    ["sigma", "weight"],
    "How to calculate mask when building the octree",
)
flags.DEFINE_float(
    "weight_thresh",
    0.001,
    "Weight threshold to keep a voxel",
)
flags.DEFINE_integer(
    "projection_samples",
    10000,
    "Number of rays to sample for SH projection.",
)

# Load bbox from dataset
flags.DEFINE_bool(
    "bbox_from_data",
    False,
    "Use bounding box from dataset if possible",
)
flags.DEFINE_float(
    "data_bbox_scale",
    1.0,
    "Scaling factor to apply to the bounding box from dataset (before autoscale), " +
    "if bbox_from_data is used",
)
flags.DEFINE_bool(
    "autoscale",
    True,
    "Automatic scaling, after bbox_from_data",
)
flags.DEFINE_bool(
    "bbox_cube",
    False,
    "Force bbox to be a cube",
)
flags.DEFINE_float(
    "bbox_scale",
    1.0,
    "Scaling factor to apply to the bounding box at the end (after load, autoscale)",
)
flags.DEFINE_float(
    "scale_alpha_thresh",
    0.01,
    "Alpha threshold to keep a voxel in initial sigma thresholding for autoscale",
)
# For integrated eval (to avoid slow load)
flags.DEFINE_bool(
    "eval",
    True,
    "Evaluate after building the octree",
)

def generate_rays(w, h, focal, camtoworlds, equirect=False):
    """
    Generate perspective camera rays. Principal point is at center.
    Args:
        w: int image width
        h: int image heigth
        focal: float real focal length
        camtoworlds: jnp.ndarray [B, 4, 4] c2w homogeneous poses
        equirect: if true, generates spherical rays instead of pinhole
    Returns:
        rays: Rays a namedtuple(origins [B, 3], directions [B, 3], viewdirs [B, 3])
    """
    x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
        np.arange(w, dtype=np.float32),  # X-Axis (columns)
        np.arange(h, dtype=np.float32),  # Y-Axis (rows)
        indexing="xy",
    )

    camera_dirs = np.stack(
        [
            (x - w * 0.5) / focal,
            -(y - h * 0.5) / focal,
            -np.ones_like(x),
        ],
        axis=-1,
    )

    #  camera_dirs = camera_dirs / np.linalg.norm(camera_dirs, axis=-1, keepdims=True)

    c2w = camtoworlds[:, None, None, :3, :3]
    camera_dirs = camera_dirs[None, Ellipsis, None]
    directions = np.matmul(c2w, camera_dirs)[Ellipsis, 0]
    origins = np.broadcast_to(
        camtoworlds[:, None, None, :3, -1], directions.shape
    )
    norms = np.linalg.norm(directions, axis=-1, keepdims=True)
    viewdirs = directions / norms
    return origins, viewdirs


def calculate_grid_weights_plus(dataset, sigmas, reso, center, radius):
    N = reso // 2
    w, h, focal = dataset.w, dataset.h, dataset.focal
    step_size = FLAGS.renderer_step_size
    sigma_thresh = 0.0
    voxelsize = 2 * radius.numpy() / reso
    translate = center.numpy()

    grid_data = sigmas.reshape((reso, reso, reso)).numpy()  # contain negative value!(need sigma thres first?)
    grid_data[grid_data<sigma_thresh] = 0
    maximum_weight = torch.zeros((reso, reso, reso))
    # dataset.camtoworlds.shape = (100, 4, 4)
    origins, viewdirs = generate_rays(w, h, focal, dataset.camtoworlds)  # (100, 800, 800, 3), (100, 800, 800, 3)
    
    # trans to index space
    origins = (origins - translate) / voxelsize + np.array([reso//2, reso//2, reso//2])
    viewdirs = viewdirs / voxelsize
    
    # compute the range of t s.t. o+t*d \in Grid
    tranges1 = - origins / (viewdirs+1e-6)
    tranges2 = (reso - 1 - origins) / (viewdirs+1e-6)
    tranges = np.concatenate([tranges1[..., np.newaxis], tranges2[..., np.newaxis]], axis=-1)
    tranges = np.sort(tranges, axis=-1)  # (100, 800, 800, 3, 2)
    tmins = np.max(np.ascontiguousarray(tranges[..., 0]), axis=-1)
    tmaxs = np.min(np.ascontiguousarray(tranges[..., 1]), axis=-1)
    del tranges1, tranges2, tranges
    for idx in tqdm(range(dataset.size)):
        tmin, tmax = tmins[idx], tmaxs[idx]  # (800, 800) x2
        mask = tmin < tmax
        tmin, tmax = tmin[mask][:, np.newaxis], tmax[mask][:, np.newaxis]  # (M, 1) x2
        origin, viewdir = origins[idx][mask][:, np.newaxis, :], viewdirs[idx][mask][:, np.newaxis, :]  # (M, 1, 3)
        delta = (tmax - tmin) / N  # (M, 1)
        t_sampled = np.arange(0, N)[np.newaxis, :] * delta + tmin  # (M, N)
        
        points = np.around(origin + t_sampled[..., np.newaxis] * viewdir).astype(np.int32)  # (M, N, 3)
        # print(points[1000, :10], points[1000, -10:])
        sigma_sampled = grid_data[points[...,0], points[...,1], points[...,2]]  # (M, N)
        exps = np.exp(-delta*sigma_sampled)  # (M, N)
        Ts = np.concatenate([np.ones((exps.shape[0], 1)), np.cumprod(exps, axis=1)[:, :-1]], axis=1)
        weights = Ts * (1 - exps)  # (M, N)
        grid_weight = torch.zeros((reso, reso, reso)).numpy()
        grid_weight[points[...,0], points[...,1], points[...,2]] = weights
        maximum_weight = torch.max(maximum_weight, torch.from_numpy(grid_weight))

    return maximum_weight

def calculate_grid_weights(dataset, sigmas, reso, center, radius, thres):
    N = 64
    w, h, focal = dataset.w, dataset.h, dataset.focal
    sigma_thresh = 0.0
    voxelsize = 2 * radius.numpy() / reso
    translate = center.numpy()

    grid_data = sigmas.reshape((reso, reso, reso)).numpy()  # contain negative value!(need sigma thres first?)
    grid_data[grid_data<sigma_thresh] = 0
    maximum_weight = torch.zeros((reso, reso, reso), dtype=torch.bool)
    # dataset.camtoworlds.shape = (100, 4, 4)
    origins_all, viewdirs_all = generate_rays(w, h, focal, dataset.camtoworlds)  # (100, 800, 800, 3) x2
    # trans to index space (100, 800, 800, 3)
    dists = np.linalg.norm(origins_all-translate, axis=-1)
    print(dists.shape)
    print(np.max(dists), np.min(dists))
    
    origins_all = (origins_all - translate) / voxelsize + np.array([reso//2, reso//2, reso//2])
    viewdirs_all = viewdirs_all / voxelsize
    


    import time
    for idx in tqdm(range(dataset.size)):
        t = time.time()
        origins, viewdirs = origins_all[idx], viewdirs_all[idx]
        print(time.time()-t)
        t = time.time()
        # compute the range of t s.t. o+t*d \in Grid

        tmin, tmax = tmins[idx], tmaxs[idx]  # (800, 800) x2
        mask = tmin < tmax
        tmin, tmax = tmin[mask][:, np.newaxis], tmax[mask][:, np.newaxis]  # (M, 1) x2
        origin, viewdir = origins[idx][mask][:, np.newaxis, :], viewdirs[idx][mask][:, np.newaxis, :]  # (M, 1, 3)
        delta = (tmax - tmin) / N  # (M, 1)
        t_sampled = np.arange(0, N)[np.newaxis, :] * delta + tmin  # (M, N)
        points = np.around(origin + t_sampled[..., np.newaxis] * viewdir).astype(np.int32)  # (M, N, 3)
        sigma_sampled = grid_data[points[...,0], points[...,1], points[...,2]]  # (M, N)
        exps = np.exp(-delta*sigma_sampled)  # (M, N)
        Ts = np.concatenate([np.ones((exps.shape[0], 1)), np.cumprod(exps, axis=1)[:, :-1]], axis=1)
        weights = Ts * (1 - exps)  # (M, N)
        wei_mask = weights > thres
        points = points[wei_mask]  # (Masked, 3)
        maximum_weight[points[...,0], points[...,1], points[...,2]] = True
        fff

    return maximum_weight


def project_nerf_to_sh(nerf, sh_deg, points):
    """
    Args:
        points: [N, 3]
    Returns:
        coeffs for rgb. [N, C * (sh_deg + 1)**2]
    """
    nerf.use_viewdirs = True

    def _sperical_func(viewdirs):
        # points: [num_points, 3]
        # viewdirs: [num_rays, 3]
        # raw_rgb: [num_points, num_rays, 3]
        # sigma: [num_points]
        raw_rgb, sigma = nerf.eval_points_raw(points, viewdirs, cross_broadcast=True)
        return raw_rgb, sigma

    coeffs, sigma = sh_proj.ProjectFunctionNeRF(
        order=sh_deg,
        sperical_func=_sperical_func,
        batch_size=points.shape[0],
        sample_count=FLAGS.projection_samples,
        device=points.device)

    return coeffs.reshape([points.shape[0], -1]), sigma


def auto_scale(args, center, radius, nerf):
    print('* Step 0: Auto scale')
    reso = 2 ** args.init_grid_depth

    radius = torch.tensor(radius, dtype=torch.float32)
    center = torch.tensor(center, dtype=torch.float32)
    scale = 0.5 / radius
    offset = 0.5 * (1.0 - center / radius)
    arr = (torch.arange(0, reso, dtype=torch.float32) + 0.5) / reso
    xx = (arr - offset[0]) / scale[0]
    yy = (arr - offset[1]) / scale[1]
    zz = (arr - offset[2]) / scale[2]
    
    grid = torch.stack(torch.meshgrid(xx, yy, zz)).reshape(3, -1).T

    out_chunks = []
    for i in tqdm(range(0, grid.shape[0], args.chunk)):
        grid_chunk = grid[i:i+args.chunk].cuda()
        if nerf.use_viewdirs:
            fake_viewdirs = torch.zeros([grid_chunk.shape[0], 3], device=grid_chunk.device)
        else:
            fake_viewdirs = None
        rgb, sigma = nerf.eval_points_raw(grid_chunk, fake_viewdirs)
        del grid_chunk, rgb
        out_chunks.append(sigma.squeeze(-1).detach().cpu())
    sigmas = torch.cat(out_chunks, 0)
    del out_chunks
    print(sigmas.shape, sigmas.min(), sigmas.max())
    approx_delta = 2.0 / reso
    sigma_thresh = -np.log(1.0 - args.scale_alpha_thresh) / approx_delta
    mask = sigmas >= sigma_thresh
    print(approx_delta, sigma_thresh)
    grid = grid[mask]
    print(grid.shape)
    del mask

    lc = grid.min(dim=0)[0] - 0.5 / reso
    uc = grid.max(dim=0)[0] + 0.5 / reso
    return ((lc + uc) * 0.5).tolist(), ((uc - lc) * 0.5).tolist()


def step1(args, vdbgrids, nerf, dataset):
    print('* Step 1: Grid eval')
    reso = 2 ** (args.init_grid_depth + 1)
    center = torch.tensor(vdbgrids[-1]['center'], dtype=torch.float32)
    radius = torch.tensor(vdbgrids[-1]['radius'], dtype=torch.float32)
    print(center.dtype, radius.dtype)
    # build transform
    # take reso=8 for example,
    # the index space is [-4,-3,-2,-1,0,1,2,3]
    # the sampled grid points are [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    transform = vdb.createLinearTransform()
    transform.scale(2 * radius.double().numpy() / reso)
    transform.translate(center.double().numpy())
    vdbgrids[-1].transform = transform

    # offset and scale can transform the 3D grid to fit scene
    offset = 0.5 * (1 - center / radius)
    scale = 0.5 / radius

    arr = (torch.arange(0, reso, dtype=torch.float32) + 0.5) / reso
    xx = (arr - offset[0]) / scale[0]
    yy = (arr - offset[1]) / scale[1]
    zz = (arr - offset[2]) / scale[2]

    grid = torch.stack(torch.meshgrid(xx, yy, zz)).reshape(3, -1).T
    print('init grid', grid.shape)

    ### not quite understand this setting
    approx_delta = 2.0 / reso
    sigma_thresh = -np.log(1.0 - args.alpha_thresh) / approx_delta

    out_chunks = []
    for i in tqdm(range(0, grid.shape[0], args.chunk)):
        grid_chunk = grid[i:i+args.chunk].cuda()
        if nerf.use_viewdirs:
            fake_viewdirs = torch.zeros([grid_chunk.shape[0], 3], device=grid_chunk.device)
        else:
            fake_viewdirs = None
        rgb, sigma = nerf.eval_points_raw(grid_chunk, fake_viewdirs)
        del grid_chunk, rgb
        out_chunks.append(sigma.squeeze(-1).detach().cpu())
    sigmas = torch.cat(out_chunks, 0)
    del out_chunks

    vdbgrids[-2].copyFromArray(sigmas.view((reso, reso, reso)).numpy(), ijk=(-reso//2, -reso//2, -reso//2))

    if args.masking_mode == "sigma":
        mask = sigmas >= sigma_thresh
    elif args.masking_mode == 'weight':
        print ("* Calculating grid weights")
        grid_weights = calculate_grid_weights_plus(dataset,
            sigmas, reso, center, radius,) # FLAGS.weight_thresh)
        mask = grid_weights.reshape(-1) >= FLAGS.weight_thresh
        del grid_weights
    else:
        raise ValueError
    sigmas[~mask] = vdbgrids[-2].background
    vdbgrids[-2].clear()
    vdbgrids[-2].copyFromArray(sigmas.view((reso, reso, reso)).numpy(), ijk=(-reso//2, -reso//2, -reso//2))

    del sigmas
    indexarr = torch.arange(-reso/2, reso/2, dtype=torch.float32)
    indexgrid = torch.stack(torch.meshgrid(indexarr, indexarr, indexarr)).reshape(3, -1).T
    assert indexgrid.shape == grid.shape
    indexgrid[~mask] = torch.tensor([reso, reso, reso], dtype=torch.float32)
    indexgrid = indexgrid.view(reso, reso, reso, -1).contiguous().numpy()  # contiguous() is very important!!!
    vdbgrids[-1].copyFromArray(indexgrid, ijk=(-reso//2, -reso//2, -reso//2))
    grid = grid[mask]
    del mask
    print(grid.shape)
    assert grid.shape[0] == vdbgrids[-1].activeLeafVoxelCount()
    torch.cuda.empty_cache()


def step2(args, vdbgrids, nerf):
    print('* Step 2: AA', args.samples_per_cell)

    transform = vdbgrids[-1].transform
    import time

    t = time.time()
    dimx, dimy, dimz = vdbgrids[-1].evalActiveVoxelDim()
    idxcoords = np.zeros((dimx, dimy, dimz, 3))
    arr_sigmas = np.zeros((dimx, dimy, dimz))
    arr_shs = np.zeros((dimx, dimy, dimz, 3, 16))
    vdbgrids[-1].copyToArray(idxcoords, ijk=vdbgrids[-1].evalActiveVoxelBoundingBox()[0])
    mask = np.all(idxcoords!=vdbgrids[-1].background[0], axis=-1)
    idxcoords_valid = torch.from_numpy(idxcoords[mask]).float()  # (newdim, 3)
    if args.use_viewdirs:
        chunk_size = args.chunk // (args.samples_per_cell * args.projection_samples // 10)
    else:
        chunk_size = args.chunk // (args.samples_per_cell)
    output_sigma = []
    output_sh = []
    for i in tqdm(range(0, idxcoords_valid.shape[0], chunk_size)):
        chunk_inds = idxcoords_valid[i:i+chunk_size].unsqueeze(-2)   # (chunk_size, 1, 3)
        shape0 = chunk_inds.shape[0]
        points = chunk_inds + (torch.rand(shape0, args.samples_per_cell, 3)-0.5)  # (chunk_size, n_samples, 3)
        points = points.view(-1, 3)
        points = points * torch.tensor(transform.voxelSize()) + torch.tensor(transform.indexToWorld((0,0,0)))
        if not args.use_viewdirs:  # trained NeRF-SH/SG model returns rgb as coeffs
            rgb, sigma = nerf.eval_points_raw(points.cuda())
        else:  # vanilla NeRF model returns rgb, so we project them into coeffs (only SH supported)
            rgb, sigma = project_nerf_to_sh(nerf, args.sh_deg, points.cuda())
        rgb = rgb.reshape(shape0, args.samples_per_cell, 3, -1).mean(1).detach().cpu().numpy()  # (chunk_size, 3, 16)
        sigma = sigma.reshape(shape0, args.samples_per_cell).mean(1).detach().cpu().numpy()  # (chunk_size, )
        output_sigma.extend(sigma)
        output_sh.extend(rgb)
        del rgb, sigma
    print(transform.voxelSize())
    print(transform.indexToWorld((0,0,0)))
    output_sigma = np.array(output_sigma)
    output_sh = np.array(output_sh)
    arr_sigmas[mask] = output_sigma
    arr_shs[mask] = output_sh
    vdbgrids[-2].clear()
    vdbgrids[-2].copyFromArray(arr_sigmas)

    print("save to vdb")
    print(vdbgrids[-1].evalActiveVoxelBoundingBox())
    print(vdbgrids[-2].evalActiveVoxelBoundingBox())
    for vdbid in range(len(vdbgrids)-2):
        vdbgrids[vdbid].copyFromArray(np.ascontiguousarray(arr_shs[..., vdbid]))
        # print("???", arr_shs.flags['C_CONTIGUOUS'])   True
        # print("???", arr_shs[..., 0].flags['C_CONTIGUOUS'])    False!!!

    print("Time Used:", (time.time()-t)/60)
    print("----After Eval----")
    num = str(len(vdbgrids)-2)
    for vdbid in range(len(vdbgrids)-2):
        strid = str(vdbid+1)
        vdbgrids[vdbid]['name'] = '0'*(len(num)-len(strid))+strid
    vdbgrids[-2]['name'] = 'Sigma'
    vdbgrids[-1]['name'] = 'IndexCoord'

    savedir = osp.dirname(FLAGS.output)
    vdb.write(osp.join(savedir, "density.vdb"), grids=vdbgrids[-2:-1])
    vdb.write(osp.join(savedir, "color.vdb"), grids=vdbgrids[:-2])

        
def main(unused_argv):
    utils.set_random_seed(20200823)
    utils.update_flags(FLAGS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("* Using ", device)
    nerf = models.get_model_state(FLAGS, device=device, restore=True)
    nerf.eval()
    print("* Load Successfully!")

    assert FLAGS.sh_deg > 0
    data_format = f'SH{(FLAGS.sh_deg + 1) ** 2}'
    print('* Detected format:', data_format)

    base_dir = osp.dirname(FLAGS.output)
    if base_dir:
        os.makedirs(base_dir, exist_ok=True)
    assert FLAGS.data_dir  # Dataset is required now
    dataset = datasets.get_dataset("train", FLAGS)
    if FLAGS.bbox_from_data:
        assert dataset.bbox is not None  # Dataset must be NSVF
        center = (dataset.bbox[:3] + dataset.bbox[3:6]) * 0.5
        radius = (dataset.bbox[3:6] - dataset.bbox[:3]) * 0.5 * FLAGS.data_bbox_scale
        print('Bounding box from data: c', center, 'r', radius)
    else:
        center = list(map(float, FLAGS.center.split()))
        if len(center) == 1:
            center *= 3
        radius = list(map(float, FLAGS.radius.split()))
        if len(radius) == 1:
            radius *= 3

    if FLAGS.autoscale:
        print('Before Autoscale: center ', center, 'radius', radius)
        center, radius = auto_scale(FLAGS, center, radius, nerf)
        print('Autoscale result center', center, 'radius', radius)


    radius = [r * FLAGS.bbox_scale for r in radius]
    if FLAGS.bbox_cube:
        radius = [max(radius)] * 3

    num_rgb_channels = FLAGS.num_rgb_channels
    if FLAGS.sh_deg >= 0:
        assert FLAGS.sg_dim == -1, (
            "You can only use up to one of: SH or SG")
        num_rgb_channels *= (FLAGS.sh_deg + 1) ** 2
    data_dim =  1 + num_rgb_channels  # alpha + rgb
    print('data dim is', data_dim)

    # the last grid stores sigma, the others store SHs
    grids = [vdb.Vec3SGrid() for _ in range((FLAGS.sh_deg + 1) ** 2)]  # store SHs
    grids.append(vdb.FloatGrid())  # store sigma
    reso = 2 ** (FLAGS.init_grid_depth + 1)
    grids.append(vdb.Vec3SGrid([reso, reso, reso]))  # store world coords
    #  the index will not reach this value since usually [-reso/2, reso/2-1]

    metadata = {'data_dim': data_dim, 'data_format': data_format, 'device_type': device,
                'center': center, 'radius': radius}
    grids[-1].updateMetadata(metadata)
    print(grids[-1].metadata)
    
    step1(FLAGS, grids, nerf, dataset)
    step2(FLAGS, grids, nerf)



if __name__ == "__main__":
    app.run(main)