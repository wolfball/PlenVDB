_base_ = '../default.py'

expname = 'vdb_hotdog'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='/data2/hyan/data/nerf_synthetic/hotdog',
    dataset_type='blender',
    white_bkgd=True,
)

