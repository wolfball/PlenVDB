_base_ = '../default.py'

expname = 'vdb_chair'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='/data2/hyan/data/nerf_synthetic/chair',
    dataset_type='blender',
    white_bkgd=True,
)

