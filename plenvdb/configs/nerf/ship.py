_base_ = '../default.py'

expname = 'vdb_ship'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='/data2/hyan/data/nerf_synthetic/ship',
    dataset_type='blender',
    white_bkgd=True,
)

