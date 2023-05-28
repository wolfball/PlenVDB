_base_ = '../default.py'

expname = 'vdb_ficus'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='/data2/hyan/data/nerf_synthetic/ficus',
    dataset_type='blender',
    white_bkgd=True,
)

