_base_ = '../default.py'

expname = 'vdb_lego'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='/data2/hyan/data/nerf_synthetic/lego',
    dataset_type='blender',
    white_bkgd=True,
)

