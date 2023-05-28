_base_ = '../default.py'

expname = 'vdb_mic'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='/data2/hyan/data/nerf_synthetic/mic',
    dataset_type='blender',
    white_bkgd=True,
)

