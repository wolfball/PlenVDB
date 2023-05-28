_base_ = '../default.py'

expname = 'vdb_drums'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='/data2/hyan/data/nerf_synthetic/drums',
    dataset_type='blender',
    white_bkgd=True,
)

