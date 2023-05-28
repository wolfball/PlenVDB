_base_ = '../default.py'

expname = 'vdb_materials'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='/data2/hyan/data/nerf_synthetic/materials',
    dataset_type='blender',
    white_bkgd=True,
)

