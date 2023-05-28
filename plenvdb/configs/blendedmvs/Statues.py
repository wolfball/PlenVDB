_base_ = '../default.py'

expname = 'vdb_Statues'
basedir = './logs/blended_mvs'

data = dict(
    datadir='./data/BlendedMVS/Statues/',
    dataset_type='blendedmvs',
    inverse_y=True,
    white_bkgd=True,
)

