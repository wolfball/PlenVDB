_base_ = '../default.py'

expname = 'vdb_Fountain'
basedir = './logs/blended_mvs'

data = dict(
    datadir='./data/BlendedMVS/Fountain/',
    dataset_type='blendedmvs',
    inverse_y=True,
    white_bkgd=False,
)

