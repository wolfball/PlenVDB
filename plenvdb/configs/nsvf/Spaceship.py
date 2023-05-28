_base_ = '../default.py'

expname = 'vdb_Spaceship'
basedir = './logs/nsvf_synthetic'

data = dict(
    datadir='./data/Synthetic_NSVF/Spaceship',
    dataset_type='nsvf',
    inverse_y=True,
    white_bkgd=True,
)

