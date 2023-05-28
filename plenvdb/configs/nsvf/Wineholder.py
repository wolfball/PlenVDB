_base_ = '../default.py'

expname = 'vdb_Wineholder'
basedir = './logs/nsvf_synthetic'

data = dict(
    datadir='./data/Synthetic_NSVF/Wineholder',
    dataset_type='nsvf',
    inverse_y=True,
    white_bkgd=True,
)

