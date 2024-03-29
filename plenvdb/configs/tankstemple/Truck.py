_base_ = '../default.py'

expname = 'vdb_Truck'
basedir = './logs/tanks_and_temple'

data = dict(
    datadir='./data/TanksAndTemple/Truck',
    dataset_type='tankstemple',
    inverse_y=True,
    load2gpu_on_the_fly=True,
    white_bkgd=True,
)

coarse_train = dict(
    pervoxel_lr_downrate=2,
)

