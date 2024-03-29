_base_ = './nerf_unbounded_default.py'

expname = 'vdb_stump_unbounded'

data = dict(
    datadir='/data2/hyan/data/stump',
    factor=4,
    movie_render_kwargs=dict(
        shift_x=0.0,  # positive right
        shift_y=-0.2, # negative down
        shift_z=0,
        scale_r=0.8,
        pitch_deg=-20, # negative look downward
    ),
)

