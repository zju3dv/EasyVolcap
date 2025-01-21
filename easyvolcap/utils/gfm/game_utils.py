from dataclasses import dataclass


@dataclass
class GameConfig:
    type: str
    near: float
    far: float
    inverse: bool = False
    fov_type: str = 'vfov'
    delta: int = 0
    scale: float = 1.0


game_cfgs = {
    '2077': GameConfig(
        type='2077',
        near=0.02,
        far=1e5,
        delta=-2
    ),
    'wukong': GameConfig(
        type='wukong',
        near=0.05,
        far=1e5,
        delta=-1,
        scale=0.005,
        fov_type='hfov'
    ),
    'rdr2': GameConfig(
        type='rdr2',
        near=0.05,
        far=1e5,
        delta=-2
    ),
    'default': GameConfig(
        type='default',
        near=0.02,
        far=1e9,
        delta=0
    ),
}
