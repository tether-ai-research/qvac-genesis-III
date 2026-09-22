from mmengine.config import read_base

with read_base():
    from .genesis_iii import arc_challenge_datasets  # noqa: F401
