from mmengine.config import read_base

with read_base():
    from .genesis_iii import gpqa_diamond_datasets  # noqa: F401
