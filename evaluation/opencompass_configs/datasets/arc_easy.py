from mmengine.config import read_base

with read_base():
    from .genesis_iii import arc_easy_datasets  # noqa: F401
