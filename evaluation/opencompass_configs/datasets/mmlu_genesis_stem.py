from mmengine.config import read_base

with read_base():
    from .genesis_iii import mmlu_genesis_stem_datasets  # noqa: F401
