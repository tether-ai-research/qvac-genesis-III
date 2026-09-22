"""In-process Transformers candidate model."""

from genesis_iii_eval import GenesisIIIEnvHuggingFace


models = [
    dict(
        type=GenesisIIIEnvHuggingFace,
        abbr="genesis-iii-transformers",
        max_seq_len=4096,
        max_out_len=256,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
