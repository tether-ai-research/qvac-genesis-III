"""OpenAI-compatible vLLM candidate client."""

from genesis_iii_eval import GenesisIIIEnvOpenAI


models = [
    dict(
        type=GenesisIIIEnvOpenAI,
        abbr="genesis-iii-vllm",
        max_seq_len=4096,
        max_out_len=256,
        query_per_second=256,
        batch_size=4096,
        max_workers=4096,
        retry=5,
        mode="none",
    )
]
