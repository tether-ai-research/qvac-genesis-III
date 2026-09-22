"""Runtime-selected OpenCompass models for Genesis III."""

import json
import os
from concurrent.futures import ThreadPoolExecutor

from opencompass.evaluator import GenericLLMEvaluator
from opencompass.models import (
    HuggingFaceBaseModel,
    HuggingFacewithChatTemplate,
    OpenAI,
    OpenAISDK,
)
from opencompass.registry import MODELS
from tqdm import tqdm


def _generation_kwargs(temperature):
    if temperature > 0:
        return {"do_sample": True, "temperature": temperature}
    return {"do_sample": False}


def _model_kwargs(revision):
    kwargs = {
        "device_map": "auto",
        "torch_dtype": "torch.bfloat16",
        "trust_remote_code": False,
        "low_cpu_mem_usage": True,
    }
    if revision:
        kwargs["revision"] = revision
    return kwargs


def _tokenizer_kwargs(revision):
    kwargs = {"trust_remote_code": False}
    if revision:
        kwargs["revision"] = revision
    return kwargs


class _BoundedConcurrencyMixin:
    max_workers: int

    def generate(self, inputs, max_out_len=512, temperature=0.7, **kwargs):
        if self.temperature is not None:
            temperature = self.temperature
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(
                tqdm(
                    executor.map(
                        self._generate,
                        inputs,
                        [max_out_len] * len(inputs),
                        [temperature] * len(inputs),
                    ),
                    total=len(inputs),
                    desc="Inferencing",
                )
            )


@MODELS.register_module()
class GenesisIIIEnvOpenAI(_BoundedConcurrencyMixin, OpenAI):
    """Candidate client for an OpenAI-compatible vLLM server."""

    def __init__(self, **kwargs):
        generation = json.loads(
            os.environ.get("GENERATION_PARAMS", '{"temperature": 0.0}')
        )
        temperature = float(generation.pop("temperature", 0.0))
        configured_workers = kwargs.pop("max_workers", 4096)
        self.max_workers = int(
            os.environ.get("MAX_WORKERS", str(configured_workers))
        )
        kwargs.update(
            path=os.environ["MODEL_PATH"],
            openai_api_base=(
                os.environ.get(
                    "VLLM_MODEL_API_BASE", "http://127.0.0.1:5000/v1"
                )
                + "/chat/completions"
            ),
            key=os.environ.get("VLLM_MODEL_API_KEY", "EMPTY"),
            max_seq_len=int(os.environ.get("MAX_SEQ_LEN", "4096")),
            query_per_second=int(os.environ.get("QUERY_PER_SECOND", "256")),
            temperature=temperature,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                **generation,
            },
        )
        super().__init__(**kwargs)


@MODELS.register_module()
class GenesisIIIEnvHuggingFace(HuggingFaceBaseModel):
    """In-process Transformers candidate model using plain completion prompts."""

    def __init__(self, **kwargs):
        generation = json.loads(
            os.environ.get("GENERATION_PARAMS", '{"temperature": 0.0}')
        )
        temperature = float(generation.pop("temperature", 0.0))
        revision = os.environ.get("MODEL_REVISION", "")
        kwargs.update(
            path=os.environ["MODEL_PATH"],
            max_seq_len=int(os.environ.get("MAX_SEQ_LEN", "4096")),
            model_kwargs=_model_kwargs(revision),
            tokenizer_kwargs=_tokenizer_kwargs(revision),
            generation_kwargs={
                **_generation_kwargs(temperature),
                **generation,
            },
        )
        super().__init__(**kwargs)


@MODELS.register_module()
class GenesisIIIOpenAISDK(_BoundedConcurrencyMixin, OpenAISDK):
    """OpenAI-compatible judge client with bounded request concurrency."""

    def __init__(self, max_workers=2048, **kwargs):
        super().__init__(**kwargs)
        self.max_workers = int(max_workers)


def build_judge_config(backend=None):
    """Resolve a judge config without loading model weights."""
    backend = backend or os.environ.get("GENESIS_JUDGE_BACKEND", "vllm")
    temperature = float(os.environ.get("OC_JUDGE_TEMPERATURE", "0.01"))
    max_out_len = int(os.environ.get("OC_JUDGE_MAX_OUT_LEN", "4096"))
    max_seq_len = int(os.environ.get("OC_JUDGE_MAX_SEQ_LEN", "32768"))

    if backend == "transformers":
        revision = os.environ.get("OC_JUDGE_REVISION", "")
        return dict(
                type=HuggingFacewithChatTemplate,
                path=os.environ["OC_JUDGE_MODEL"],
                model_kwargs=_model_kwargs(revision),
                tokenizer_kwargs=_tokenizer_kwargs(revision),
                generation_kwargs=_generation_kwargs(temperature),
                batch_size=int(
                    os.environ.get("OC_JUDGE_HF_BATCH_SIZE", "1")
                ),
                max_out_len=max_out_len,
                max_seq_len=max_seq_len,
                mode="none",
            )
    if backend == "vllm":
        batch_size = int(os.environ.get("OC_JUDGE_BATCH_SIZE", "2048"))
        return dict(
            type=GenesisIIIOpenAISDK,
            path=os.environ["OC_JUDGE_MODEL"],
            key=os.environ.get("OC_JUDGE_API_KEY", "EMPTY"),
            openai_api_base=os.environ.get(
                "OC_JUDGE_API_BASE", "http://127.0.0.1:4000/v1"
            ),
            meta_template=dict(
                round=[
                    dict(role="HUMAN", api_role="HUMAN"),
                    dict(role="BOT", api_role="BOT", generate=True),
                ]
            ),
            query_per_second=int(os.environ.get("OC_JUDGE_QPS", "256")),
            batch_size=batch_size,
            max_workers=int(
                os.environ.get("OC_JUDGE_MAX_WORKERS", str(batch_size))
            ),
            temperature=temperature,
            tokenizer_path="gpt-4o-2024-05-13",
            verbose=True,
            max_out_len=max_out_len,
            max_seq_len=max_seq_len,
        )
    raise ValueError(f"Unsupported judge backend: {backend}")


class GenesisIIIGenericLLMEvaluator(GenericLLMEvaluator):
    """Select the vLLM/API or local Transformers judge at runtime."""

    def __init__(self, *args, judge_cfg=None, **kwargs):
        if judge_cfg:
            raise ValueError(
                "Configure the Genesis III judge through runtime variables."
            )

        super().__init__(*args, judge_cfg=build_judge_config(), **kwargs)
