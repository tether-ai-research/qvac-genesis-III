from pathlib import Path

from mmengine import Config

from genesis_iii_eval import build_judge_config


ROOT = Path(__file__).resolve().parents[1]


def test_dataset_groups_cover_paper_tasks():
    config = Config.fromfile(
        ROOT / "opencompass_configs/datasets/genesis_iii.py"
    )
    assert len(config.mmlu_genesis_stem_datasets) == 19
    assert len(config.arc_challenge_datasets) == 1
    assert len(config.arc_easy_datasets) == 1
    assert len(config.gpqa_diamond_datasets) == 1
    assert len(config.genesis_iii_datasets) == 22
    assert len(config.smoke_datasets) == 1
    assert (
        config.smoke_datasets[0]["reader_cfg"]["test_range"] == "[0:2]"
    )

    for dataset in config.genesis_iii_datasets:
        template = dataset["infer_cfg"]["prompt_template"]["template"]
        prompts = [turn["prompt"] for turn in template["round"]]
        assert all(not prompt.endswith(" ") for prompt in prompts)


def test_dataset_entrypoints_follow_opencompass_naming_contract():
    expected_sizes = {
        "smoke": 1,
        "arc_easy": 1,
        "arc_challenge": 1,
        "gpqa_diamond": 1,
        "mmlu_genesis_stem": 19,
    }
    for name, expected_size in expected_sizes.items():
        config = Config.fromfile(
            ROOT / f"opencompass_configs/datasets/{name}.py"
        )
        datasets = config[f"{name}_datasets"]
        assert len(datasets) == expected_size


def test_candidate_backend_configs_load():
    vllm = Config.fromfile(
        ROOT / "opencompass_configs/models/vllm_genesis_iii.py"
    )
    transformers = Config.fromfile(
        ROOT / "opencompass_configs/models/hf_genesis_iii.py"
    )
    assert vllm.models[0]["abbr"] == "genesis-iii-vllm"
    assert transformers.models[0]["abbr"] == "genesis-iii-transformers"
    assert transformers.models[0]["batch_size"] == 1


def test_judge_backend_selection(monkeypatch):
    monkeypatch.setenv("OC_JUDGE_MODEL", "judge")
    monkeypatch.setenv("OC_JUDGE_REVISION", "revision")

    vllm = build_judge_config("vllm")
    transformers = build_judge_config("transformers")

    assert vllm["type"].__name__ == "GenesisIIIOpenAISDK"
    assert transformers["type"].__name__ == "HuggingFacewithChatTemplate"
    assert transformers["model_kwargs"]["revision"] == "revision"
    assert transformers["batch_size"] == 1


def test_core_runner_is_scheduler_independent():
    runner = (ROOT / "run.sh").read_text()
    wrapper = (ROOT / "run.sbatch").read_text()
    assert "#SBATCH" not in runner
    assert "srun " not in runner
    assert 'exec ./run.sh "$@"' in wrapper


def test_public_setup_uses_official_opencompass():
    setup = (ROOT / "setup.sh").read_text()
    assert "https://github.com/open-compass/opencompass.git" in setup
    assert "Vitabile/qvac-research-evaluate" not in setup
