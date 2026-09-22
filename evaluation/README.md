# Genesis III evaluation

This directory contains the generation-first, LLM-as-a-parser evaluation
implementation from the Genesis III paper. Candidate models generate free-form
responses, then CompassJudger extracts one final multiple-choice answer for
scoring. Any Hugging Face model or local checkpoint can be evaluated as a
candidate.

Benchmarks:

- ARC-Easy and ARC-Challenge
- GPQA Diamond
- 19 MMLU STEM domains

## Requirements

- Linux, Git, `curl`, and Python 3.10+ (validated with Python 3.12)
- Hugging Face access to the candidate and judge repositories
- CUDA GPUs with BF16 support


## Setup

The setup script checks out official OpenCompass 0.4.2 at its pinned commit,
verifies GPQA, creates a virtual environment, and installs this local
extension.

Paper/default backend:

```bash
./setup.sh --backend vllm
```

Transformers-only installation:

```bash
./setup.sh --backend transformers
```

Authenticate first with `hf auth login` while the repositories are private.

## Run without SLURM

`run.sh` contains no scheduler commands. Execute it on a standalone GPU host or
after entering an allocated worker:

```bash
./run.sh --backend vllm
```

The alternative backend runs both candidates and CompassJudger in-process with
Hugging Face Transformers:

```bash
./run.sh --backend transformers
```

Transformers is substantially slower for the 32B judge. vLLM remains the
recommended backend for the full paper evaluation.

Useful options:

```bash
# Two-example end-to-end smoke test
./run.sh --backend vllm --datasets smoke

# One benchmark and one model; --models matches substrings of the
# configured model IDs
./run.sh --backend transformers --models combined --datasets arc_easy

# Run or resume only one phase
./run.sh --phase infer --models combined
./run.sh --phase eval --models combined
```

Available benchmark groups are `mmlu_genesis_stem`, `arc_challenge`,
`arc_easy`, `gpqa_diamond`, and the two-example `smoke` group.

## Optional SLURM wrapper

`run.sbatch` only requests resources and invokes `run.sh`. It does not contain
evaluation logic and does not assume a site-specific partition:

```bash
sbatch --partition=<gpu-partition> run.sbatch --backend vllm
```

For a one-GPU Transformers smoke test on an 80 GB GPU:

```bash
sbatch --partition=<gpu-partition> --gres=gpu:1 \
  run.sbatch --backend transformers --models combined --datasets smoke
```

The same direct `run.sh` commands work inside an interactive allocation.

## Configuration

`config.yaml` is an annotated sample that ships with one model entry (the
combined Genesis III checkpoint). Edit it to evaluate your own models: add
one entry per candidate, each with an optional pinned Hub revision, and run
them one at a time or in selected subsets. The file also configures:

- the CompassJudger parser model and revision;
- context lengths and generation settings;
- the benchmark groups;
- vLLM tensor parallelism and request concurrency.

Use `--models` and `--datasets` to select subsets at launch instead of
editing the file. Set `OPENCOMPASS_DIR`, `PYTHON`, `OUTPUT_ROOT`, or
`LOG_ROOT` when custom locations are needed.

## Outputs

Each run preserves:

- `predictions/<model>/<dataset>.json`: prompts, raw candidate text, and gold
  answers;
- `results/<model>/<dataset>.json`: parser responses and extracted answers;
- `summary/summary_*.csv`: per-task metrics and counts;
- `run_metadata/`: effective YAML, package versions, OpenCompass commit, GPU
  information, and timestamp.

Aggregate a run's summaries:

```bash
python summarize_results.py outputs/vllm \
  --output metrics.json
```
