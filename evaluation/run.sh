#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$ROOT/config.yaml"
BACKEND="vllm"
PHASE="all"
MODEL_SELECTOR=""
DATASET_SELECTOR=""
OPENCOMPASS_DIR="${OPENCOMPASS_DIR:-$ROOT/vendor/opencompass}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
OUTPUT_ROOT=""
LOG_ROOT="${LOG_ROOT:-$ROOT/logs}"
SERVER_PID=""

usage() {
    cat <<'EOF'
Usage: ./run.sh [options]

Options:
  --backend vllm|transformers  Inference backend for candidates and judge.
  --config PATH                Evaluation YAML (default: config.yaml).
  --phase all|infer|eval       Run both phases or one phase.
  --models LIST                Comma-separated IDs or substrings (for example: fa).
  --datasets LIST              Comma-separated benchmark groups (for example: arc_easy).
  --output PATH                Output directory.
  -h, --help                   Show this help.

This command is scheduler-independent. Run it on a machine where the required
GPUs are already visible. SLURM users may submit run.sbatch instead.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            BACKEND="${2:?--backend requires a value}"
            shift 2
            ;;
        --config)
            CONFIG="$(realpath "${2:?--config requires a path}")"
            shift 2
            ;;
        --phase)
            PHASE="${2:?--phase requires a value}"
            shift 2
            ;;
        --models)
            MODEL_SELECTOR="${2:?--models requires a value}"
            shift 2
            ;;
        --datasets)
            DATASET_SELECTOR="${2:?--datasets requires a value}"
            shift 2
            ;;
        --output)
            OUTPUT_ROOT="$(realpath -m "${2:?--output requires a path}")"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$BACKEND" != "vllm" && "$BACKEND" != "transformers" ]]; then
    echo "Unsupported backend: $BACKEND" >&2
    exit 2
fi
if [[ "$PHASE" != "all" && "$PHASE" != "infer" && "$PHASE" != "eval" ]]; then
    echo "Unsupported phase: $PHASE" >&2
    exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment not found: $PYTHON; run ./setup.sh first." >&2
    exit 1
fi
if [[ ! -f "$OPENCOMPASS_DIR/run.py" ]]; then
    echo "OpenCompass checkout not found: $OPENCOMPASS_DIR" >&2
    exit 1
fi
if [[ "$BACKEND" == "vllm" ]] && ! "$PYTHON" -c "import vllm" 2>/dev/null; then
    echo "vLLM is not installed; rerun ./setup.sh --backend vllm." >&2
    exit 1
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/$BACKEND}"
STATE_DIR="$(mktemp -d "$ROOT/.run-state.XXXXXX")"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

cleanup() {
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    rm -rf "$STATE_DIR"
}
trap cleanup EXIT INT TERM

"$PYTHON" - "$CONFIG" "$MODEL_SELECTOR" "$DATASET_SELECTOR" "$STATE_DIR" <<'PY'
import json
import pathlib
import shlex
import sys

import yaml

config_path, model_selector, dataset_selector, state_path = sys.argv[1:]
config = yaml.safe_load(pathlib.Path(config_path).read_text())
state = pathlib.Path(state_path)

models = config["models"]
if model_selector:
    selectors = [item.strip() for item in model_selector.split(",") if item.strip()]
    models = [
        model
        for model in models
        if any(selector in model["id"] for selector in selectors)
    ]
if not models:
    raise SystemExit("No models matched --models")

datasets = config["benchmarks"]
if dataset_selector:
    requested = [
        item.strip() for item in dataset_selector.split(",") if item.strip()
    ]
    valid_datasets = set(datasets) | set(config.get("smoke_benchmarks", []))
    unknown = sorted(set(requested) - valid_datasets)
    if unknown:
        raise SystemExit(f"Unknown benchmark groups: {', '.join(unknown)}")
    datasets = requested

(state / "models.tsv").write_text(
    "".join(f"{model['id']}\t{model.get('revision', '')}\n" for model in models)
)
(state / "datasets.txt").write_text("\n".join(datasets) + "\n")

runtime = config["runtime"]
values = {
    "MODEL_TEMPERATURE": config["generation"]["temperature"],
    "MAX_SEQ_LEN": config["generation"]["max_input_tokens"],
    "MAX_OUT_LEN": config["generation"]["max_output_tokens"],
    "JUDGE_MODEL": config["judge"]["model"],
    "JUDGE_REVISION": config["judge"].get("revision", ""),
    "JUDGE_TEMPERATURE": config["judge"]["temperature"],
    "JUDGE_MAX_LEN": config["judge"]["max_input_tokens"],
    "JUDGE_MAX_OUT_LEN": config["judge"]["max_output_tokens"],
    "MODEL_PORT": runtime["candidate_port"],
    "JUDGE_PORT": runtime["judge_port"],
    "MODEL_TP": runtime["candidate_tensor_parallel_size"],
    "JUDGE_TP": runtime["judge_tensor_parallel_size"],
    "GPU_MEMORY": runtime["gpu_memory_utilization"],
    "MAX_WORKERS": runtime["candidate_workers"],
    "JUDGE_WORKERS": runtime["judge_workers"],
    "QUERY_PER_SECOND": runtime["query_per_second"],
    "HF_BATCH_SIZE": runtime.get("transformers_batch_size", 1),
}
(state / "settings.env").write_text(
    "".join(f"{key}={shlex.quote(str(value))}\n" for key, value in values.items())
)
PY

# shellcheck disable=SC1090
source "$STATE_DIR/settings.env"
mapfile -t DATASETS < "$STATE_DIR/datasets.txt"
MODELS=()
MODEL_REVISIONS=()
while IFS=$'\t' read -r model revision; do
    MODELS+=("$model")
    MODEL_REVISIONS+=("$revision")
done < "$STATE_DIR/models.tsv"

wait_for_server() {
    local port="$1"
    for _ in $(seq 1 180); do
        if curl --silent --fail --max-time 5 \
            --header "Authorization: Bearer $VLLM_MODEL_API_KEY" \
            "http://127.0.0.1:$port/v1/models" >/dev/null; then
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "vLLM exited before becoming ready." >&2
            return 1
        fi
        sleep 10
    done
    echo "Timed out waiting for vLLM on port $port." >&2
    return 1
}

start_server() {
    local model="$1"
    local revision="$2"
    local port="$3"
    local max_len="$4"
    local tensor_parallel="$5"
    local memory="$6"
    local log_file="$7"
    shift 7

    local revision_args=()
    if [[ -n "$revision" ]]; then
        revision_args=(--revision "$revision")
    fi
    "$PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --served-model-name "$model" \
        --host 127.0.0.1 \
        --port "$port" \
        --api-key "$VLLM_MODEL_API_KEY" \
        --tensor-parallel-size "$tensor_parallel" \
        --dtype bfloat16 \
        --gpu-memory-utilization "$memory" \
        --max-model-len "$max_len" \
        "${revision_args[@]}" \
        "$@" >"$log_file" 2>&1 &
    SERVER_PID=$!
    wait_for_server "$port"
}

stop_server() {
    [[ -z "$SERVER_PID" ]] && return
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
}

run_opencompass() {
    (
        cd "$OPENCOMPASS_DIR"
        "$PYTHON" run.py "$@"
    )
}

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_MODEL_API_KEY=EMPTY
export MAX_SEQ_LEN MAX_OUT_LEN MAX_WORKERS QUERY_PER_SECOND HF_BATCH_SIZE
export GENERATION_PARAMS="{\"temperature\": $MODEL_TEMPERATURE}"
export GENESIS_JUDGE_BACKEND="$BACKEND"

metadata="$OUTPUT_ROOT/run_metadata"
mkdir -p "$metadata"
cp "$CONFIG" "$metadata/config.yaml"
"$PYTHON" -m pip freeze > "$metadata/python-environment.txt"
git -C "$OPENCOMPASS_DIR" rev-parse HEAD > "$metadata/opencompass-commit.txt"
date --iso-8601=seconds > "$metadata/run-started-at.txt"
if command -v nvidia-smi >/dev/null; then
    nvidia-smi > "$metadata/nvidia-smi.txt"
fi

if [[ "$PHASE" == "all" || "$PHASE" == "infer" ]]; then
    echo "Phase 1/2: candidate generation with $BACKEND"
    for index in "${!MODELS[@]}"; do
        model="${MODELS[$index]}"
        revision="${MODEL_REVISIONS[$index]}"
        slug="${model##*/}"
        export MODEL_PATH="$model"
        export MODEL_REVISION="$revision"

        model_config="hf_genesis_iii"
        if [[ "$BACKEND" == "vllm" ]]; then
            model_config="vllm_genesis_iii"
            export VLLM_MODEL_API_BASE="http://127.0.0.1:$MODEL_PORT/v1"
            start_server \
                "$model" "$revision" "$MODEL_PORT" "$MAX_SEQ_LEN" \
                "$MODEL_TP" "$GPU_MEMORY" "$LOG_ROOT/${slug}-vllm.log" \
                --chat-template "$ROOT/simple_completion.jinja"
        fi

        run_opencompass \
            --models "$model_config" \
            --datasets "${DATASETS[@]}" \
            --config-dir "$ROOT/opencompass_configs" \
            --work-dir "$OUTPUT_ROOT/${slug}-${BACKEND}" \
            --mode infer \
            --debug
        stop_server
    done
fi

if [[ "$PHASE" == "all" || "$PHASE" == "eval" ]]; then
    echo "Phase 2/2: answer extraction with $BACKEND"
    export OC_JUDGE_MODEL="$JUDGE_MODEL"
    export OC_JUDGE_REVISION="$JUDGE_REVISION"
    export OC_JUDGE_API_BASE="http://127.0.0.1:$JUDGE_PORT/v1"
    export OC_JUDGE_API_KEY=EMPTY
    export OC_JUDGE_BATCH_SIZE="$JUDGE_WORKERS"
    export OC_JUDGE_MAX_WORKERS="$JUDGE_WORKERS"
    export OC_JUDGE_HF_BATCH_SIZE="$HF_BATCH_SIZE"
    export OC_JUDGE_QPS="$QUERY_PER_SECOND"
    export OC_JUDGE_TEMPERATURE="$JUDGE_TEMPERATURE"
    export OC_JUDGE_MAX_SEQ_LEN="$JUDGE_MAX_LEN"
    export OC_JUDGE_MAX_OUT_LEN="$JUDGE_MAX_OUT_LEN"

    if [[ "$BACKEND" == "vllm" ]]; then
        start_server \
            "$JUDGE_MODEL" "$JUDGE_REVISION" "$JUDGE_PORT" \
            "$JUDGE_MAX_LEN" "$JUDGE_TP" "0.9" \
            "$LOG_ROOT/compassjudger-vllm.log" \
            --enable-prefix-caching
    fi

    model_config="hf_genesis_iii"
    if [[ "$BACKEND" == "vllm" ]]; then
        model_config="vllm_genesis_iii"
    fi
    for index in "${!MODELS[@]}"; do
        model="${MODELS[$index]}"
        revision="${MODEL_REVISIONS[$index]}"
        slug="${model##*/}"
        export MODEL_PATH="$model"
        export MODEL_REVISION="$revision"
        run_opencompass \
            --models "$model_config" \
            --datasets "${DATASETS[@]}" \
            --config-dir "$ROOT/opencompass_configs" \
            --work-dir "$OUTPUT_ROOT/${slug}-${BACKEND}" \
            --mode eval \
            --reuse latest \
            --debug
    done
    stop_server
fi

echo "Evaluation complete: $OUTPUT_ROOT"
