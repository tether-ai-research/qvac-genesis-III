#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENCOMPASS_COMMIT="dc8deb6af0d452c3134ce72693b460c1e1774ed6"
OPENCOMPASS_REPOSITORY="https://github.com/open-compass/opencompass.git"
OPENCOMPASS_DIR="${OPENCOMPASS_DIR:-$ROOT/vendor/opencompass}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
BACKEND="vllm"
INSTALL=true

usage() {
    cat <<'EOF'
Usage: ./setup.sh [--backend vllm|transformers] [--no-install]

vllm         Install the paper backend and Transformers support (default).
transformers Install OpenCompass without the optional vLLM package.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            BACKEND="${2:?--backend requires vllm or transformers}"
            shift 2
            ;;
        --no-install)
            INSTALL=false
            shift
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

if [[ ! -f "$OPENCOMPASS_DIR/run.py" ]]; then
    if [[ "$OPENCOMPASS_DIR" != "$ROOT/vendor/opencompass" ]]; then
        echo "OPENCOMPASS_DIR does not contain run.py: $OPENCOMPASS_DIR" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$OPENCOMPASS_DIR")"
    git init "$OPENCOMPASS_DIR"
    git -C "$OPENCOMPASS_DIR" remote add origin "$OPENCOMPASS_REPOSITORY"
    git -C "$OPENCOMPASS_DIR" fetch --depth 1 origin "$OPENCOMPASS_COMMIT"
    git -C "$OPENCOMPASS_DIR" checkout --detach FETCH_HEAD
fi

actual_commit="$(git -C "$OPENCOMPASS_DIR" rev-parse HEAD)"
if [[ "$actual_commit" != "$OPENCOMPASS_COMMIT" ]]; then
    echo "Expected OpenCompass $OPENCOMPASS_COMMIT, found $actual_commit" >&2
    exit 1
fi

mkdir -p "$OPENCOMPASS_DIR/data/gpqa"
gpqa_file="$OPENCOMPASS_DIR/data/gpqa/gpqa_diamond.csv"
gpqa_url="https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
gpqa_sha256="41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305"
if [[ ! -f "$gpqa_file" ]]; then
    curl --fail --location "$gpqa_url" --output "$gpqa_file"
fi
echo "$gpqa_sha256  $gpqa_file" | sha256sum --check -

if [[ "$INSTALL" == "true" ]]; then
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/python" -m pip install --upgrade pip
    "$VENV_DIR/bin/python" -m pip install \
        --constraint "$ROOT/constraints.txt" "PyYAML>=6.0,<7" pytest

    patch_file="$ROOT/patches/opencompass-0.4.2-python312.patch"
    if ! git -C "$OPENCOMPASS_DIR" apply --reverse --check "$patch_file" \
        2>/dev/null; then
        git -C "$OPENCOMPASS_DIR" apply --check "$patch_file"
        git -C "$OPENCOMPASS_DIR" apply "$patch_file"
    fi

    opencompass_target="$OPENCOMPASS_DIR"
    if [[ "$BACKEND" == "vllm" ]]; then
        opencompass_target="$OPENCOMPASS_DIR[vllm]"
    fi
    "$VENV_DIR/bin/python" -m pip install \
        --constraint "$ROOT/constraints.txt" -e "$opencompass_target"
    "$VENV_DIR/bin/python" -m pip install -e "$ROOT"

    "$VENV_DIR/bin/python" -c \
        "import genesis_iii_eval, opencompass, transformers; print('Imports OK:', opencompass.__version__, transformers.__version__)"
    if [[ "$BACKEND" == "vllm" ]]; then
        "$VENV_DIR/bin/python" -c \
            "import vllm; print('vLLM:', vllm.__version__)"
    fi
fi

cat <<EOF
OpenCompass: $OPENCOMPASS_DIR
Environment: $VENV_DIR
Backend:     $BACKEND

Run from a GPU machine:
  ./run.sh --backend $BACKEND
EOF
