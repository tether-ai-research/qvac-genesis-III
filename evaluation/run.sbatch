#!/usr/bin/env bash
#SBATCH --job-name=genesis-iii-eval
#SBATCH --output=genesis-iii-eval-%j.out
#SBATCH --error=genesis-iii-eval-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --exclusive

set -euo pipefail

submit_dir="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -x "$submit_dir/run.sh" ]]; then
    eval_dir="$submit_dir"
elif [[ -x "$submit_dir/evaluation/run.sh" ]]; then
    eval_dir="$submit_dir/evaluation"
else
    echo "Submit from the repository root or evaluation directory." >&2
    exit 1
fi

cd "$eval_dir"
exec ./run.sh "$@"
